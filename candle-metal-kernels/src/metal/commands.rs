use crate::metal::{
    BlitCommandEncoder, CommandBuffer, CommandSemaphore, CommandStatus, ComputeCommandEncoder,
};
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Accumulated timing breakdown for flush_and_wait_fast calls.
/// Used to diagnose GPU↔CPU sync overhead in inference hot paths.
#[derive(Default)]
struct SyncTimings {
    count: u64,
    sem_us: u64,
    lock_us: u64,
    commit_us: u64,
    wait_us: u64,
    total_us: u64,
}

thread_local! {
    static SYNC_TIMINGS: RefCell<SyncTimings> = RefCell::new(SyncTimings::default());
}

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;
const DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE: usize = 5;

/// Creates a new command buffer from the queue with an attached semaphore for tracking its state.
pub fn create_command_buffer(
    command_queue: &CommandQueue,
    semaphore: Arc<CommandSemaphore>,
) -> Result<CommandBuffer, MetalKernelError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer::new(raw, semaphore))
        .ok_or(MetalKernelError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
}

/// A pool entry containing a command buffer, its usage count, and synchronization primitives.
/// The `state` mutex guards the current buffer and the in-flight list for coherent updates.
/// `compute_count` and `semaphore` remain accessible without locking for selection/coordination.
pub struct CommandBufferEntry {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    semaphore: Arc<CommandSemaphore>,
}

pub struct Commands {
    /// Maintains a pool of command buffers, allowing
    /// the pool to balance load across multiple buffers and improve GPU utilization.
    /// Can be shared across threads safely.
    pool: Vec<Arc<CommandBufferEntry>>,
    /// Single command queue for the entire device.
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        let pool_size = match std::env::var("CANDLE_METAL_COMMAND_POOL_SIZE") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE),
            _ => DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE,
        };

        let pool = (0..pool_size)
            .map(|_| Self::create_pool_entry(&command_queue))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            pool,
            command_queue,
            compute_per_buffer,
        })
    }

    fn create_pool_entry(
        command_queue: &CommandQueue,
    ) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(command_queue, Arc::clone(&semaphore))?;

        Ok(Arc::new(CommandBufferEntry {
            state: Mutex::new(EntryState {
                current: cb,
                in_flight: Vec::new(),
            }),
            compute_count: AtomicUsize::new(0),
            semaphore,
        }))
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |cb| cb.compute_command_encoder())
    }

    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |cb| cb.blit_command_encoder())
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    // Selects an entry from the pool using a two-phase strategy:
    /// 1. Try non-blocking: find any available buffer without waiting
    /// 2. Fallback: select the least-loaded buffer and wait for availability
    fn select_entry(&self) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        // Phase 1: Try to find an available buffer without blocking
        for entry in &self.pool {
            if let Ok(mut status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Available) {
                    *status = CommandStatus::Encoding;
                    return Ok(Arc::clone(entry));
                }
            }
        }

        // Phase 2: Select the buffer with the most work and wait for it
        let entry = self
            .pool
            .iter()
            .max_by_key(|e| e.compute_count.load(Ordering::Acquire))
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Command buffer pool is empty".to_string(),
            ))?;

        let entry = Arc::clone(entry);
        {
            let mut guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            *guard = CommandStatus::Encoding;
        }

        Ok(entry)
    }

    /// Creates an encoder from the selected entry, recycling the buffer if needed.
    /// When recycling, the old committed buffer is moved to `in_flight` so we can later wait on it.
    fn finalize_entry<F, E>(
        &self,
        entry: Arc<CommandBufferEntry>,
        create_encoder: F,
    ) -> Result<(bool, E), MetalKernelError>
    where
        F: FnOnce(&mut CommandBuffer) -> E,
    {
        let mut state = entry.state.lock()?;

        let count = entry.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&entry, &mut state, 1)?;
        }

        let encoder = create_encoder(&mut state.current);

        Ok((flush, encoder))
    }

    /// Flushes all buffers and waits for their completion.
    /// Commits any pending work on the current buffers, moves them to in-flight,
    /// then waits on all in-flight buffers including those from prior recycles.
    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            // Under state lock, commit current if it has pending work and swap to a fresh one.
            let to_wait: Vec<CommandBuffer> = {
                // Ensure no active encoder is still encoding on this entry.
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));

                let mut state = entry.state.lock()?;

                if entry.compute_count.load(Ordering::Acquire) > 0 {
                    self.commit_swap_locked(&entry, &mut state, 0)?;
                }

                // Drain `in_flight` into a local vec to wait without holding the lock.
                // Replaces `state.in_flight` with an empty vec and returns its previous contents.
                std::mem::take(&mut state.in_flight)
            };

            for cb in to_wait {
                Self::ensure_completed(&cb)?;
            }
        }

        Ok(())
    }

    /// Fast flush+wait optimized for single-threaded inference.
    ///
    /// Unlike `flush_and_wait()` which iterates ALL pool entries (acquiring semaphores
    /// and locks for each), this only touches entries that actually have pending work.
    /// For serial inference with 24+ sync points per token, this saves ~0.5мс per sync
    /// by skipping 4 empty pool entries.
    pub fn flush_and_wait_fast(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            // Quick atomic check — skip entries with no pending work (no lock needed)
            if entry.compute_count.load(Ordering::Acquire) == 0 {
                // Still need to check in_flight, but only if there might be something
                if let Ok(state) = entry.state.try_lock() {
                    if state.in_flight.is_empty() {
                        continue; // Nothing to do for this entry
                    }
                } else {
                    continue; // Someone else holds the lock, skip
                }
            }

            // This entry has work — do the full sync
            let t0 = std::time::Instant::now();
            let to_wait: Vec<CommandBuffer> = {
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));
                let t_sem = t0.elapsed();

                let mut state = entry.state.lock()?;
                let t_lock = t0.elapsed();

                if entry.compute_count.load(Ordering::Acquire) > 0 {
                    self.commit_swap_locked(&entry, &mut state, 0)?;
                }
                let t_commit = t0.elapsed();

                let result = std::mem::take(&mut state.in_flight);
                // Log timing breakdown (accumulate via thread_local)
                SYNC_TIMINGS.with(|t| {
                    let mut t = t.borrow_mut();
                    t.count += 1;
                    t.sem_us += t_sem.as_micros() as u64;
                    t.lock_us += (t_lock - t_sem).as_micros() as u64;
                    t.commit_us += (t_commit - t_lock).as_micros() as u64;
                });
                result
            };

            let t_pre_wait = t0.elapsed();
            for cb in to_wait {
                Self::ensure_completed(&cb)?;
            }
            let t_total = t0.elapsed();
            SYNC_TIMINGS.with(|t| {
                let mut t = t.borrow_mut();
                t.wait_us += (t_total - t_pre_wait).as_micros() as u64;
                t.total_us += t_total.as_micros() as u64;
            });
        }

        Ok(())
    }

    /// Get and reset accumulated sync timing stats.
    /// Returns: (count, sem_us, lock_us, commit_us, wait_us, total_us)
    pub fn take_sync_timings() -> (u64, u64, u64, u64, u64, u64) {
        SYNC_TIMINGS.with(|t| {
            let mut t = t.borrow_mut();
            let result = (
                t.count,
                t.sem_us,
                t.lock_us,
                t.commit_us,
                t.wait_us,
                t.total_us,
            );
            *t = SyncTimings::default();
            result
        })
    }

    /// Flushes all buffers without waiting for completion.
    /// Commits any pending work and moves current buffers to in-flight.
    pub fn flush(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            let _guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));

            let mut state = entry.state.lock()?;

            if entry.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(&entry, &mut state, 0)?;
            }
        }

        Ok(())
    }

    /// Статистика command buffer pool (best-effort, не atomic snapshot):
    /// `(entries, in_flight_total, encoding_entries, total_compute_count)`.
    ///
    /// Значения могут быть несогласованы между собой, т.к. lock на каждый entry
    /// захватывается и отпускается независимо. Используйте только для диагностики.
    pub fn pool_stats(&self) -> Result<(usize, usize, usize, usize), MetalKernelError> {
        let mut in_flight_total = 0usize;
        let mut encoding_entries = 0usize;
        let mut total_compute_count = 0usize;

        for entry in &self.pool {
            let state = entry.state.lock()?;
            in_flight_total += state.in_flight.len();
            drop(state);

            if let Ok(status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Encoding) {
                    encoding_entries += 1;
                }
            }

            total_compute_count += entry.compute_count.load(Ordering::Acquire);
        }

        Ok((
            self.pool.len(),
            in_flight_total,
            encoding_entries,
            total_compute_count,
        ))
    }

    /// Commit the current command buffer, swap in a fresh one, push the old into `in_flight`,
    /// and reset `compute_count` to `reset_to`.
    fn commit_swap_locked(
        &self,
        entry: &CommandBufferEntry,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        state.current.commit();
        let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        entry.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // TODO: Avoid redundant allocation before drop
        let _ = self.flush();
    }
}
