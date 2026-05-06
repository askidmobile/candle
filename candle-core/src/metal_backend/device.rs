use crate::{DType, Result};

#[cfg(feature = "ug")]
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{
    metal::{
        BlitCommandEncoder, Buffer, BufferMap, Commands, ComputeCommandEncoder, Device,
        MTLResourceOptions,
    },
    Kernels,
};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use super::MetalError;

/// Unique identifier for metal devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: Device,

    pub(crate) commands: Arc<RwLock<Commands>>,

    /// Simple allocator struct.
    /// The buffers are stored in size buckets since ML tends to use similar shapes over and over.
    /// We store the buffers in [`Arc`] because it's much faster than Obj-c internal ref counting
    /// (could be linked to FFI communication overhead).
    ///
    /// Whenever a buffer has a strong_count==1, we can reuse it, it means it was dropped in the
    /// graph calculation, and only we the allocator kept a reference to it, therefore it's free
    /// to be reused. However, in order for this to work, we need to guarantee the order of
    /// operation, so that this buffer is not being used by another kernel at the same time.
    /// Arc is the CPU reference count, it doesn't mean anything on the GPU side of things.
    ///
    /// Whenever we actually allocate a new buffer, we make a full sweep to clean up unused buffers
    /// (strong_count = 1).
    pub(crate) buffers: Arc<RwLock<BufferMap>>,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
    /// Last seed value set on this device.
    pub(crate) seed_value: Arc<RwLock<u64>>,
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
pub const RESOURCE_OPTIONS: MTLResourceOptions =
    objc2_metal::MTLResourceOptions(MTLResourceOptions::StorageModeShared.bits());
//| MTLResourceOptions::HazardTrackingModeUntracked.bits(),
//);

// Resource options used for `new_private_buffer`. This uses `private` where supported.
#[cfg(target_os = "ios")]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
#[cfg(not(target_os = "ios"))]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModePrivate;

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    #[cfg(all(feature = "ug", not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<ComputePipeline> {
        let mut buf = vec![];
        candle_ug::metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, None)
            .map_err(MetalError::from)?;
        let func = lib
            .get_function(func_name, None)
            .map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &Device {
        &self.device
    }

    fn drop_unused_buffers(&self) -> Result<()> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        for subbuffers in buffers.values_mut() {
            let newbuffers = subbuffers
                .iter()
                .filter(|s| Arc::strong_count(*s) > 1)
                .map(Arc::clone)
                .collect();
            *subbuffers = newbuffers;
        }
        Ok(())
    }

    /// Flush all unused buffers from the Metal buffer pool.
    ///
    /// Removes buffers with `strong_count == 1` (only the pool holds a reference)
    /// and removes empty size buckets. Call after model loading to reclaim memory
    /// used by intermediate buffers (dequantization temporaries, etc.).
    pub fn flush_buffers(&self) -> Result<()> {
        self.drop_unused_buffers()?;
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        buffers.retain(|_, v| !v.is_empty());
        Ok(())
    }

    /// Агрессивная очистка buffer pool.
    ///
    /// В отличие от `flush_buffers()` полностью очищает все size buckets
    /// после ожидания завершения in-flight GPU команд. Также очищает
    /// кэши скомпилированных Metal libraries и pipeline state objects.
    ///
    /// # Важные замечания
    ///
    /// - **Cold-start spike**: после вызова все Metal shaders будут перекомпилированы
    ///   при следующем использовании (~50-200мс на каждый source). Вызывайте только
    ///   при teardown движка, а не между инференсами.
    ///
    /// - **Живые Arc-ссылки**: буферы, на которые существуют `Arc`-ссылки из
    ///   живых тензоров (`MetalStorage`), не будут физически освобождены — только
    ///   удалены из pool. Новые аллокации создадут новые буферы вместо
    ///   переиспользования. Вызывайте после drop всех тензоров модели.
    pub fn purge_buffer_pool(&self) -> Result<()> {
        self.wait_until_completed()?;
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        buffers.clear();
        let _ = self.kernels.clear_caches();
        Ok(())
    }

    /// Returns buffer pool statistics: `(total_buffers, total_bytes, unused_buffers, unused_bytes)`.
    ///
    /// Unused buffers have `strong_count == 1` and can be reclaimed by [`flush_buffers`].
    pub fn buffer_pool_stats(&self) -> Result<(usize, usize, usize, usize)> {
        let buffers = self.buffers.read().map_err(MetalError::from)?;
        let mut total_count = 0usize;
        let mut total_bytes = 0usize;
        let mut unused_count = 0usize;
        let mut unused_bytes = 0usize;
        for subbuffers in buffers.values() {
            for buf in subbuffers {
                let len = buf.length();
                total_count += 1;
                total_bytes += len;
                if Arc::strong_count(buf) == 1 {
                    unused_count += 1;
                    unused_bytes += len;
                }
            }
        }
        Ok((total_count, total_bytes, unused_count, unused_bytes))
    }

    /// Статистика command pool:
    /// `(entries, in_flight_total, encoding_entries, total_compute_count)`.
    pub fn command_pool_stats(&self) -> Result<(usize, usize, usize, usize)> {
        let commands = self.commands.read().map_err(MetalError::from)?;
        Ok(commands.pool_stats().map_err(MetalError::from)?)
    }

    /// Размеры kernel caches: `(libraries, pipelines)`.
    pub fn kernel_cache_stats(&self) -> Result<(usize, usize)> {
        Ok(self.kernels.cache_stats().map_err(MetalError::from)?)
    }

    /// Сводная статистика runtime памяти Metal:
    /// `(current_allocated_bytes, recommended_max_working_set_bytes)`.
    pub fn metal_memory_stats(&self) -> (usize, usize) {
        (
            self.device.current_allocated_size(),
            self.device.recommended_max_working_set_size(),
        )
    }

    pub fn command_encoder(&self) -> Result<ComputeCommandEncoder> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        let (flush, command_encoder) = commands.command_encoder().map_err(MetalError::from)?;
        if flush {
            self.drop_unused_buffers()?
        }
        Ok(command_encoder)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandEncoder> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        let (flush, command_encoder) = commands.blit_command_encoder().map_err(MetalError::from)?;
        if flush {
            self.drop_unused_buffers()?
        }
        Ok(command_encoder)
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.wait_until_completed().map_err(MetalError::from)?;
        Ok(())
    }

    /// Fast sync optimized for single-threaded inference workloads.
    /// Only processes pool entries that have pending work, skipping empty ones.
    pub fn wait_until_completed_fast(&self) -> Result<()> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.flush_and_wait_fast().map_err(MetalError::from)?;
        Ok(())
    }

    /// Commit any pending command buffers to the GPU **without** waiting for
    /// completion. Enables CPU↔GPU pipelining when used inside hot loops
    /// (e.g. Yttri Qwen3.5 prefill periodic flush at every N layers): CPU
    /// continues encoding the next batch of ops while GPU executes the
    /// committed work in parallel.
    ///
    /// **Hazard warning.** `flush_no_wait` does NOT clean up the buffer pool
    /// or wait for in-flight work. The buffer pool may hand out a buffer
    /// (matched by size and `Arc::strong_count == 1`) that GPU is still
    /// writing to in a previous CB — leading to data races. Use ONLY inside
    /// a `with_shared_encoder` scope or when the caller can guarantee no
    /// pool reuse touches buffers from in-flight CBs.
    ///
    /// The caller must still call `wait_until_completed()` (or
    /// `wait_until_completed_fast()`) before reading GPU outputs from CPU.
    pub fn flush_no_wait(&self) -> Result<()> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.flush().map_err(MetalError::from)?;
        Ok(())
    }

    /// Run a closure inside a graph-compute scope where all Metal compute
    /// ops accumulate into a single command buffer (one `commit` at scope
    /// exit) instead of the default behaviour of auto-flushing every
    /// `compute_per_buffer` ops.
    ///
    /// Modeled on llama.cpp's `ggml_backend_metal_graph_compute`: a long
    /// sequence of ops (e.g. a model forward pass) is encoded into one
    /// CB. CPU encoding still runs serially, but the CPU↔GPU sync overhead
    /// from per-batch commits is removed. Buffer pool reuse stays safe
    /// because all ops live in the same CB and Metal's shared-storage
    /// hazards are enforced by hardware between encoders within a CB.
    ///
    /// Memory profile improves too: intermediate buffers go back to the
    /// pool as soon as their last `Tensor` is dropped (which happens
    /// continuously during encoding), so peak in-flight memory tracks
    /// the natural data-dependency depth of the graph rather than the
    /// raw layer count.
    ///
    /// Caveats:
    /// - The closure must complete its CPU side cleanly before any GPU
    ///   output is read; this function calls `wait_until_completed_fast`
    ///   on exit so the caller may read tensors right after.
    /// - Large CBs slow down the GPU's first-byte latency, so for very
    ///   small graphs (single-token decode) the default 50-op auto-flush
    ///   path is usually faster — keep `with_shared_encoder` for prefill.
    /// - Nesting is supported via thread_local stacking of the scope
    ///   limit (closure-local restore at exit).
    pub fn with_shared_encoder<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        // Effectively-infinite limit so finalize_entry never auto-flushes.
        // Real-world forward'ы у нас 200-300 ops; миллион даёт огромный запас.
        const SCOPE_LIMIT: usize = 1_000_000;

        let prev_limit = candle_metal_kernels::metal::commands::graph_scope_limit();
        candle_metal_kernels::metal::commands::set_graph_scope_limit(SCOPE_LIMIT);

        struct ScopeGuard {
            prev: usize,
        }
        impl Drop for ScopeGuard {
            fn drop(&mut self) {
                candle_metal_kernels::metal::commands::set_graph_scope_limit(self.prev);
            }
        }
        let _guard = ScopeGuard { prev: prev_limit };

        let result = f();

        // Final sync: CB commit + wait. После этого pool reuse безопасен,
        // GPU outputs готовы для CPU чтения.
        self.wait_until_completed_fast()?;

        result
    }

    /// Get and reset accumulated sync timing stats from flush_and_wait_fast.
    /// Returns: (count, sem_us, lock_us, commit_us, wait_us, total_us)
    pub fn take_sync_timings(&self) -> (u64, u64, u64, u64, u64, u64) {
        Commands::take_sync_timings()
    }

    /// Static version — no device instance needed (timings are thread-local).
    pub fn take_sync_timings_static() -> (u64, u64, u64, u64, u64, u64) {
        Commands::take_sync_timings()
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Creates a new buffer (not necessarily zeroed).
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        self.allocate_buffer(size)
    }

    /// Creates a new private buffer (not necessarily zeroed).
    ///
    /// This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
    pub fn new_private_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        let buffer = self
            .device
            .new_buffer(size, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        Ok(Arc::new(buffer))
    }

    /// Creates a new buffer from data.
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        let new_buffer = self
            .device
            .new_buffer_with_data(data.as_ptr().cast(), size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;

        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    /// Создаёт Metal-буфер без копирования данных (zero-copy) из mmap'd памяти.
    ///
    /// Буфер НЕ добавляется в пул буферов MetalDevice — он привязан к mmap
    /// и не должен переиспользоваться другими операциями.
    ///
    /// Требования:
    /// - `ptr` ДОЛЖЕН быть page-aligned (mmap гарантирует это)
    /// - `len` ДОЛЖЕН быть кратен page size (иначе Metal вернёт ошибку)
    /// - Вызывающий код ОБЯЗАН гарантировать, что mmap живёт дольше буфера
    pub fn new_buffer_no_copy(
        &self,
        ptr: *mut std::ffi::c_void,
        len: usize,
    ) -> Result<Arc<Buffer>> {
        let new_buffer = self
            .device
            .new_buffer_no_copy(ptr, len, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        Ok(Arc::new(new_buffer))
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let buffer = self.allocate_buffer(size_in_bytes)?;
        let blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        blit.end_encoding();
        Ok(buffer)
    }

    /// The critical allocator algorithm
    pub fn allocate_buffer(&self, size: usize) -> Result<Arc<Buffer>> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        if let Some(b) = find_available_buffer(size, &buffers) {
            // Cloning also ensures we increment the strong count
            return Ok(b.clone());
        }
        let size = buf_size(size);
        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = self
            .device
            .new_buffer(size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    /// Create a metal GPU capture trace on [`path`].
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
        let descriptor = MTLCaptureDescriptor::new();
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.set_capture_device(self.device().as_ref());
        // The [set_output_url] call requires an absolute path so we convert it if needed.
        if path.as_ref().is_absolute() {
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        } else {
            let path = std::env::current_dir()?.join(path);
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        }

        capture
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|e| MetalError::from(e.to_string()))?;
        Ok(())
    }
}

fn buf_size(size: usize) -> usize {
    size.saturating_sub(1).next_power_of_two()
}

fn find_available_buffer(size: usize, buffers: &BufferMap) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size = usize::MAX;
    for (buffer_size, subbuffers) in buffers.iter() {
        if buffer_size >= &size && buffer_size < &best_buffer_size {
            for sub in subbuffers {
                if Arc::strong_count(sub) == 1 {
                    best_buffer = Some(sub);
                    best_buffer_size = *buffer_size;
                }
            }
        }
    }
    best_buffer.cloned()
}
