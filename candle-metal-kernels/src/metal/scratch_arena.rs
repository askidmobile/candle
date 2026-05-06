// Pre-allocated scratch arena для intermediate Metal буферов (T-269 Phase 2 + Phase 3c).
//
// Slot lifecycle:
// 1. arena.try_acquire(size) → Some(Arc<Buffer>) если найден свободный slot
//    с capacity >= rounded(size). strong_count slot становится 2 (arena + caller).
// 2. Caller владеет Arc, передаёт в Tensor / MetalStorage.
// 3. Когда Tensor drop'd, Arc::drop → strong_count goes to 1.
// 4. Если caller вызвал GPU sync (wait_until_completed_fast), GPU done писать.
// 5. Следующий try_acquire видит strong_count == 1 → reuse safe.
//
// Безопасность: arena reuse возможен ТОЛЬКО после внешнего fence.
// Caller (Yttri forward()) обязан вызвать wait_until_completed_fast() ДО
// следующей allocate_buffer() чтобы GPU завершил запись в предыдущие slots.
// Это гарантируется sync per 4 layers в prefill loop.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::MetalKernelError;

use super::buffer::{Buffer, MTLResourceOptions};
use super::device::Device;

/// Один slot в scratch arena.
///
/// `buffer` — pre-allocated MTLBuffer с StorageModeShared.
/// Slot "свободен" когда `Arc::strong_count(&buffer) == 1`
/// (только arena держит ссылку).
pub struct ArenaSlot {
    pub buffer: Arc<Buffer>,
    pub capacity: usize,
}

/// Pre-allocated scratch arena для intermediate Metal буферов.
///
/// Все slots выделяются при создании через `new()`. `try_acquire()`
/// ищет свободный slot с достаточной ёмкостью — lock-free через
/// `Arc::strong_count`. При неудаче — caller fallback к pool.
///
/// # Безопасность
///
/// `Send + Sync` объявлены unsafe потому что `Arc<Buffer>` внутри не Send
/// из-за objc2 internals. Inference single-threaded, поэтому безопасно —
/// аналогично существующему `Commands` и `Buffer` в candle-metal-kernels.
///
/// GPU safety гарантируется внешним fence: перед reuse caller обязан
/// вызвать `MetalDevice::wait_until_completed_fast()`.
pub struct ScratchArena {
    /// Pre-allocated slots. Индекс фиксирован на весь lifetime arena.
    pub slots: Vec<ArenaSlot>,
    /// Round-robin start для поиска свободного slot (оптимизация).
    /// Cell<usize> — нет мьютекса, inference single-threaded.
    acquire_start: Cell<usize>,
}

unsafe impl Send for ScratchArena {}
unsafe impl Sync for ScratchArena {}

impl ScratchArena {
    /// Создать arena с заданными размерами slots.
    ///
    /// Каждый элемент `slot_sizes` → один MTLBuffer (StorageModeShared).
    /// Выделения происходят сразу, при создании arena.
    ///
    /// `slot_sizes` должен содержать **точные байтовые размеры** (не rounded),
    /// обычно это power-of-2 значения из Phase 1 trace.
    pub fn new(device: &Device, slot_sizes: &[usize]) -> Result<Self, MetalKernelError> {
        let opts = MTLResourceOptions::StorageModeShared;
        let mut slots = Vec::with_capacity(slot_sizes.len());
        for &sz in slot_sizes {
            let buf = device.new_buffer(sz, opts)?;
            slots.push(ArenaSlot {
                buffer: Arc::new(buf),
                capacity: sz,
            });
        }
        Ok(Self {
            slots,
            acquire_start: Cell::new(0),
        })
    }

    /// Попробовать взять slot из arena без блокировки.
    ///
    /// Ищет slot с:
    /// - `capacity >= rounded(requested)` — буфер достаточно большой
    /// - `Arc::strong_count(&buffer) == 1` — только arena держит ссылку
    ///   (slot не используется каким-то живым Tensor'ом)
    ///
    /// При успехе: инкрементирует strong_count (возвращает Arc::clone),
    /// обновляет round-robin pointer.
    ///
    /// При неудаче: возвращает `None` → caller должен fallback к pool.
    ///
    /// # Safety
    ///
    /// Caller ОБЯЗАН гарантировать GPU fence (`wait_until_completed_fast`)
    /// перед тем как slot может быть выдан для нового использования.
    /// Иначе GPU может писать в buffer одновременно с новым kernel.
    pub fn try_acquire(&self, requested: usize) -> Option<Arc<Buffer>> {
        // Round-up до power-of-two (как buf_size() в device.rs).
        // .max(64) — минимальный MTLBuffer alignment.
        let rounded = requested.saturating_sub(1).next_power_of_two().max(64);
        let n = self.slots.len();
        let start = self.acquire_start.get();

        for offset in 0..n {
            let idx = (start + offset) % n;
            let slot = &self.slots[idx];
            // Arc::strong_count — AtomicUsize::load(Relaxed). Дёшево.
            // Single-threaded inference: нет TOCTOU race.
            if slot.capacity >= rounded && Arc::strong_count(&slot.buffer) == 1 {
                // Продвигаем start чтобы следующий запрос не начинал с того же.
                self.acquire_start.set((idx + 1) % n);
                return Some(Arc::clone(&slot.buffer));
            }
        }
        None
    }

    /// Суммарный байтовый размер всех pre-allocated slots.
    pub fn total_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.capacity).sum()
    }

    /// Количество свободных slots (strong_count == 1).
    ///
    /// Полезно для diagnostics: если `free_count() == 0` во время forward —
    /// arena исчерпана и fallback к pool активен.
    pub fn free_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| Arc::strong_count(&s.buffer) == 1)
            .count()
    }

    /// Гистограмма ёмкостей: `(capacity_bytes, count)` отсортированная по ёмкости.
    pub fn capacity_histogram(&self) -> Vec<(usize, usize)> {
        let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
        for s in &self.slots {
            *hist.entry(s.capacity).or_insert(0) += 1;
        }
        hist.into_iter().collect()
    }
}

// ============================================================
// UnifiedScratchArena — Phase 3c: один большой MTLBuffer +
// bump allocator + offset dispatch.
//
// Дизайн: один MTLBuffer (~700 МБ) создаётся при загрузке модели.
// Все intermediate aллокации — это offsets в нём.
// После каждого sync-fence (per N layers) offset сбрасывается
// (reset), и все предыдущие virtual allocations освобождаются.
//
// Это transliteration ggml_dyn_tallocr simplified version.
// Не нужен complex free-list — у нас sync-per-N-layers boundary
// даёт natural reset point.
//
// # Использование
//
// 1. Создать один раз при загрузке модели:
//    let arena = Arc::new(UnifiedScratchArena::new(&device, 764 * 1024 * 1024)?);
//
// 2. В forward prefill loop перед каждой группой layers:
//    arena.reset(); // сбросить bump pointer
//
// 3. При allocate_buffer — вместо нового MTLBuffer:
//    let (buf, offset) = arena.try_acquire(size)?;
//    call_kernel(..., &buf, offset);
//
// 4. После sync fence (wait_until_completed_fast):
//    // GPU завершил, offset стал безопасным для reset.
//    arena.reset();
//
// # Безопасность
//
// GPU ОБЯЗАН завершить использование всех offset'ов из текущего
// "эпизода" (period между двумя reset()) ДО вызова reset().
// Caller обеспечивает это через wait_until_completed_fast().
// ============================================================

/// Результат успешного acquire из UnifiedScratchArena.
pub struct UnifiedAlloc {
    /// Shared backing buffer (Arc — arena держит ещё одну ссылку).
    pub buffer: Arc<Buffer>,
    /// Byte offset в buffer.
    pub offset_in_bytes: usize,
}

/// Bump-allocator scratch arena: один MTLBuffer, offset dispatch.
///
/// Thread-safety: `Send + Sync` unsafe по той же причине что `ScratchArena`.
/// Inference single-threaded, bump_offset — AtomicUsize для корректности
/// при потенциальном multi-thread (forward не параллелен сейчас).
pub struct UnifiedScratchArena {
    /// Единственный большой MTLBuffer (StorageModeShared).
    backing: Arc<Buffer>,
    /// Текущий bump offset в байтах.
    bump_offset: AtomicUsize,
    /// Максимальный размер backing buffer.
    capacity: usize,
    /// Счётчик exhausted alloc'ов (диагностика).
    exhausted_count: AtomicUsize,
}

unsafe impl Send for UnifiedScratchArena {}
unsafe impl Sync for UnifiedScratchArena {}

/// Metal требует минимального alignment 256 байт для buffer offset'ов.
const METAL_BUFFER_OFFSET_ALIGNMENT: usize = 256;

#[inline(always)]
fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

impl UnifiedScratchArena {
    /// Создать UnifiedScratchArena с заданным capacity (байты).
    ///
    /// Весь backing buffer выделяется сразу. Рекомендуемый размер
    /// для Qwen3.5-2B pp4096: 764 МБ (по profiling Phase 1).
    pub fn new(device: &Device, capacity: usize) -> Result<Self, MetalKernelError> {
        let backing = device.new_buffer(capacity, MTLResourceOptions::StorageModeShared)?;
        Ok(Self {
            backing: Arc::new(backing),
            bump_offset: AtomicUsize::new(0),
            capacity,
            exhausted_count: AtomicUsize::new(0),
        })
    }

    /// Попробовать выделить `size` байт из arena.
    ///
    /// Возвращает `Some((Arc<Buffer>, offset))` если есть место.
    /// Возвращает `None` если arena исчерпана (caller fallback к pool).
    ///
    /// Offset выровнен по `METAL_BUFFER_OFFSET_ALIGNMENT` (256 байт).
    pub fn try_acquire(&self, size: usize) -> Option<UnifiedAlloc> {
        if size == 0 {
            return None;
        }
        // Align size UP до кратного alignment для следующего alloc.
        let aligned = align_up(size, METAL_BUFFER_OFFSET_ALIGNMENT);
        let offset = self.bump_offset.fetch_add(aligned, Ordering::Relaxed);
        if offset + aligned > self.capacity {
            // Перемотать назад (arena исчерпана, не тратим место дальше).
            self.bump_offset.fetch_sub(aligned, Ordering::Relaxed);
            self.exhausted_count.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(UnifiedAlloc {
            buffer: Arc::clone(&self.backing),
            offset_in_bytes: offset,
        })
    }

    /// Сброс bump offset к нулю.
    ///
    /// ОПАСНО: вызывать ТОЛЬКО после `wait_until_completed_fast()`.
    /// GPU должен завершить все operations использующие предыдущие offset'ы.
    #[inline]
    pub fn reset(&self) {
        self.bump_offset.store(0, Ordering::Relaxed);
    }

    /// Текущее использование bump в байтах.
    pub fn used_bytes(&self) -> usize {
        self.bump_offset.load(Ordering::Relaxed)
    }

    /// Ёмкость backing buffer в байтах.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Сколько раз arena исчерпывалась с момента создания (диагностика).
    pub fn exhausted_count(&self) -> usize {
        self.exhausted_count.load(Ordering::Relaxed)
    }
}
