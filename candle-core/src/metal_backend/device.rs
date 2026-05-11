use crate::{DType, Result};

#[cfg(feature = "ug")]
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{
    metal::{
        BlitCommandEncoder, Buffer, BufferMap, Commands, ComputeCommandEncoder, Device,
        MTLResourceOptions, ScratchArena, UnifiedScratchArena,
    },
    Kernels,
};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

#[allow(unused_imports)]
use libc;

// ─────────────────────────────────────────────────────────────────────────────
// WeightResidencySet — MTLResidencySet lifecycle management (T-271 Phase 4 redux)
//
// Задача: сообщить macOS что weight buffers (GGUF mmaps) нужны GPU постоянно
// и не должны выгружаться в compressed memory. Это закрывает 5.7× RSS gap
// между Yttri (2.5 GB) и llama.cpp (434 MB) при работе с той же моделью.
//
// Паттерн из llama.cpp ggml-metal-device.m:
// - Per-buffer residency set создаётся при init каждого Metal buffer
// - requestResidency() сразу при создании set — память permanently wired
// - Background thread с keep_alive interval (default 3 мин) поддерживает
//   residency alive через периодические requestResidency() каждые 500ms
//
// Наша адаптация для Rust/candle (без per-buffer sets, один set на модель):
// - WeightResidencySet создаётся при init модели
// - new_buffer_no_copy добавляет каждый weight buffer через addAllocation()
// - commit_and_request() вызывается после загрузки всех весов
// - Background Rust thread каждые HEARTBEAT_INTERVAL_S секунд вызывает
//   requestResidency() для предотвращения macOS eviction при idle
// - Drop вызывает endResidency() + removeAllAllocations() + завершает thread
//
// macOS version: MTLResidencySet доступен с macOS 15.0. Проверяется в runtime
// через sysctl kern.osproductversion. Если < 15 или env YTTRI_DISABLE_RESIDENCY_SET=1
// — возвращает None, поведение не меняется.
// ─────────────────────────────────────────────────────────────────────────────

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLAllocation, MTLDevice as MTLDeviceProtocol, MTLResidencySet, MTLResidencySetDescriptor};
use objc2_foundation::NSString;

/// Интервал heartbeat-запросов residency (секунды).
/// llama.cpp использует 500ms; мы берём 30с — достаточно для предотвращения
/// eviction, меньше CPU overhead.
const HEARTBEAT_INTERVAL_S: u64 = 30;

/// Проверяет поддержку MTLResidencySet в runtime (macOS 15+).
/// Результат кешируется в OnceLock — однократная проверка при первом вызове.
fn supports_residency_set() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // Быстрая проверка через env override
        if std::env::var("YTTRI_DISABLE_RESIDENCY_SET")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            return false;
        }
        // macOS version check через sysctl kern.osproductversion
        // MTLResidencySet доступен с 15.0; мы на 26.x — всегда true,
        // но проверяем корректно для portability.
        macos_version_ge_15()
    })
}

/// Возвращает true если macOS >= 15.0 по kern.osproductversion.
fn macos_version_ge_15() -> bool {
    use std::ffi::CString;
    let mut buf = [0u8; 64];
    let mut size = buf.len();
    let key = match CString::new("kern.osproductversion") {
        Ok(k) => k,
        Err(_) => return true, // fallback: assume supported
    };
    let ret = unsafe {
        libc::sysctlbyname(
            key.as_ptr(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return true; // sysctlbyname failed — assume supported
    }
    // Parse "X.Y.Z" → major version
    let s = std::str::from_utf8(&buf[..size.saturating_sub(1)]).unwrap_or("26.0");
    let major: u32 = s.split('.').next().and_then(|m| m.parse().ok()).unwrap_or(26);
    major >= 15
}

/// Newtype wrapper делающий MTLResidencySet Send+Sync.
///
/// Safety: MTLResidencySet thread-safe по Apple документации —
/// requestResidency/endResidency/addAllocation/commit можно вызывать
/// с любого потока. Все ObjC retain/release операции в Retained<>
/// атомарны.
struct SendableRset(Retained<ProtocolObject<dyn MTLResidencySet>>);
unsafe impl Send for SendableRset {}
unsafe impl Sync for SendableRset {}

impl std::ops::Deref for SendableRset {
    type Target = ProtocolObject<dyn MTLResidencySet>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Внутреннее состояние residency set — содержит Objective-C объект.
/// Завёрнут в Mutex чтобы addAllocation/commit/requestResidency могли
/// вызываться из разных потоков (хотя типично это происходит последовательно).
struct ResidencyInner {
    rset: SendableRset,
    /// true = requestResidency() уже был вызван, memory wired.
    resident: bool,
}

// Safety: ResidencyInner содержит только SendableRset (Send+Sync) и bool.
unsafe impl Send for ResidencyInner {}
unsafe impl Sync for ResidencyInner {}

/// Публичный handle на residency set модели.
///
/// Создаётся через `MetalDevice::new_weight_residency_set()`.
/// Использование:
/// 1. Загрузить веса через `new_buffer_no_copy` — каждый буфер автоматически
///    добавляется в set если device содержит этот WeightResidencySet.
/// 2. Вызвать `commit_and_request()` после загрузки всех весов.
/// 3. Держать `Arc<WeightResidencySet>` живым пока модель в памяти.
///    Drop() автоматически вызовет endResidency().
pub struct WeightResidencySet {
    inner: Mutex<ResidencyInner>,
    /// Сигнал останова background heartbeat thread.
    stop: Arc<std::sync::atomic::AtomicBool>,
    /// JoinHandle для graceful shutdown при Drop.
    thread_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

unsafe impl Send for WeightResidencySet {}
unsafe impl Sync for WeightResidencySet {}

impl WeightResidencySet {
    /// Создать новый WeightResidencySet на данном MTLDevice.
    /// Возвращает None если macOS < 15 или YTTRI_DISABLE_RESIDENCY_SET=1.
    pub fn new(device: &Device) -> Option<Arc<Self>> {
        if !supports_residency_set() {
            return None;
        }
        let desc = MTLResidencySetDescriptor::new();
        desc.setLabel(Some(&NSString::from_str("yttri-weights")));
        let raw_device: &ProtocolObject<dyn MTLDeviceProtocol> =
            ProtocolObject::from_ref(device.as_ref());
        let rset = raw_device.newResidencySetWithDescriptor_error(&desc).ok()?;
        let rset = SendableRset(rset);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = stop.clone();
        // Для heartbeat thread создаём отдельный Arc<SendableRset>
        // чтобы не клонировать всю структуру.
        let rset_for_thread = Arc::new(SendableRset(rset.0.clone()));
        // Background heartbeat thread — аналог llama.cpp background dispatch
        // Периодически вызывает requestResidency() каждые HEARTBEAT_INTERVAL_S секунд.
        // macOS может "забыть" о residency при memory pressure; heartbeat
        // переодически напоминает ОС что буферы нужны GPU.
        // Heartbeat thread — запрашивает requestResidency() каждые 30с
        // только если YTTRI_WIRE_WEIGHTS=1. Иначе thread спит вечно.
        let wire_weights = std::env::var("YTTRI_WIRE_WEIGHTS").map(|v| v == "1").unwrap_or(false);
        let handle = std::thread::Builder::new()
            .name("yttri-residency-heartbeat".to_string())
            .spawn(move || {
                if !wire_weights {
                    // По умолчанию heartbeat не активен — requestResidency не вызывается.
                    return;
                }
                while !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_secs(HEARTBEAT_INTERVAL_S));
                    if !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                        rset_for_thread.requestResidency();
                    }
                }
            })
            .ok();
        Some(Arc::new(Self {
            inner: Mutex::new(ResidencyInner { rset, resident: false }),
            stop,
            thread_handle: Mutex::new(handle),
        }))
    }

    /// Добавить Metal buffer в residency set (uncommitted).
    ///
    /// Вызывается автоматически из `MetalDevice::new_buffer_no_copy`.
    /// Threshold: только буферы >= 1 MiB (weight tensors, не маленькие промежуточные).
    pub fn add_buffer(&self, buffer: &Buffer) {
        if buffer.length() < 1024 * 1024 {
            return;
        }
        if let Ok(inner) = self.inner.lock() {
            let mtl_alloc: &ProtocolObject<dyn MTLAllocation> =
                ProtocolObject::from_ref(buffer.as_ref());
            inner.rset.addAllocation(mtl_alloc);
        }
    }

    /// Зафиксировать все добавленные allocations (без requestResidency).
    ///
    /// Должен вызываться ОДИН РАЗ после загрузки всех весов модели.
    /// commit() — регистрирует буферы в residency set без их принудительного wiring.
    ///
    /// Для принудительного wiring вызовите `request_residency()` отдельно.
    /// Default: только commit(), без requestResidency() — иначе Physical footprint +500 MB.
    pub fn commit_and_request(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.rset.commit();
            // requestResidency() намеренно НЕ вызывается здесь.
            // Оно wires память в GPU (non-evictable), увеличивая Physical footprint на ~500 MB.
            // Вместо этого macOS сам управляет eviction — страницы становятся resident
            // по demand при первом доступе GPU.
            // Для явного wiring: YTTRI_WIRE_WEIGHTS=1 (см. request_residency()).
            if std::env::var("YTTRI_WIRE_WEIGHTS").map(|v| v == "1").unwrap_or(false) {
                inner.rset.requestResidency();
                inner.resident = true;
            }
            let alloc_size = inner.rset.allocatedSize();
            let alloc_mb = alloc_size / (1024 * 1024);
            eprintln!(
                "[yttri] MTLResidencySet committed: {} allocations, ~{} MB tracked",
                inner.rset.allocationCount(),
                alloc_mb,
            );
        }
    }

    /// Вернуть true если residency уже запрошена.
    pub fn is_resident(&self) -> bool {
        self.inner.lock().map(|g| g.resident).unwrap_or(false)
    }

    /// Принудительно завершить residency (обычно вызывается через Drop).
    fn end_residency_inner(inner: &mut ResidencyInner) {
        if inner.resident {
            inner.rset.endResidency();
            inner.rset.removeAllAllocations();
            inner.rset.commit();
            inner.resident = false;
        }
    }
}

impl Drop for WeightResidencySet {
    fn drop(&mut self) {
        // Останавливаем background thread
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Ok(mut guard) = self.thread_handle.lock() {
            if let Some(handle) = guard.take() {
                // Не ждём join чтобы не блокировать Drop — thread сам завершится
                // после текущего sleep интервала.
                let _ = handle.join();
            }
        }
        // Снимаем residency
        if let Ok(mut inner) = self.inner.lock() {
            Self::end_residency_inner(&mut inner);
        }
    }
}

use super::MetalError;

// ─────────────────────────────────────────────────────────────────────────────
// Allocation tracing API (T-269 Phase 1)
//
// Default off — нет overhead в hot path когда trace не активен.
// Активируется через begin_allocation_trace() только во время calibration
// forward. Результат забирается через end_allocation_trace().
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    static TRACE_ALLOC: Cell<bool> = const { Cell::new(false) };
    static TRACE_LOG: RefCell<Vec<TraceEntry>> = RefCell::new(Vec::new());
}

/// Одна запись о Metal buffer аллокации во время calibration forward.
#[derive(Clone, Debug)]
pub struct TraceEntry {
    /// Запрошенный размер (raw, до round-up).
    pub size: usize,
    /// Фактический размер буфера после buf_size() round-up.
    pub rounded_size: usize,
    /// Момент аллокации (монотонное время).
    pub timestamp: std::time::Instant,
    /// true = буфер взят из pool (reuse), false = новый MTLBuffer.
    pub from_pool: bool,
}

/// Начать трассировку аллокаций. Очищает предыдущий лог.
/// Вызвать перед calibration forward, после warmup.
pub fn begin_allocation_trace() {
    TRACE_LOG.with(|l| l.borrow_mut().clear());
    TRACE_ALLOC.with(|c| c.set(true));
}

/// Остановить трассировку и вернуть собранный лог.
/// Возвращает все TraceEntry с момента begin_allocation_trace().
pub fn end_allocation_trace() -> Vec<TraceEntry> {
    TRACE_ALLOC.with(|c| c.set(false));
    TRACE_LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
}

/// Проверить активна ли трассировка на текущем потоке.
#[inline(always)]
pub fn allocation_trace_active() -> bool {
    TRACE_ALLOC.with(|c| c.get())
}

#[inline(always)]
fn record_trace(size: usize, rounded: usize, from_pool: bool) {
    TRACE_LOG.with(|l| {
        l.borrow_mut().push(TraceEntry {
            size,
            rounded_size: rounded,
            timestamp: std::time::Instant::now(),
            from_pool,
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch arena thread_local activation (T-269 Phase 2)
//
// ACTIVE_ARENA содержит текущую scratch arena для данного потока (если активна).
// Активируется через MetalDevice::activate_scratch_arena() перед prefill loop,
// деактивируется через deactivate_scratch_arena() после final fence.
//
// Default: None — arena не активна, allocate_buffer использует обычный pool.
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    static ACTIVE_ARENA: RefCell<Option<Arc<ScratchArena>>> = const { RefCell::new(None) };
    /// Активная UnifiedScratchArena для Phase 3c offset dispatch.
    /// None = отключена (default). Активируется через activate_unified_arena().
    static ACTIVE_UNIFIED_ARENA: RefCell<Option<Arc<UnifiedScratchArena>>> = const { RefCell::new(None) };
    /// Если true — следующий allocate_buffer пропускает arena fast path.
    /// Используется для исключения long-lived allocations (KV cache) из arena.
    /// Автоматически сбрасывается при каждом чтении (одноразовый флаг).
    static SKIP_ARENA_NEXT: Cell<bool> = const { Cell::new(false) };
}

/// Пометить следующую аллокацию как "не для arena" (long-lived buffer).
///
/// Вызывать перед созданием KV cache буферов (`Tensor::zeros`) или других
/// долгоживущих буферов которые не должны занимать arena slots.
///
/// Флаг автоматически сбрасывается после одного `allocate_buffer` вызова.
pub fn skip_arena_next_alloc() {
    SKIP_ARENA_NEXT.with(|s| s.set(true));
}

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
    pub(crate) completion_aware_pool: bool,

    /// MTLResidencySet для weight buffers (T-271 Phase 4 redux).
    ///
    /// Создаётся через `new_weight_residency_set()` после init device.
    /// None = macOS < 15 или YTTRI_DISABLE_RESIDENCY_SET=1.
    /// `new_buffer_no_copy` автоматически добавляет буферы в этот set.
    ///
    /// Arc<Mutex<>> чтобы можно было менять через &self (MetalDevice Clone).
    pub(crate) weight_residency: Arc<Mutex<Option<Arc<WeightResidencySet>>>>,
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

    pub fn completion_aware_pool_enabled(&self) -> bool {
        self.completion_aware_pool
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

    /// Like `with_shared_encoder` but does NOT wait for GPU completion on exit.
    ///
    /// All Metal ops inside the closure go into a single CB (no auto-commits).
    /// The CB is committed when the pool entry is next flushed, but the caller
    /// MUST call `wait_until_completed_fast()` before reading any GPU outputs
    /// from CPU, or before any operation that might reuse buffers from this scope.
    ///
    /// Designed for Yttri's multi-CB prefill: the prefill loop encodes all 32
    /// layers into one CB, then the caller waits once at the very end. This
    /// removes ~6 unnecessary commit→swap cycles during the prefill encoding
    /// phase, reducing CPU dispatch overhead.
    pub fn with_single_cb_scope<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
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

        // No wait! Caller handles GPU sync later.
        f()
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

    // ─────────────────────────────────────────────────────────────────────
    // WeightResidencySet API (T-271 Phase 4 redux)
    // ─────────────────────────────────────────────────────────────────────

    /// Создать WeightResidencySet для данного device и установить его
    /// как активный. После этого каждый `new_buffer_no_copy` (GGUF weights)
    /// автоматически добавляет буфер в set.
    ///
    /// Возвращает `Arc<WeightResidencySet>` — сохранить пока модель живёт.
    /// Вызвать `commit_weight_residency()` после загрузки всех весов.
    ///
    /// Если macOS < 15 или YTTRI_DISABLE_RESIDENCY_SET=1 — возвращает None.
    /// Создать WeightResidencySet для данного device и установить его
    /// как активный. После этого каждый `new_buffer_no_copy` (GGUF weights)
    /// автоматически добавляет буфер в set.
    ///
    /// Возвращает `Arc<WeightResidencySet>` — сохранить пока модель живёт.
    /// Вызвать `commit_weight_residency()` после загрузки всех весов.
    ///
    /// Если macOS < 15 или YTTRI_DISABLE_RESIDENCY_SET=1 — возвращает None.
    pub fn new_weight_residency_set(&self) -> Option<Arc<WeightResidencySet>> {
        let rset = WeightResidencySet::new(&self.device)?;
        if let Ok(mut guard) = self.weight_residency.lock() {
            *guard = Some(rset.clone());
        }
        Some(rset)
    }

    /// Зафиксировать residency set и запросить wiring для всех добавленных
    /// weight buffers. Вызывать ОДИН РАЗ после загрузки всех весов модели.
    ///
    /// No-op если residency set не установлен.
    pub fn commit_weight_residency(&self) {
        if let Ok(guard) = self.weight_residency.lock() {
            if let Some(ref rset) = *guard {
                rset.commit_and_request();
            }
        }
    }

    /// Получить клон активного WeightResidencySet (если есть).
    pub fn weight_residency_set(&self) -> Option<Arc<WeightResidencySet>> {
        self.weight_residency.lock().ok()?.as_ref().cloned()
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
    /// T-271 Phase 4 redux: если установлен `weight_residency` set,
    /// автоматически добавляет буфер через `addAllocation()` (uncommitted).
    /// Вызвать `commit_weight_residency()` после загрузки всех весов.
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
        let new_buffer = Arc::new(new_buffer);
        // Добавляем в residency set если установлен.
        // Только крупные буферы (>= 1 MiB) — weight tensors, не промежуточные.
        if let Ok(guard) = self.weight_residency.lock() {
            if let Some(ref rset) = *guard {
                rset.add_buffer(&new_buffer);
            }
        }
        Ok(new_buffer)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let buffer = self.allocate_buffer(size_in_bytes)?;
        let blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        blit.end_encoding();
        Ok(buffer)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Scratch arena API (T-269 Phase 2)
    // ─────────────────────────────────────────────────────────────────────

    /// Создать scratch arena с заданными slot sizes.
    ///
    /// `slot_sizes` — байтовые размеры каждого slot (обычно power-of-2,
    /// из Phase 1 trace результатов).
    ///
    /// Арена выделяет все MTLBuffer'ы сразу при создании.
    /// Возвращает `Arc<ScratchArena>` — готова к активации через
    /// `activate_scratch_arena`.
    pub fn create_scratch_arena(&self, slot_sizes: &[usize]) -> Result<Arc<ScratchArena>> {
        Ok(Arc::new(
            ScratchArena::new(&self.device, slot_sizes).map_err(MetalError::from)?,
        ))
    }

    // ─────────────────────────────────────────────────────────────────────
    // UnifiedScratchArena API (T-269 Phase 3c)
    // ─────────────────────────────────────────────────────────────────────

    /// Создать UnifiedScratchArena (один большой MTLBuffer + bump allocator).
    ///
    /// `capacity` — суммарный размер в байтах (рекомендуется ~764 МБ для Qwen3.5-2B).
    /// Выделяет один MTLBuffer при создании.
    pub fn create_unified_arena(
        &self,
        capacity: usize,
    ) -> Result<Arc<UnifiedScratchArena>> {
        Ok(Arc::new(
            UnifiedScratchArena::new(&self.device, capacity).map_err(MetalError::from)?,
        ))
    }

    /// Активировать unified arena для текущего потока.
    ///
    /// После активации `try_acquire_unified` можно использовать для offset dispatch.
    /// Default off — нет влияния на `allocate_buffer` пока.
    pub fn activate_unified_arena(&self, arena: Arc<UnifiedScratchArena>) {
        ACTIVE_UNIFIED_ARENA.with(|a| *a.borrow_mut() = Some(arena));
    }

    /// Деактивировать unified arena для текущего потока.
    pub fn deactivate_unified_arena(&self) {
        ACTIVE_UNIFIED_ARENA.with(|a| *a.borrow_mut() = None);
    }

    /// Сбросить bump offset unified arena (после GPU fence).
    ///
    /// ОБЯЗАТЕЛЕН вызов `wait_until_completed_fast()` ДО reset.
    pub fn reset_unified_arena(&self) {
        ACTIVE_UNIFIED_ARENA.with(|a| {
            if let Some(arena) = a.borrow().as_ref() {
                arena.reset();
            }
        });
    }

    /// Попробовать выделить из unified arena.
    ///
    /// Возвращает `(Arc<Buffer>, offset_in_bytes)` при успехе.
    /// Возвращает `None` если arena не активна или исчерпана.
    ///
    /// # Использование
    ///
    /// ```ignore
    /// if let Some(alloc) = device.try_acquire_unified(size) {
    ///     call_kernel(..., &alloc.buffer, alloc.offset_in_bytes);
    ///     // buffer и offset передаются напрямую в encoder без создания отдельного MTLBuffer
    /// } else {
    ///     let buffer = device.allocate_buffer(size)?;
    ///     call_kernel(..., &buffer, 0);
    /// }
    /// ```
    pub fn try_acquire_unified(
        &self,
        size: usize,
    ) -> Option<candle_metal_kernels::metal::UnifiedAlloc> {
        ACTIVE_UNIFIED_ARENA.with(|a| {
            a.borrow()
                .as_ref()
                .and_then(|arena| arena.try_acquire(size))
        })
    }

    /// Выделить из unified arena с возвратом `(Arc<Buffer>, offset)`.
    ///
    /// Если unified arena активна — bump allocate из неё.
    /// Если arena не активна или исчерпана — fallback к `allocate_buffer(size)` с offset=0.
    ///
    /// Результат используется для создания `MetalStorage::new_with_offset`.
    ///
    /// # Safety
    ///
    /// Caller обязан гарантировать GPU fence (`wait_until_completed_fast`) перед
    /// следующим `reset_unified_arena()`. Иначе GPU может записывать в offset
    /// одновременно с новым kernel на том же offset.
    pub fn new_buffer_unified(
        &self,
        element_count: usize,
        dtype: crate::DType,
        _name: &str,
    ) -> Result<(Arc<Buffer>, usize)> {
        let size = element_count * dtype.size_in_bytes();
        // Пробуем unified arena first.
        if let Some(alloc) = self.try_acquire_unified(size) {
            return Ok((alloc.buffer, alloc.offset_in_bytes));
        }
        // Fallback: обычный pool, offset = 0.
        let buf = self.allocate_buffer(size)?;
        Ok((buf, 0))
    }

    /// Активировать scratch arena для текущего потока.
    ///
    /// После вызова `allocate_buffer` будет сначала пробовать взять slot
    /// из arena (lock-free через Arc::strong_count). Если arena не имеет
    /// подходящего свободного slot — fallback к обычному pool.
    ///
    /// Вызывать перед prefill loop. Парная деактивация —
    /// `deactivate_scratch_arena()` — ОБЯЗАТЕЛЬНА после final GPU fence.
    pub fn activate_scratch_arena(&self, arena: Arc<ScratchArena>) {
        ACTIVE_ARENA.with(|a| *a.borrow_mut() = Some(arena));
    }

    /// Деактивировать scratch arena для текущего потока.
    ///
    /// После вызова `allocate_buffer` использует только обычный pool.
    /// Вызывать ВСЕГДА после prefill loop (даже при панике — используй RAII guard).
    pub fn deactivate_scratch_arena(&self) {
        ACTIVE_ARENA.with(|a| *a.borrow_mut() = None);
    }

    /// Получить текущую активную scratch arena (для диагностики).
    pub fn active_scratch_arena(&self) -> Option<Arc<ScratchArena>> {
        ACTIVE_ARENA.with(|a| a.borrow().clone())
    }

    /// The critical allocator algorithm
    pub fn allocate_buffer(&self, size: usize) -> Result<Arc<Buffer>> {
        // ─── SCRATCH ARENA FAST PATH (T-269 Phase 2) ───────────────────
        // Проверяем arena ДО pool lock. Lock-free: только атомарный
        // strong_count check внутри try_acquire.
        // Default off: ACTIVE_ARENA содержит None → бесплатный borrow().
        //
        // SKIP_ARENA_NEXT: если установлен — пропускаем arena для этой аллокации.
        // Используется для исключения long-lived buffers (KV cache) из arena.
        let skip_arena = SKIP_ARENA_NEXT.with(|s| {
            let v = s.get();
            if v {
                s.set(false); // одноразовый флаг
            }
            v
        });
        if !skip_arena {
            if let Some(buf) = ACTIVE_ARENA.with(|a| {
                a.borrow()
                    .as_ref()
                    .and_then(|arena| arena.try_acquire(size))
            }) {
                // Arena hit: записываем в trace как from_pool=false
                // чтобы Phase 1/2 trace мог различать arena vs pool vs new.
                if allocation_trace_active() {
                    record_trace(size, buf.length(), false);
                }
                return Ok(buf);
            }
        }
        // ─── EXISTING POOL PATH (unchanged) ─────────────────────────────

        // Проверяем trace один раз — нет branch misprediction в hot path когда off.
        let trace_on = allocation_trace_active();

        let completed_command_buffer_id = if self.completion_aware_pool {
            let commands = self.commands.read().map_err(MetalError::from)?;
            Some(commands.completed_command_buffer_id())
        } else {
            None
        };
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        if let Some(b) = find_available_buffer(size, &buffers, completed_command_buffer_id) {
            // Cloning also ensures we increment the strong count
            if trace_on {
                record_trace(size, b.length(), true);
            }
            return Ok(b.clone());
        }
        let rounded = buf_size(size);
        let subbuffers = buffers.entry(rounded).or_insert(vec![]);

        let new_buffer = self
            .device
            .new_buffer(rounded, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        if trace_on {
            record_trace(size, rounded, false);
        }
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

fn find_available_buffer(
    size: usize,
    buffers: &BufferMap,
    completed_command_buffer_id: Option<u64>,
) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size = usize::MAX;
    for (buffer_size, subbuffers) in buffers.iter() {
        if buffer_size >= &size && buffer_size < &best_buffer_size {
            for sub in subbuffers {
                let gpu_completed = completed_command_buffer_id
                    .map(|completed| sub.last_used_command_buffer_id() <= completed)
                    .unwrap_or(true);
                if Arc::strong_count(sub) == 1 && gpu_completed {
                    best_buffer = Some(sub);
                    best_buffer_size = *buffer_size;
                }
            }
        }
    }
    best_buffer.cloned()
}

pub(crate) fn completion_aware_pool_enabled_from_env() -> bool {
    std::env::var("CANDLE_METAL_COMPLETION_AWARE_POOL")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}
