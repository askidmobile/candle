//! Настройка CUDA default memory pool (stream-ordered allocator).
//!
//! cudarc 0.19 `CudaStream::alloc` уже использует `cuMemAllocAsync` — драйверный
//! пул. НО у default pool release threshold = 0: память возвращается ОС на
//! каждой синхронизации stream, и следующая аллокация снова мапит страницы
//! (дорого, миллисекунды каждая; в decode hot path — десятки аллокаций/шаг).
//!
//! Фикс — одна настройка: RELEASE_THRESHOLD = u64::MAX. Пул удерживает память
//! между синхронизациями, reuse становится почти бесплатным. Это заменяет
//! собственный caching allocator: драйверный пул уже bucketed и stream-aware.
//!
//! Использование: вызвать `retain_default_mempool(&device)` сразу после
//! создания CUDA device. Откат: не вызывать (поведение по умолчанию CUDA).

use crate::cuda_backend::device::CudaDevice;
use crate::Result;
use cudarc::driver::sys;

/// Установить release threshold = max на default memory pool устройства.
/// После этого пул не отдаёт страницы ОС на sync-точках → аллокации
/// переиспользуют уже замапленную память без page-map стоимости.
pub fn retain_default_mempool(dev: &CudaDevice) -> Result<()> {
    let cu_dev = dev.context.cu_device();
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let res = sys::cuDeviceGetDefaultMemPool(&mut pool, cu_dev);
        if res != sys::CUresult::CUDA_SUCCESS {
            crate::bail!("cuDeviceGetDefaultMemPool failed: {res:?}");
        }
        let mut threshold: u64 = u64::MAX;
        let res = sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &mut threshold as *mut u64 as *mut _,
        );
        if res != sys::CUresult::CUDA_SUCCESS {
            crate::bail!("cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed: {res:?}");
        }
    }
    Ok(())
}

/// Свободная VRAM (MiB) по драйверу. Для условного включения retain:
/// retain на карте с малым запасом = reserved-память пула добивает карту
/// до paging (измерено: 12GB карта 98% → 7x регресс decode).
pub fn free_mib(dev: &CudaDevice) -> Result<u64> {
    let (free, _total) = dev
        .context
        .mem_get_info()
        .map_err(|e| crate::Error::Msg(format!("cuMemGetInfo: {e}")))?;
    Ok((free / 1024 / 1024) as u64)
}

/// Текущее использование пула (MiB): (used, reserved). Для trace-диагностики.
pub fn default_mempool_usage(dev: &CudaDevice) -> Result<(u64, u64)> {
    let cu_dev = dev.context.cu_device();
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let res = sys::cuDeviceGetDefaultMemPool(&mut pool, cu_dev);
        if res != sys::CUresult::CUDA_SUCCESS {
            crate::bail!("cuDeviceGetDefaultMemPool failed: {res:?}");
        }
        let mut used: u64 = 0;
        let mut reserved: u64 = 0;
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            &mut used as *mut u64 as *mut _,
        );
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            &mut reserved as *mut u64 as *mut _,
        );
        Ok((used / 1024 / 1024, reserved / 1024 / 1024))
    }
}
