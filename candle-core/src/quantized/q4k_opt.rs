//! Pre-repacked Q4_K metadata для оптимизированного matmul kernel (Yttri T-275).
//!
//! Background: оригинальный `block_q4_K` хранит scales/mins в packed 6-bit формате
//! (12 bytes для 8 sub-blocks). Hot path Metal kernel'а тратит ~9-13% ALU
//! instructions на unpacking (см. `get_scale_min_k4_just2` + per-sub-block
//! `il < 2 ? d : d/16.h` selects + per-element FP32 conversion).
//!
//! `Q4KOptMetadata` хранит pre-computed `dl = d * scale` и `ml = dmin * min`
//! для каждого sub-block в `half` массивах. Kernel читает их напрямую через
//! `device const half *` параметр, обходя весь decode pipeline.
//!
//! Memory cost: 16 `half` per block × ~14M blocks для Qwen3.5-4B ≈ 470 MB.
//! Принято пользователем как trade-off за +15-25% prefill speedup.
//!
//! Layout pre-packed data, per block (32 bytes):
//! ```text
//! [dl_0, ml_0, dl_1, ml_1, dl_2, ml_2, ..., dl_7, ml_7]
//! ```
//! где dl_i = d * scales[i], ml_i = dmin * mins[i] (scales/mins decoded из
//! 6-bit packed format).

use crate::quantized::k_quants::BlockQ4K;
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

/// Pre-repacked Q4_K metadata (CPU-resident).
///
/// Создаётся через `repack_q4k_for_opt` для каждой Q4_K_M weight matrix.
/// `data` хранится в порядке blocks как в исходном QTensor (row-major).
#[derive(Debug, Clone)]
pub struct Q4KOptMetadata {
    /// 16 `f16` values per block (8 sub-blocks × pair `(dl, ml)`).
    pub data: Vec<f16>,
    /// Кол-во Q4_K blocks. `data.len() == block_count * 16`.
    pub block_count: usize,
    /// Identity guard — привязка к конкретному QTensor instance.
    /// Защита от reuse metadata от другого weight tensor.
    pub source_nonce: u64,
    /// Версия layout для future evolution.
    pub layout_version: u8,
}

/// Текущая версия layout. Increment при изменении data ordering или формата.
pub const Q4K_OPT_LAYOUT_V1: u8 = 1;

impl Q4KOptMetadata {
    /// Total memory footprint в байтах (CPU-side).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f16>()
    }

    /// Идентификатор инстанса source weight.
    pub fn source_nonce(&self) -> u64 {
        self.source_nonce
    }

    /// Количество Q4_K blocks.
    pub fn block_count(&self) -> usize {
        self.block_count
    }
}

/// Repack Q4_K blocks в pre-computed `dl/ml` half pairs.
///
/// Алгоритм декодирования 6-bit packed scales/mins взят из `BlockQ4K::vec_dot_unopt`
/// (k_quants.rs:1430-1440) — биткомпиляция через KMASK1/2/3 + `utmp[3]`.
///
/// Для каждого блока вычисляет 8 пар `(dl_i, ml_i)`:
/// ```text
/// dl_i = d * scales_decoded[i]
/// ml_i = dmin * mins_decoded[i]
/// ```
///
/// Возвращает [`Q4KOptMetadata`] с layout V1.
pub fn repack_q4k_for_opt(blocks: &[BlockQ4K]) -> Result<Q4KOptMetadata> {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let block_count = blocks.len();
    let mut data = Vec::with_capacity(block_count * 16);

    let mut utmp: [u32; 4] = [0; 4];
    let mut scales_u8: [u8; 8] = [0; 8];
    let mut mins_u8: [u8; 8] = [0; 8];

    for block in blocks {
        // Декодирование 6-bit packed scales/mins (12 bytes → 8 scales + 8 mins).
        // Точно тот же алгоритм что в BlockQ4K::vec_dot_unopt.
        LittleEndian::read_u32_into(&block.scales, &mut utmp[0..3]);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        LittleEndian::write_u32_into(&utmp[0..2], &mut scales_u8);
        LittleEndian::write_u32_into(&utmp[2..4], &mut mins_u8);

        let d_f32 = block.d.to_f32();
        let dmin_f32 = block.dmin.to_f32();

        // Interleaved storage: (dl_0, ml_0, dl_1, ml_1, ..., dl_7, ml_7)
        for i in 0..8 {
            let dl = d_f32 * scales_u8[i] as f32;
            let ml = dmin_f32 * mins_u8[i] as f32;
            data.push(f16::from_f32(dl));
            data.push(f16::from_f32(ml));
        }
    }

    Ok(Q4KOptMetadata {
        data,
        block_count,
        source_nonce: rand::random::<u64>(),
        layout_version: Q4K_OPT_LAYOUT_V1,
    })
}

// ════════════════════════════════════════════════════════════════════════════════
// GPU upload (только под metal feature)
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "metal")]
pub use metal_impl::Q4KOptMetadataGpu;

#[cfg(feature = "metal")]
mod metal_impl {
    use super::Q4KOptMetadata;
    use crate::Result;
    use std::sync::Arc;

    /// Pre-repacked Q4_K metadata uploaded в Metal buffer.
    ///
    /// Создаётся через `Q4KOptMetadata::upload_to_metal`. Buffer живёт пока
    /// `Q4KOptMetadataGpu` (через Arc) удерживается живым.
    #[derive(Debug, Clone)]
    pub struct Q4KOptMetadataGpu {
        pub buffer: Arc<candle_metal_kernels::metal::Buffer>,
        pub source_nonce: u64,
        pub byte_size: usize,
    }

    impl Q4KOptMetadata {
        /// Загрузить metadata в Metal device buffer.
        ///
        /// Создаёт новый MTLBuffer copy данных. `source_nonce` копируется
        /// для последующей валидации (метод [`Q4KOptMetadataGpu::matches`]).
        pub fn upload_to_metal(
            &self,
            metal_device: &crate::MetalDevice,
        ) -> Result<Q4KOptMetadataGpu> {
            let buffer = metal_device
                .new_buffer_with_data(&self.data)
                .map_err(|e| crate::Error::Msg(format!("Q4KOpt upload failed: {e}")))?;
            Ok(Q4KOptMetadataGpu {
                buffer,
                source_nonce: self.source_nonce,
                byte_size: self.size_bytes(),
            })
        }
    }

    impl Q4KOptMetadataGpu {
        /// Проверка идентичности metadata исходному QTensor через `source_nonce`.
        pub fn matches(&self, expected_nonce: u64) -> bool {
            self.source_nonce == expected_nonce
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Unit tests
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::k_quants::{BlockQ4K, K_SCALE_SIZE, QK_K};

    /// Helper: создать BlockQ4K с заданными d/dmin/scales для теста.
    fn make_test_block(d: f32, dmin: f32, scales: [u8; K_SCALE_SIZE]) -> BlockQ4K {
        BlockQ4K {
            d: f16::from_f32(d),
            dmin: f16::from_f32(dmin),
            scales,
            qs: [0u8; QK_K / 2],
        }
    }

    /// Decoder reference — биткомпиляция scales/mins, эквивалентная inline в
    /// `BlockQ4K::vec_dot_unopt` (k_quants.rs:1430-1440). Используется только в тестах.
    fn decode_scales_mins_reference(scales: &[u8; K_SCALE_SIZE]) -> ([u8; 8], [u8; 8]) {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        let mut scales_u8: [u8; 8] = [0; 8];
        let mut mins_u8: [u8; 8] = [0; 8];

        LittleEndian::read_u32_into(scales, &mut utmp[0..3]);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
        LittleEndian::write_u32_into(&utmp[0..2], &mut scales_u8);
        LittleEndian::write_u32_into(&utmp[2..4], &mut mins_u8);
        (scales_u8, mins_u8)
    }

    /// Round-trip: для известных d/dmin/scales blocks, `repack_q4k_for_opt`
    /// производит dl[i] = d * decoded_scales[i] и ml[i] = dmin * decoded_mins[i].
    #[test]
    fn test_repack_q4k_matches_reference_decoder() {
        // Реалистичные значения: d, dmin маленькие положительные, scales заполнены
        // патерном чтобы обходить все ветки KMASK логики.
        let d = 0.015625f32;
        let dmin = 0.0078125f32;
        let mut scales = [0u8; K_SCALE_SIZE];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 3) & 0xFF) as u8;
        }

        let block = make_test_block(d, dmin, scales);
        let metadata = repack_q4k_for_opt(&[block]).expect("repack");

        assert_eq!(metadata.block_count, 1);
        assert_eq!(metadata.data.len(), 16);
        assert_eq!(metadata.layout_version, Q4K_OPT_LAYOUT_V1);

        // Reference decoding
        let (ref_scales, ref_mins) = decode_scales_mins_reference(&scales);

        // Interleaved (dl_0, ml_0, dl_1, ml_1, ..., dl_7, ml_7)
        for i in 0..8 {
            let dl_actual = metadata.data[i * 2].to_f32();
            let ml_actual = metadata.data[i * 2 + 1].to_f32();
            let dl_expected = d * ref_scales[i] as f32;
            let ml_expected = dmin * ref_mins[i] as f32;
            let dl_diff = (dl_actual - f16::from_f32(dl_expected).to_f32()).abs();
            let ml_diff = (ml_actual - f16::from_f32(ml_expected).to_f32()).abs();
            assert!(
                dl_diff < 1e-5,
                "dl[{i}] mismatch: actual={dl_actual} expected={dl_expected}"
            );
            assert!(
                ml_diff < 1e-5,
                "ml[{i}] mismatch: actual={ml_actual} expected={ml_expected}"
            );
        }
    }

    /// Multiple blocks → layout сохраняет порядок.
    #[test]
    fn test_repack_multiple_blocks_layout() {
        let block_a = make_test_block(1.0, 0.5, [1u8; K_SCALE_SIZE]);
        let block_b = make_test_block(2.0, 0.25, [2u8; K_SCALE_SIZE]);

        let metadata = repack_q4k_for_opt(&[block_a, block_b]).expect("repack");

        assert_eq!(metadata.block_count, 2);
        assert_eq!(metadata.data.len(), 32);

        // Per-block: scales [1,1,...] и mins [1,1,...] после decode.
        // d=1.0, dmin=0.5 → dl[i] = scales_decoded[i] (любые в [0..63]), ml[i] = 0.5 * mins_decoded[i].
        // Block A's data — первые 16 f16; Block B's data — следующие 16.
        let block_a_data: Vec<f32> = metadata.data[0..16].iter().map(|h| h.to_f32()).collect();
        let block_b_data: Vec<f32> = metadata.data[16..32].iter().map(|h| h.to_f32()).collect();

        // Both blocks имеют разные dl/ml потому что scales [1,...] vs [2,...] decode по-разному.
        assert_ne!(block_a_data, block_b_data, "block A and B должны различаться");
    }

    /// Identity guard: два независимых repack'а имеют разные `source_nonce`.
    #[test]
    fn test_source_nonce_uniqueness() {
        let block = make_test_block(1.0, 0.5, [0u8; K_SCALE_SIZE]);
        let meta1 = repack_q4k_for_opt(&[block.clone()]).expect("repack 1");
        let meta2 = repack_q4k_for_opt(&[block]).expect("repack 2");
        assert_ne!(
            meta1.source_nonce(),
            meta2.source_nonce(),
            "rand::random<u64> collision (extremely unlikely)"
        );
    }

    /// size_bytes() корректно считает CPU footprint.
    #[test]
    fn test_size_bytes_calculation() {
        let blocks: Vec<BlockQ4K> = (0..100)
            .map(|i| make_test_block(0.01 * i as f32, 0.005, [i as u8; K_SCALE_SIZE]))
            .collect();
        let metadata = repack_q4k_for_opt(&blocks).expect("repack");

        // 100 blocks × 16 f16 × 2 bytes/f16 = 3200 bytes
        assert_eq!(metadata.size_bytes(), 100 * 16 * 2);
        assert_eq!(metadata.block_count(), 100);
    }
}
