//! `BatchModel`-адаптер над реальной Qwen3.5-4B (`ModelWeights`).
//!
//! ## Архитектурная стена (подтверждено при портировании)
//! `DeltaNetLayer::forward` содержит `debug_assert_eq!(seq_len, 1)` (1469) и
//! `debug_assert_eq!(b_sz, 1)` (1711); Metal-ядра `dispatch_delta_rule` не имеют
//! batch-оси (grid per v-head, F32 per-token scratch). ⇒ true batched decode
//! (один matmul по B токенам через общие веса) на текущем Candle **невозможен**
//! без переписывания 4 Metal-ядер delta_rule + fused GDN под batch-измерение.
//!
//! ## Реализованный путь: time-multiplexing одной weight copy
//! Веса — одни (`ModelWeights` shared). Per-slot recurrent state (GDN ssm/conv)
//! и KV-cache вынесены в `Vec<Option<StateSnapshot>>` и свопаются через
//! `ModelWeights::restore_state` / `snapshot_state` (T-274 prompt-cache API).
//! Каждый decode-шаг: restore state слота → `forward([tok], pos)` → snapshot
//! обратно. Parity — бит-точная (каждый слот изолирован своим state).
//!
//! Цена: snapshot ~114 МБ/слот (24 DeltaNet × 2.1 МБ + 8 Attention KV) →
//! своп ~B×228 МБ/decode-шаг. Это и измеряет штраф архитектуры Candle за
//! multi-slot без переписывания ядер (гипотеза: aggregate SLOWER sequential,
//! что подтверждает D-12). True continuous-batching (×1.15-1.35) требует
//! batch-axis в Metal-ядрах — out of scope прототипа.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use crate::model::{BatchModel, DecodeBatch, PrefillChunk};
use crate::real::model_weights::{ModelWeights, StateSnapshot};

/// Адаптер реальной Qwen3.5-4B над `BatchModel` (time-multiplexed).
pub struct Qwen35BatchAdapter {
    model: ModelWeights,
    device: Device,
    /// Per-slot сохранённый state (GDN + KV) + позиция. None = свежий слот.
    slot_snaps: Vec<Option<StateSnapshot>>,
    eos: u32,
    vocab: usize,
}

impl Qwen35BatchAdapter {
    /// Загрузить модель из GGUF (zero-copy на Metal) и подготовить N слотов.
    pub fn load(gguf_path: &Path, device: Device, num_slots: usize) -> Result<Self> {
        use candle_core::quantized::gguf_file;
        use std::fs::File;
        use std::sync::Arc;

        let file = File::open(gguf_path).map_err(|e| anyhow!("open GGUF: {e}"))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| anyhow!("mmap GGUF: {e}"))?;
        let mmap = Arc::new(mmap);

        // Один проход чтения GGUF: EOS + vocab (из token_embd.weight shape[0]) + веса.
        // Qwen3.5-4B vocab = 248320 (padding до 256 для квантизации), eos=248046 —
        // хардкод 151943 был неверным для этого GGUF (ломал assert logits.len==vocab).
        let mut c = std::io::Cursor::new(mmap.as_ref());
        let ct = gguf_file::Content::read(&mut c).map_err(|e| anyhow!("read GGUF: {e}"))?;

        let eos = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(151645);

        let vocab = ct
            .tensor_infos
            .get("token_embd.weight")
            .and_then(|info| info.shape.dims().first().copied())
            .unwrap_or(0);
        log::info!(
            "[qwen35-batch] GGUF: eos={eos}, vocab(shape)={vocab}, mmap={:.1} MB",
            mmap.len() as f64 / 1024.0 / 1024.0
        );

        // Загрузка весов (zero-copy на Metal, обычный путь на CPU).
        #[cfg(target_os = "macos")]
        let model = if matches!(device, Device::Metal(_)) {
            ModelWeights::from_gguf_zero_copy(ct, mmap, &device)
                .map_err(|e| anyhow!("load weights zero-copy: {e}"))?
        } else {
            ModelWeights::from_gguf(ct, mmap, &device).map_err(|e| anyhow!("load weights: {e}"))?
        };
        #[cfg(not(target_os = "macos"))]
        let model =
            ModelWeights::from_gguf(ct, mmap, &device).map_err(|e| anyhow!("load weights: {e}"))?;

        Ok(Self {
            model,
            device,
            slot_snaps: (0..num_slots).map(|_| None).collect(),
            eos,
            vocab,
        })
    }

    /// Загрузить с явным vocab (из GGUF metadata `tokenizer.ggml.tokens` len).
    pub fn load_with_vocab(
        gguf_path: &Path,
        device: Device,
        num_slots: usize,
        vocab: usize,
    ) -> Result<Self> {
        let mut a = Self::load(gguf_path, device, num_slots)?;
        a.vocab = vocab;
        Ok(a)
    }

    /// Делегированный доступ к модели (для profiling / debug_capture).
    pub fn model(&self) -> &ModelWeights {
        &self.model
    }

    /// Сбросить state ВСЕЙ модели (внутренний single-stream state → нули).
    /// Per-slot snapshots не трогает (они — source of truth per-slot).
    fn clear_model_state(&mut self) {
        self.model.clear_state();
    }
}

impl BatchModel for Qwen35BatchAdapter {
    fn vocab_size(&self) -> usize {
        // Основной load-path читает shape[0] у token_embd.weight из GGUF.
        // Fallback нужен только для нестандартного GGUF без token_embd metadata.
        if self.vocab != 0 {
            self.vocab
        } else {
            151943
        }
    }

    fn prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<Vec<f32>> {
        let sidx = chunk.slot_idx;
        if chunk.reset_first {
            // Новый запрос: обнуляем внутренний state модели, позиция слота = 0.
            self.clear_model_state();
            // restore не нужен (свежий state); snapshot слота тоже сбрасываем.
            self.slot_snaps[sidx] = None;
        } else if let Some(snap) = self.slot_snaps[sidx].as_ref() {
            // Продолжение prefill после чанка: восстанавливаем state слота.
            self.model
                .restore_state(&self.device, snap)
                .map_err(|e| anyhow!("prefill restore: {e}"))?;
        } else {
            // reset_first=false, но snapshot'а нет —Fresh slot; это ошибка scheduler'а.
            return Err(anyhow!(
                "prefill_chunk: slot {sidx} без snapshot и не reset"
            ));
        }

        // forward(prompt chunk) — prefill path (seq_len>1). index_pos = start_pos.
        let ids = Tensor::from_vec(
            chunk.tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(),
            (1usize, chunk.tokens.len()),
            &self.device,
        )?;
        let logits = self
            .model
            .forward(&ids, chunk.start_pos)
            .map_err(|e| anyhow!("prefill forward: {e}"))?;
        // logits: [1, vocab] (forward уже берёт последний токен). Снимаем в Vec<f32>.
        let logits_f32 = logits
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()
            .map_err(|e| anyhow!("prefill logits to_vec1: {e}"))?;

        // Сохранить state слота на позиции start_pos + tokens.len().
        let new_pos = chunk.start_pos + chunk.tokens.len();
        let snap = self
            .model
            .snapshot_state(&self.device, new_pos)
            .map_err(|e| anyhow!("prefill snapshot: {e}"))?;
        self.slot_snaps[sidx] = Some(snap);

        Ok(logits_f32)
    }

    fn decode_batch(&mut self, batch: &DecodeBatch) -> Result<Vec<Vec<f32>>> {
        // Time-multiplexed: per-slot restore → forward(1 token) → snapshot.
        let mut out = Vec::with_capacity(batch.items.len());
        for it in &batch.items {
            let sidx = it.slot_idx;
            // Restore state слота (должен быть после prefill).
            if let Some(snap) = self.slot_snaps[sidx].as_ref() {
                self.model
                    .restore_state(&self.device, snap)
                    .map_err(|e| anyhow!("decode restore slot {sidx}: {e}"))?;
            } else {
                return Err(anyhow!("decode_batch: slot {sidx} без snapshot"));
            }
            // forward([tok], pos) — single-token decode path.
            let ids = Tensor::from_vec(vec![it.token], (1usize, 1usize), &self.device)?;
            let logits = self
                .model
                .forward(&ids, it.pos)
                .map_err(|e| anyhow!("decode forward slot {sidx}: {e}"))?;
            let logits_f32 = logits
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1()
                .map_err(|e| anyhow!("decode logits to_vec1 slot {sidx}: {e}"))?;
            // Сохранить state слота на позиции pos+1.
            let snap = self
                .model
                .snapshot_state(&self.device, it.pos + 1)
                .map_err(|e| anyhow!("decode snapshot slot {sidx}: {e}"))?;
            self.slot_snaps[sidx] = Some(snap);
            out.push(logits_f32);
        }
        Ok(out)
    }

    fn reset_slot(&mut self, idx: usize) -> Result<()> {
        self.slot_snaps[idx] = None;
        Ok(())
    }
}

impl Qwen35BatchAdapter {
    /// EOS token id модели.
    pub fn eos(&self) -> u32 {
        self.eos
    }
}
