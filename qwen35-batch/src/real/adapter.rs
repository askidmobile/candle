//! `BatchModel`-адаптер над реальной Qwen3.5-4B (`ModelWeights`).
//!
//! ## Архитектура (Phases 3+4+5: true batched decode)
//! Prefill — per-slot через `forward()` (seq_len>1, single-slot state:
//! `metal_ctx`/`cuda_ctx` + `kv_cache`). После prefill snapshot state слота и
//! мигрируется в batched decode buffers (`seed_slot_batched`) — DeltaNet state
//! в slot-регион batched GPU-буфера, attention KV-cache в `kv_cache_batched[slot]`.
//!
//! Decode — true batched: один `forward_decode_batch([B,1], positions)` для
//! B слотов одновременно (batched projections + batched delta_rule kernel с осью
//! slot + per-slot KV/RoPE/SDPA). Per-slot state живёт в batched buffers —
//! никаких snapshot shuffle (в отличие от time-multiplexed). Parity bit-exact
//! (каждый слот изолирован своим slot-регионом state, math идентичен single).
//!
//! Fallback: если batched GPU-контекст отсутствует (batched decode disabled),
//! `seed_slot_batched`/`forward_decode_batch` недоступны — адаптер откатывается
//! на time-multiplexed path (restore→forward→snapshot per slot).

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use crate::model::{BatchModel, DecodeBatch, MultimodalPrefill, PrefillChunk};
use crate::real::model_profile::ModelProfile;
use crate::real::model_weights::{ModelWeights, StateSnapshot, DECODE_BATCH_CAPACITY};
use crate::real::multimodal::{GridThw, PositionPlan};
use crate::real::vision::Qwen35Vision;

/// Адаптер реальной Qwen3.5-4B над `BatchModel` (true batched decode).
struct InstalledMultimodal {
    token_ids: Vec<u32>,
    grids: Vec<GridThw>,
    patches: Tensor,
    mm_token_types: Vec<u8>,
    plan: PositionPlan,
    features: Option<Tensor>,
}

fn slice_position_plan(plan: &PositionPlan, start: usize, len: usize) -> Result<PositionPlan> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow!("position-plan range overflow"))?;
    let slice = |axis: &[u32]| -> Result<Vec<u32>> {
        Ok(axis
            .get(start..end)
            .ok_or_else(|| anyhow!("position-plan range is out of bounds"))?
            .to_vec())
    };
    Ok(PositionPlan {
        text_positions: slice(&plan.text_positions)?,
        rope_positions: [
            slice(&plan.rope_positions[0])?,
            slice(&plan.rope_positions[1])?,
            slice(&plan.rope_positions[2])?,
        ],
        decode_rope_delta: plan.decode_rope_delta,
    })
}

/// Адаптер реальной Qwen3.5-4B над `BatchModel` (true batched decode).
pub struct Qwen35BatchAdapter {
    model: ModelWeights,
    device: Device,
    /// Валидированный profile модели (Phase 1 preflight).
    profile: ModelProfile,
    /// Per-slot snapshot после prefill — источник state для seed в batched buffers.
    /// Хранится и для time-multiplexed fallback (если batched decode disabled).
    slot_snaps: Vec<Option<StateSnapshot>>,
    /// Признак того, что слот уже засеян в batched buffers (после prefill).
    /// True = batched decode может использовать этот slot без повторного seed.
    slot_seeded: Vec<bool>,
    /// On-demand Vision component. Phase 8 owns TTL/load barrier; adapter only
    /// consumes explicitly loaded component and per-request payloads.
    vision: Option<Qwen35Vision>,
    multimodal: Vec<Option<InstalledMultimodal>>,
    rope_deltas: Vec<i64>,
    eos: u32,
    vocab: usize,
}

impl Qwen35BatchAdapter {
    /// Загрузить модель из GGUF (zero-copy на Metal) и подготовить N слотов.
    pub fn load(gguf_path: &Path, device: Device, num_slots: usize) -> Result<Self> {
        if num_slots > DECODE_BATCH_CAPACITY as usize {
            return Err(anyhow!(
                "num_slots {num_slots} exceeds decode capacity {DECODE_BATCH_CAPACITY}"
            ));
        }
        use candle_core::quantized::gguf_file;
        use std::fs::File;
        use std::sync::Arc;

        let file = File::open(gguf_path).map_err(|e| anyhow!("open GGUF: {e}"))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| anyhow!("mmap GGUF: {e}"))?;
        let mmap = Arc::new(mmap);

        // Один проход чтения GGUF: EOS + vocab (из token_embd.weight shape[0]) + веса.
        let mut c = std::io::Cursor::new(mmap.as_ref());
        let ct = gguf_file::Content::read(&mut c).map_err(|e| anyhow!("read GGUF: {e}"))?;

        // Phase 1 preflight: validate architecture, metadata, and tensor contracts
        // BEFORE heavy tensor loading. Fail-fast with aggregated errors.
        let file_size = mmap.len() as u64;
        let profile = ModelProfile::read_and_validate(&ct, file_size)
            .map_err(|e| anyhow!("GGUF validation failed: {e}"))?;
        log::info!(
            "[qwen35-batch] profile: arch={:?} blocks={} hidden={} ctx={} quant_count={} fingerprint={}",
            profile.architecture,
            profile.block_count,
            profile.hidden_size,
            profile.context_length,
            profile.quant_set.len(),
            profile.fingerprint.hash,
        );

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
            profile,
            slot_snaps: (0..num_slots).map(|_| None).collect(),
            slot_seeded: vec![false; num_slots],
            vision: None,
            multimodal: (0..num_slots).map(|_| None).collect(),
            rope_deltas: vec![0; num_slots],
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

    /// Prefix-cache инфраструктура: snapshot state последнего prefill'а слота.
    /// Сервер забирает его в LRU-кэш; при повторном prompt'е — inject + primed admit.
    pub fn slot_snapshot(&self, slot: usize) -> Option<StateSnapshot> {
        self.slot_snaps[slot].clone()
    }

    /// Внедрить snapshot слоту (prefix-cache hit): при следующем prefill_chunk
    /// с reset_first=false модель восстановит state из этого snapshot'а.
    pub fn inject_slot_snapshot(&mut self, slot: usize, snap: StateSnapshot) {
        self.slot_snaps[slot] = Some(snap);
        self.slot_seeded[slot] = false;
    }

    /// Делегированный доступ к модели (для profiling / debug_capture).
    pub fn model(&self) -> &ModelWeights {
        &self.model
    }

    /// Доступ к валидированному profile модели (Phase 1).
    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    pub fn load_vision(&mut self, gguf_path: &Path) -> Result<()> {
        self.vision = Some(
            Qwen35Vision::load(gguf_path, self.device.clone())
                .map_err(|error| anyhow!("load Vision component: {error}"))?,
        );
        Ok(())
    }

    pub fn unload_vision(&mut self) {
        self.vision = None;
        for payload in &mut self.multimodal {
            *payload = None;
        }
    }

    /// Полный сброс (все слоты) — только для тестов / teardown.
    #[allow(dead_code)]
    fn clear_all_state(&mut self) {
        self.model.clear_state();
        self.model.clear_state_batched(&self.device);
        for s in self.slot_snaps.iter_mut() {
            *s = None;
        }
        for f in self.slot_seeded.iter_mut() {
            *f = false;
        }
        for delta in &mut self.rope_deltas {
            *delta = 0;
        }
        for payload in &mut self.multimodal {
            *payload = None;
        }
    }
}

impl BatchModel for Qwen35BatchAdapter {
    fn vocab_size(&self) -> usize {
        if self.vocab != 0 {
            self.vocab
        } else {
            151943
        }
    }

    fn install_multimodal(&mut self, slot: usize, payload: MultimodalPrefill) -> Result<()> {
        if slot >= self.multimodal.len() {
            return Err(anyhow!("multimodal slot {slot} is out of range"));
        }
        if self.vision.is_none() {
            return Err(anyhow!("Vision component is not loaded"));
        }
        if payload.token_ids.len() != payload.mm_token_types.len()
            || payload
                .rope_positions
                .iter()
                .any(|axis| axis.len() != payload.token_ids.len())
            || payload.mm_token_types.iter().all(|kind| *kind == 0)
            || payload.mm_token_types.iter().any(|kind| *kind > 2)
        {
            return Err(anyhow!("multimodal token/position lengths differ or contain no media"));
        }
        if payload.patch_values.len()
            != payload
                .patch_rows
                .checked_mul(payload.patch_width)
                .ok_or_else(|| anyhow!("multimodal patch size overflow"))?
        {
            return Err(anyhow!("multimodal patch value count mismatch"));
        }
        let expected_features = payload
            .mm_token_types
            .iter()
            .filter(|kind| **kind != 0)
            .count();
        let grids: Vec<_> = payload
            .media_grids
            .into_iter()
            .map(|[t, h, w]| GridThw { t, h, w })
            .collect();
        let actual_features = grids.iter().try_fold(0usize, |total, grid| {
            if !grid.h.is_multiple_of(2) || !grid.w.is_multiple_of(2) {
                return Err(anyhow!("multimodal grid is not divisible by merge size"));
            }
            let count = grid
                .t
                .checked_mul(grid.h / 2)
                .and_then(|value| value.checked_mul(grid.w / 2))
                .ok_or_else(|| anyhow!("multimodal grid overflow"))?;
            total
                .checked_add(count)
                .ok_or_else(|| anyhow!("multimodal feature count overflow"))
        })?;
        if actual_features != expected_features {
            return Err(anyhow!(
                "multimodal feature/grid count {actual_features} != placeholder count {expected_features}"
            ));
        }
        let patches = Tensor::from_vec(
            payload.patch_values,
            (payload.patch_rows, payload.patch_width),
            &self.device,
        )?;
        let plan = PositionPlan {
            text_positions: (0..payload.token_ids.len())
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?,
            rope_positions: payload.rope_positions,
            decode_rope_delta: payload.decode_rope_delta,
        };
        self.multimodal[slot] = Some(InstalledMultimodal {
            token_ids: payload.token_ids,
            grids,
            patches,
            mm_token_types: payload.mm_token_types,
            plan,
            features: None,
        });
        self.rope_deltas[slot] = payload.decode_rope_delta;
        Ok(())
    }

    fn prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<Vec<f32>> {
        let sidx = chunk.slot_idx;
        if sidx >= self.slot_snaps.len() || chunk.tokens.is_empty() {
            return Err(anyhow!("prefill slot is out of range or chunk is empty"));
        }
        if chunk.reset_first {
            // Keep installed media for first chunk; reset only model state.
            self.model.clear_state();
            self.slot_snaps[sidx] = None;
            self.slot_seeded[sidx] = false;
            if self.multimodal[sidx].is_none() {
                self.rope_deltas[sidx] = 0;
            }
        } else if let Some(snap) = self.slot_snaps[sidx].as_ref() {
            // Продолжение prefill после чанка: восстанавливаем single-slot state.
            self.model
                .restore_state(&self.device, snap)
                .map_err(|e| anyhow!("prefill restore: {e}"))?;
        } else {
            return Err(anyhow!(
                "prefill_chunk: slot {sidx} без snapshot и не reset"
            ));
        }

        let ids = Tensor::from_vec(
            chunk.tokens.clone(),
            (1usize, chunk.tokens.len()),
            &self.device,
        )?;
        let logits = if let Some(media) = self.multimodal[sidx].as_mut() {
            let end = chunk
                .start_pos
                .checked_add(chunk.tokens.len())
                .ok_or_else(|| anyhow!("prefill range overflow"))?;
            if media.token_ids.get(chunk.start_pos..end) != Some(chunk.tokens.as_slice()) {
                return Err(anyhow!(
                    "multimodal prefill tokens differ from installed prompt"
                ));
            }
            if media.features.is_none() {
                media.features = Some(
                    self.vision
                        .as_ref()
                        .ok_or_else(|| anyhow!("Vision component is not loaded"))?
                        .forward(&media.patches, &media.grids)
                        .map_err(|error| anyhow!("Vision forward: {error}"))?,
                );
            }
            let embeds = self.model.embed_tokens(&ids, &self.device)?;
            let mut feature_offset = media.mm_token_types[..chunk.start_pos]
                .iter()
                .filter(|kind| **kind != 0)
                .count();
            let features = media.features.as_ref().unwrap();
            let mut cursor = 0usize;
            while cursor < chunk.tokens.len() {
                if media.mm_token_types[chunk.start_pos + cursor] == 0 {
                    cursor += 1;
                    continue;
                }
                let start = cursor;
                while cursor < chunk.tokens.len()
                    && media.mm_token_types[chunk.start_pos + cursor] != 0
                {
                    cursor += 1;
                }
                let len = cursor - start;
                embeds.slice_set(
                    &features.narrow(0, feature_offset, len)?.unsqueeze(0)?,
                    1,
                    start,
                )?;
                feature_offset += len;
            }
            let plan = slice_position_plan(&media.plan, chunk.start_pos, chunk.tokens.len())?;
            self.model
                .forward_embeds_mrope(&embeds, &plan, chunk.start_pos)
                .map_err(|error| anyhow!("multimodal prefill forward: {error}"))?
        } else {
            self.model
                .forward(&ids, chunk.start_pos)
                .map_err(|e| anyhow!("prefill forward: {e}"))?
        };
        let logits_f32 = logits
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()
            .map_err(|e| anyhow!("prefill logits to_vec1: {e}"))?;

        // Сохранить snapshot state слота на позиции start_pos + tokens.len().
        let new_pos = chunk.start_pos + chunk.tokens.len();
        let snap = self
            .model
            .snapshot_state(&self.device, new_pos)
            .map_err(|e| anyhow!("prefill snapshot: {e}"))?;
        self.slot_snaps[sidx] = Some(snap);
        // Prefill изменил single-slot state; batched slot нужно пере-seed.
        self.slot_seeded[sidx] = false;

        Ok(logits_f32)
    }

    fn decode_batch(&mut self, batch: &DecodeBatch) -> Result<Vec<Vec<f32>>> {
        let b = batch.items.len();
        if batch.items.iter().any(|item| item.slot_idx >= self.slot_snaps.len()) {
            return Err(anyhow!("decode slot is out of range"));
        }
        if b == 0 {
            return Ok(vec![]);
        }

        // Сначала seed всех слотов в batched buffers (если ещё не засеяны).
        for it in &batch.items {
            let sidx = it.slot_idx;
            if !self.slot_seeded[sidx] {
                let snap = self.slot_snaps[sidx]
                    .as_ref()
                    .ok_or_else(|| anyhow!("decode_batch: slot {sidx} без snapshot"))?;
                self.model
                    .seed_slot_batched(&self.device, sidx, snap)
                    .map_err(|e| anyhow!("decode seed slot {sidx}: {e}"))?;
                self.slot_seeded[sidx] = true;
            }
        }

        // Собираем batched входы: tokens [B,1] + positions [B] + slots [B].
        // `slots` — отображение batch_idx → slot_idx: persistent state (ssm_state,
        // conv_state, kv_cache) адресуется по slot_idx, а не по batch_idx.
        // После сжатия батча (ранний EOS) это сохраняет привязку слота к его state.
        let mut tokens = Vec::with_capacity(b);
        let mut positions = Vec::with_capacity(b);
        let mut rope_positions = Vec::with_capacity(b);
        let mut slot_order = Vec::with_capacity(b); // (batch_idx, slot_idx)
        let mut slots = Vec::with_capacity(b);
        for it in &batch.items {
            tokens.push(it.token);
            positions.push(it.pos);
            let rope_position = i64::try_from(it.pos)?
                .checked_add(self.rope_deltas[it.slot_idx])
                .ok_or_else(|| anyhow!("decode RoPE position overflow"))?;
            rope_positions.push(
                usize::try_from(rope_position)
                    .map_err(|_| anyhow!("negative decode RoPE position"))?,
            );
            slot_order.push(it.slot_idx);
            slots.push(it.slot_idx as u32);
        }
        let ids = Tensor::from_vec(tokens, (b, 1usize), &self.device)?;
        let logits = self
            .model
            .forward_decode_batch(&ids, &rope_positions, &slots)
            .map_err(|e| anyhow!("decode_batch forward: {e}"))?;
        // One D2H transfer for [B, vocab], then split on host. Per-row to_vec1()
        // serialized four CUDA synchronizations/copies for B=4.
        let flat = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()
            .map_err(|e| anyhow!("decode_batch logits to_vec1: {e}"))?;
        let vocab = self.vocab_size();
        if flat.len() != b * vocab {
            return Err(anyhow!(
                "decode_batch logits length {} != batch {b} * vocab {vocab}",
                flat.len()
            ));
        }
        let out = flat.chunks_exact(vocab).map(<[f32]>::to_vec).collect();

        // Обновить per-slot snapshot из batched state после decode (для последующего
        // prefill-продолжения и для повторного seed, если слот покинет batch и вернётся).
        // Дешевле: позиция продвинулась на 1; state живёт в batched buffers, но snapshot
        // нужен для prefill-restore (который использует single-slot path).
        // Полный re-snapshot из batched buffers не реализован (требует dtoh slot-region);
        // вместо этого помечаем slot как требующий re-seed перед следующим decode —
        // batched state уже актуален в буферах, snapshot устарел только по позиции.
        for i in 0..b {
            let sidx = slot_order[i];
            // Позиция в snapshot продвинулась; сам state валиден в batched buffers.
            // Обновляем только position (для future prefill-restore корректен только
            // если последующий prefill стартует с it.pos+1 и reset — обычный паттерн).
            if let Some(snap) = self.slot_snaps[sidx].as_mut() {
                snap.position = positions[i] + 1;
            }
            // slot остаётся seeded — batched buffers уже содержат обновлённый state.
        }

        Ok(out)
    }

    fn reset_slot(&mut self, idx: usize) -> Result<()> {
        if idx >= self.slot_snaps.len() {
            return Err(anyhow!("reset slot {idx} is out of range"));
        }
        self.slot_snaps[idx] = None;
        self.slot_seeded[idx] = false;
        self.multimodal[idx] = None;
        self.rope_deltas[idx] = 0;
        Ok(())
    }
}

impl Qwen35BatchAdapter {
    /// EOS token id модели.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Размер словаря (для BatchScheduler::new).
    pub fn vocab_size(&self) -> usize {
        if self.vocab != 0 {
            self.vocab
        } else {
            151943
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_position_plan_slices_fail_closed() {
        let plan = PositionPlan {
            text_positions: vec![0, 1, 2],
            rope_positions: [vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]],
            decode_rope_delta: -1,
        };
        let slice = slice_position_plan(&plan, 1, 2).unwrap();
        assert_eq!(slice.text_positions, vec![1, 2]);
        assert_eq!(slice.rope_positions[0], vec![1, 2]);
        assert!(slice_position_plan(&plan, 2, 2).is_err());
    }

    #[test]
    fn load_rejects_excess_slots_before_opening_gguf() {
        let missing = Path::new("this-model-must-not-exist.gguf");
        let error = match Qwen35BatchAdapter::load(
            missing,
            Device::Cpu,
            DECODE_BATCH_CAPACITY as usize + 1,
        ) {
            Ok(_) => panic!("excess slots unexpectedly accepted"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            format!(
                "num_slots {} exceeds decode capacity {DECODE_BATCH_CAPACITY}",
                DECODE_BATCH_CAPACITY + 1
            )
        );
    }
}
