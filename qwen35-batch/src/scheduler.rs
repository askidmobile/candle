//! Continuous batching scheduler — оркестрирует N слотов над [`BatchModel`].
//!
//! ## Дизайн (см. research doc / D-12 Yttri wiki)
//!
//! - Слоты [`Slot`] проходят IDLE→PREFILL→DECODE→(FINISHED→IDLE).
//! - Каждый выз [`BatchScheduler::step`] делает:
//!   1. **Prefill phase**: выбирает ОДИН Prefilling-слот и кормит его одним
//!      чанком. Когда чанк завершает prompt, логиты последнего токена
//!      сэмплируются → **первый сгенерированный токен** (как в production:
//!      избегаем повторной обработки последнего токена prompt'а в рекуррентном
//!      state — критично для GDN, где повторный forward последнего токена
//!      испортил бы conv/SSM state).
//!   2. **Decode phase**: собирает ВСЕ Decoding-слоты в один [`DecodeBatch`] и
//!      вызывает `model.decode_batch` — батчевые QMatMul амортизируют чтение весов.
//!   3. Сэмплирует токены, обновляет state; завершённые слоты помечаются Finished.
//!      Сбор outputs + reset в Idle делает caller ([`BatchScheduler::run_with_collection`]).
//!
//! ## Паритет
//! [`BatchScheduler::sequential_reference`] прогоняет те же запросы по одному
//! через single-slot scheduler — baseline. Greedy детерминирован ⇒ одинаковые
//! логиты ⇒ одинаковые токены ⇒ batched == sequential.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::Result;

use crate::model::{BatchModel, DecodeBatch, DecodeItem, GreedySampler, PrefillChunk, Sampler};
use crate::slot::{Slot, SlotRequest, SlotStatus};

/// Статистика одного прогона scheduler'а.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Число decode-шагов (вызовов decode_batch).
    pub decode_steps: usize,
    /// Суммарное число токенов, эмитнутых decode-шагами + первых-из-prefill.
    pub total_decode_tokens: usize,
    /// Пиковое число одновременно декодируемых слотов.
    pub max_concurrent_decode: usize,
    /// Число prefill-чанков.
    pub prefill_chunks: usize,
    /// Полное wall-clock время прогона (нс).
    pub wall_ns: u128,
    /// Время prefill-фазы (нс).
    pub prefill_ns: u128,
    /// Время decode-фазы (нс).
    pub decode_ns: u128,
}

impl SchedulerStats {
    /// Агрегатные tok/s по decode-фазе.
    pub fn decode_aggregate_tps(&self) -> f64 {
        let s = self.decode_ns as f64 / 1e9;
        if s <= 0.0 {
            0.0
        } else {
            self.total_decode_tokens as f64 / s
        }
    }
}

/// Размер чанка prefill. В прототипе — весь prompt (prefill последовательный).
/// Уменьшение → interleaving prefill и decode (true LLM-level batching); требует
/// чанк-чекпойнтов рекуррентного GDN state (checkpoint complexity, beyond scope).
pub const PREFILL_CHUNK: usize = usize::MAX;

pub struct BatchScheduler<M: BatchModel> {
    model: M,
    slots: Vec<Slot>,
    queue: VecDeque<SlotRequest>,
    sampler: Box<dyn Sampler>,
    stats: SchedulerStats,
    eos: u32,
}

/// Результат одного шага scheduler'а.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// Нечего делать (нет активных слотов/очереди).
    Idle,
    /// Выполнен prefill-чанк (возможно с emit'ом первого сгенерированного токена).
    DidPrefill { first_token_emitted: bool },
    /// Выполнен batched decode-шаг (B = batch size).
    DidDecode(usize),
}

impl<M: BatchModel> BatchScheduler<M> {
    pub fn new(model: M, num_slots: usize, eos: u32, _vocab: usize) -> Self {
        Self {
            model,
            slots: (0..num_slots).map(Slot::new).collect(),
            queue: VecDeque::new(),
            sampler: Box::new(GreedySampler),
            stats: SchedulerStats::default(),
            eos,
        }
    }

    /// Подать запрос; возвращает индекс admit'нутого слота, либо None если ставился в очередь.
    pub fn submit(&mut self, prompt: Vec<u32>, max_new: usize) -> Option<usize> {
        let req = SlotRequest {
            prompt,
            max_new,
            eos: self.eos,
        };
        if let Some(idx) = self.idle_slot() {
            self.slots[idx].admit(req);
            return Some(idx);
        }
        self.queue.push_back(req);
        None
    }

    fn idle_slot(&self) -> Option<usize> {
        self.slots
            .iter()
            .find(|s| s.status == SlotStatus::Idle)
            .map(|s| s.idx)
    }

    /// Один шаг scheduler'а. НЕ сбрасывает Finished-слоты (сбор+reset делает caller).
    pub fn step(&mut self) -> Result<StepOutcome> {
        self.admit_from_queue();

        // 1. Prefill phase: один чанк для одного Prefilling-слота.
        if let Some((sidx, chunk_size)) = self.next_prefill_chunk() {
            let reset = self.slots[sidx].prefill_done == 0;
            let tokens = self.slots[sidx].prefill_remaining_slice()[..chunk_size].to_vec();
            let start_pos = self.slots[sidx].prefill_done;
            let chunk = PrefillChunk {
                slot_idx: sidx,
                reset_first: reset,
                tokens,
                start_pos,
            };
            let t0 = Instant::now();
            let logits = self.model.prefill_chunk(&chunk)?;
            self.stats.prefill_ns += t0.elapsed().as_nanos();
            self.stats.prefill_chunks += 1;
            self.slots[sidx].advance_prefill(chunk_size);
            // Если prompt исчерпан — сэмплируем первый сгенерированный токен
            // из логитов последнего токена prefill (избегаем повторной обработки
            // последнего токена prompt'а в рекуррентном state).
            let mut first_emitted = false;
            if self.slots[sidx].is_decoding() {
                let tok = self.sampler.sample(&logits);
                self.stats.total_decode_tokens += 1;
                self.slots[sidx].push_token(tok);
                first_emitted = true;
            }
            return Ok(StepOutcome::DidPrefill {
                first_token_emitted: first_emitted,
            });
        }

        // 2. Decode phase: батч всех Decoding-слотов (ещё Decoding, не Finished).
        let items: Vec<DecodeItem> = self
            .slots
            .iter()
            .filter(|s| s.is_decoding())
            .map(|s| DecodeItem {
                slot_idx: s.idx,
                token: s.current_token(),
                pos: s.next_pos(),
            })
            .collect();
        if items.is_empty() {
            return Ok(StepOutcome::Idle);
        }
        let batch = DecodeBatch { items };
        let b = batch.len();
        self.stats.max_concurrent_decode = self.stats.max_concurrent_decode.max(b);
        let t0 = Instant::now();
        let logits_vec = self.model.decode_batch(&batch)?;
        self.stats.decode_ns += t0.elapsed().as_nanos();
        self.stats.decode_steps += 1;
        for (it, logits) in batch.items.iter().zip(logits_vec.iter()) {
            let tok = self.sampler.sample(logits);
            self.stats.total_decode_tokens += 1;
            self.slots[it.slot_idx].push_token(tok);
        }
        Ok(StepOutcome::DidDecode(b))
    }

    fn admit_from_queue(&mut self) {
        while self.queue.front().is_some() {
            let Some(idx) = self.idle_slot() else { break };
            let req = self.queue.pop_front().unwrap();
            self.slots[idx].admit(req);
        }
    }

    fn next_prefill_chunk(&self) -> Option<(usize, usize)> {
        for s in &self.slots {
            if s.is_prefilling() {
                let n = s.prefill_remaining_slice().len().min(PREFILL_CHUNK);
                return Some((s.idx, n));
            }
        }
        None
    }

    fn all_idle(&self) -> bool {
        self.slots.iter().all(|s| s.status == SlotStatus::Idle)
    }

    /// Полный прогон: принимает prompts, прогоняет batched, возвращает сгенерированные
    /// токены per prompt в порядке submit. Собирает outputs из Finished-слотов ДО
    /// сброса (generated валиден только в Finished).
    pub fn run_with_collection(
        &mut self,
        prompts: Vec<Vec<u32>>,
        max_new: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let n = prompts.len();
        for p in prompts {
            self.submit(p, max_new);
        }
        let mut slot_order: Vec<Option<usize>> = vec![None; self.slots.len()];
        let mut next_order = 0usize;
        let mut outputs: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        let t0 = Instant::now();
        loop {
            // admit assigns order; Idle slots без order ждут admit.
            for s in &self.slots {
                if s.status != SlotStatus::Idle && slot_order[s.idx].is_none() {
                    slot_order[s.idx] = Some(next_order);
                    next_order += 1;
                }
            }
            // Собрать outputs из Finished (ДО reset) — generated ещё жив.
            let finished: Vec<(usize, usize)> = self
                .slots
                .iter()
                .filter(|s| s.status == SlotStatus::Finished)
                .filter_map(|s| slot_order[s.idx].map(|o| (s.idx, o)))
                .collect();
            for (sidx, order) in &finished {
                outputs[*order] = self.slots[*sidx].generated_tokens().to_vec();
                self.slots[*sidx].reset();
                slot_order[*sidx] = None; // освободить order для пере-admit'а.
            }
            match self.step()? {
                StepOutcome::Idle if self.all_idle() && self.queue.is_empty() => break,
                _ => {}
            }
        }
        self.stats.wall_ns = t0.elapsed().as_nanos();
        Ok(outputs)
    }

    /// Полный прогон с per-request `max_new`: каждый промпт имеет собственный
    /// лимит генерации (в отличие от `run_with_collection`, где один max_new на
    /// все). Используется для регрессионных тестов batch shrink: один слот
    /// короче остальных → раннее завершение → batch сжимается.
    ///
    /// `prompts[i].1` = max_new для i-го запроса. Возвращает сгенерированные
    /// токены per prompt в порядке submit.
    pub fn run_with_per_request_max(
        &mut self,
        prompts: Vec<(Vec<u32>, usize)>,
    ) -> Result<Vec<Vec<u32>>> {
        let n = prompts.len();
        for (p, m) in prompts {
            self.submit(p, m);
        }
        let mut slot_order: Vec<Option<usize>> = vec![None; self.slots.len()];
        let mut next_order = 0usize;
        let mut outputs: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        loop {
            for s in &self.slots {
                if s.status != SlotStatus::Idle && slot_order[s.idx].is_none() {
                    slot_order[s.idx] = Some(next_order);
                    next_order += 1;
                }
            }
            let finished: Vec<(usize, usize)> = self
                .slots
                .iter()
                .filter(|s| s.status == SlotStatus::Finished)
                .filter_map(|s| slot_order[s.idx].map(|o| (s.idx, o)))
                .collect();
            for (sidx, order) in &finished {
                outputs[*order] = self.slots[*sidx].generated_tokens().to_vec();
                self.slots[*sidx].reset();
                slot_order[*sidx] = None;
            }
            match self.step()? {
                StepOutcome::Idle if self.all_idle() && self.queue.is_empty() => break,
                _ => {}
            }
        }
        Ok(outputs)
    }

    /// Baseline: каждый prompt по одному через single-slot scheduler.
    /// Паритет: batched (N slots) == sequential per prompt.
    pub fn sequential_reference(
        model_factory: impl Fn() -> M,
        prompts: Vec<Vec<u32>>,
        max_new: usize,
        eos: u32,
        vocab: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let mut outputs = Vec::with_capacity(prompts.len());
        for p in prompts {
            let mut s = BatchScheduler::new(model_factory(), 1, eos, vocab);
            let mut o = s.run_with_collection(vec![p], max_new)?;
            outputs.append(&mut o);
        }
        Ok(outputs)
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }
    pub fn model(&self) -> &M {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GreedySampler, MockRecurrentModel, PrefillChunk};

    fn prompts() -> Vec<Vec<u32>> {
        vec![
            vec![7, 11, 13, 17],
            vec![3, 5],
            vec![29, 31, 37, 41, 43, 47],
            vec![2],
        ]
    }

    #[test]
    fn batched_equals_sequential_parity() {
        let n_slots = 4;
        let max_new = 12;
        let eos = u32::MAX; // не достигается → гасим по max_new
        let vocab = 4096;
        let p = prompts();
        let mut sched =
            BatchScheduler::new(MockRecurrentModel::new(n_slots, vocab), n_slots, eos, vocab);
        let batched = sched
            .run_with_collection(p.clone(), max_new)
            .expect("batched");
        let seq = BatchScheduler::sequential_reference(
            || MockRecurrentModel::new(1, vocab),
            p,
            max_new,
            eos,
            vocab,
        )
        .expect("seq");
        assert_eq!(batched.len(), seq.len());
        for (i, (b, s)) in batched.iter().zip(seq.iter()).enumerate() {
            assert_eq!(
                b, s,
                "parity fail на prompt {}: batched {:?} vs seq {:?}",
                i, b, s
            );
        }
    }

    #[test]
    fn batched_more_requests_than_slots_recycles() {
        let n_slots = 2;
        let vocab = 1024;
        let max_new = 3;
        let eos = u32::MAX;
        let p: Vec<Vec<u32>> = (0..5)
            .map(|i| vec![i as u32 * 7 + 2, i as u32 + 1])
            .collect();
        let mut sched =
            BatchScheduler::new(MockRecurrentModel::new(n_slots, vocab), n_slots, eos, vocab);
        let batched = sched.run_with_collection(p.clone(), max_new).expect("run");
        let seq = BatchScheduler::sequential_reference(
            || MockRecurrentModel::new(1, vocab),
            p,
            max_new,
            eos,
            vocab,
        )
        .expect("seq");
        assert_eq!(batched, seq);
        assert_eq!(batched.len(), 5);
        for o in &batched {
            assert_eq!(o.len(), max_new);
        }
    }

    #[test]
    fn stats_recorded() {
        let vocab = 512;
        let p = prompts();
        let mut sched = BatchScheduler::new(MockRecurrentModel::new(4, vocab), 4, u32::MAX, vocab);
        let _ = sched.run_with_collection(p, 8).unwrap();
        let st = sched.stats();
        assert!(st.decode_steps > 0);
        assert!(st.total_decode_tokens > 0);
        assert!(st.prefill_chunks > 0);
        assert!(st.max_concurrent_decode >= 1);
    }

    #[test]
    fn batched_shrink_parity_early_finish() {
        // Регрессия slot-indirection: один слот короче остальных → batch
        // сжимается (B=2→1) после раннего завершения. Оставшийся слот должен
        // остаться bit-exact vs sequential (state не протекает в чужой slot).
        // До фикса slot indirection это падало: оставшийся slot читал бы
        // чужой batch_idx=0 state вместо своего slot_idx state.
        let n_slots = 2;
        let vocab = 2048;
        let eos = u32::MAX;
        let prompts: Vec<(Vec<u32>, usize)> = vec![
            (vec![7, 11, 13], 10), // длинный
            (vec![3, 5], 2),       // короткий → раннее завершение → shrink
        ];

        let mut sched =
            BatchScheduler::new(MockRecurrentModel::new(n_slots, vocab), n_slots, eos, vocab);
        let batched = sched
            .run_with_per_request_max(prompts.clone())
            .expect("batched");

        // Sequential reference: каждый prompt по одному в свой single-slot scheduler.
        let mut seq = Vec::with_capacity(prompts.len());
        for (p, m) in prompts {
            let mut s = BatchScheduler::new(MockRecurrentModel::new(1, vocab), 1, eos, vocab);
            let mut o = s.run_with_per_request_max(vec![(p, m)]).expect("seq");
            seq.append(&mut o);
        }

        assert_eq!(batched.len(), seq.len());
        for (i, (b, s)) in batched.iter().zip(seq.iter()).enumerate() {
            assert_eq!(
                b, s,
                "shrink parity fail на prompt {}: batched {:?} vs seq {:?}",
                i, b, s
            );
        }
        assert_eq!(batched[0].len(), 10, "длинный слот не досчитал");
        assert_eq!(batched[1].len(), 2, "короткий слот не досчитал");
    }

    #[test]
    fn first_token_comes_from_prefill_logits() {
        // Один слот: prefill полного prompt'а → первый сгенерированный токен
        // сэмплируется ИЗ логитов последнего токена, возвращённых prefill_chunk.
        // Это критично для реальной модели: мы НЕ переобрабатываем последний
        // токен prompt'а в рекуррентном state (повторный mix/forward испортил бы
        // conv/SSM state GDN). Поэтому ожидание строим той же моделью — greedy-
        // sample логитов prefill_chunk — а не хардкожим формулу (устаревший
        // slot-seed дизайн давал seed=1 и неверное 501).
        let vocab = 1024;
        let prompt = vec![42u32, 7, 19];

        // Reference: первый токен = argmax логитов, возвращённых prefill_chunk.
        let mut ref_model = MockRecurrentModel::new(1, vocab);
        let chunk = PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt.clone(),
            start_pos: 0,
        };
        let ref_logits = ref_model.prefill_chunk(&chunk).unwrap();
        let expected = GreedySampler.sample(&ref_logits);

        // Scheduler: один слот, max_new=1 → ровно первый сгенерированный токен.
        let mut sched = BatchScheduler::new(MockRecurrentModel::new(1, vocab), 1, u32::MAX, vocab);
        let out = sched.run_with_collection(vec![prompt.clone()], 1).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0], expected, "первый токен не из prefill-логитов");
    }
}
