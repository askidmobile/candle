//! Интеграционный тест реальной Qwen3.5-4B GGUF через `real-model` адаптер.
//!
//! Запуск:
//! ```sh
//! YTTRI_MODEL_DIR=/Volumes/Askid\ Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b \
//! cargo test -p qwen35-batch --features real-model \
//!     --test real_qwen35_batch --release -- --ignored --nocapture
//! ```
//!
//! Проверяем:
//! 1. B=1 batched-path == single-stream greedy (бит-точность — time-multiplexed
//!    restore/snapshot изолирует state; тот же forward-путь, те же веса).
//! 2. B=2/4: aggregate tok/s vs sequential, per-request tok/s.
//! 3. Greedy parity B=4 vs B=1 (batched vs sequential).
//!
//! Путь: `Qwen35BatchAdapter` (time-multiplexed через snapshot/restore) под
//! `BatchScheduler`. Это НЕ true batched decode (Metal-ядра DeltaNet без batch-оси),
//! а measurement архитектурного штрафа Candle за multi-slot — см. adapter.rs.

#![cfg(feature = "real-model")]

use std::path::PathBuf;
use std::time::Instant;

use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::scheduler::BatchScheduler;
use qwen35_batch::{BatchModel, DEFAULT_NUM_SLOTS};

fn model_dir() -> PathBuf {
    std::env::var("YTTRI_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b",
            )
        })
}

fn gguf_path() -> PathBuf {
    model_dir().join("Qwen3.5-4B-Q4_K_M.gguf")
}

fn accelerator_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    {
        return candle_core::Device::new_cuda(0).expect("CUDA device");
    }
    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    {
        qwen35_batch::real::metal_utils::configure_metal_env();
        let device = candle_core::Device::new_metal(0).expect("Metal device");
        qwen35_batch::real::metal_utils::metal_probe(&device).expect("Metal probe");
        return device;
    }
    #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
    panic!("real Qwen3.5 benchmark requires feature `cuda` on this platform");
}

/// Детерминированные token IDs для parity (содержательно не важно — важна
/// идентичность batched vs sequential путей). Для качественной генерации
/// нужен настоящий tokenizer (см. REAL_MODEL.md §7).
fn dummy_prompt_ids(seed: u32, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (seed.wrapping_add(i as u32 * 7) % 1000) + 10)
        .collect()
}

#[test]
#[ignore = "требует GGUF Qwen3.5-4B на диске + GPU (долгая загрузка ~2.5 ГБ)"]
fn real_qwen35_load_and_single_forward() {
    let _ = env_logger::try_init();
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);

    let device = accelerator_device();

    let mut adapter =
        Qwen35BatchAdapter::load(&gguf, device, DEFAULT_NUM_SLOTS).expect("load Qwen35 adapter");
    let vocab = adapter.vocab_size();
    eprintln!("[real] loaded, vocab={vocab}, eos={}", adapter.eos());

    use qwen35_batch::model::PrefillChunk;
    let prompt = dummy_prompt_ids(42, 8);
    let logits = adapter
        .prefill_chunk(&PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt,
            start_pos: 0,
        })
        .expect("prefill");
    assert_eq!(logits.len(), vocab, "logits size != vocab");
    eprintln!("[real] prefill logits len={} OK", logits.len());
}

#[derive(Debug)]
struct BenchResult {
    batch: usize,
    outputs: Vec<Vec<u32>>,
    wall_ns: u128,
    prefill_ns: u128,
    decode_ns: u128,
    decode_steps: usize,
    max_concurrent: usize,
    rss_mb: f64,
    vram_mb: f64,
    vram_delta_mb: f64,
}

#[cfg(unix)]
fn current_rss_mb() -> f64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output();
    output
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb as f64 / 1024.0)
        .unwrap_or(f64::NAN)
}

#[cfg(windows)]
fn current_rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    let script = format!("(Get-Process -Id {pid}).WorkingSet64");
    std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|bytes| bytes as f64 / 1024.0 / 1024.0)
        .unwrap_or(f64::NAN)
}

/// Общая занятая VRAM GPU по nvidia-smi. Это не process-exclusive метрика
/// (Windows WDDM не всегда отдаёт per-process memory), поэтому дополнительно
/// сохраняем delta относительно значения непосредственно перед load adapter.
fn current_vram_mb() -> f64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next()?.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn run_bench(
    gguf: &std::path::Path,
    device: &candle_core::Device,
    prompts: &[Vec<u32>],
    max_new: usize,
    batch: usize,
) -> BenchResult {
    // Критично: загрузка модели находится ВНЕ таймера. Старый baseline создавал
    // fresh adapter внутри цикла по prompt'ам и ошибочно засчитывал 4 загрузки
    // 2.7-ГБ GGUF в sequential wall time, получая фиктивные 2.13× для B=4.
    let vram_before_mb = current_vram_mb();
    let adapter = Qwen35BatchAdapter::load(gguf, device.clone(), batch)
        .unwrap_or_else(|e| panic!("load adapter B={batch}: {e}"));
    let vocab = adapter.vocab_size();
    let mut sched = BatchScheduler::new(adapter, batch, u32::MAX, vocab);

    let t0 = Instant::now();
    let outputs = sched
        .run_with_collection(prompts.to_vec(), max_new)
        .unwrap_or_else(|e| panic!("run B={batch}: {e}"));
    let wall_ns = t0.elapsed().as_nanos();
    let stats = sched.stats().clone();
    let rss_mb = current_rss_mb();
    let vram_mb = current_vram_mb();
    let vram_delta_mb = vram_mb - vram_before_mb;

    BenchResult {
        batch,
        outputs,
        wall_ns,
        prefill_ns: stats.prefill_ns,
        decode_ns: stats.decode_ns,
        decode_steps: stats.decode_steps,
        max_concurrent: stats.max_concurrent_decode,
        rss_mb,
        vram_mb,
        vram_delta_mb,
    }
}

/// Регрессия slot-indirection (Phase 6 quality-gate fix): batch shrink.
///
/// Два промпта декодируются одновременно (B=2): один длинный (max_new=12),
/// другой короткий (max_new=3). Короткий завершается первым → batch сжимается
/// (B=2→1). Длинный слот продолжает в одиночном батче. Без slot indirection
/// оставшийся слот читал бы чужой persistent state (batch_idx=0 → чужой slot)
/// и дрейфовал — именно это ломало quality-gate case3.
///
/// Паритет: длинный слот batched == sequential (его собственный B=1 прогон).
/// Короткий слот тоже проверяем для полноты.
#[test]
#[ignore = "требует GGUF + GPU; регрессия slot-indirection (batch shrink)"]
fn real_qwen35_batched_equals_sequential_parity_shrink() {
    let _ = env_logger::try_init();
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);

    let device = accelerator_device();

    // Два разных промпта с разной длиной генерации → batch shrink.
    let prompts: Vec<(Vec<u32>, usize)> = vec![
        (dummy_prompt_ids(42, 8), 12), // длинный
        (dummy_prompt_ids(137, 5), 3), // короткий → раннее завершение → shrink
    ];

    // --- Batched: оба промпта в одном 2-slot scheduler'е (batch shrink в decode) ---
    let adapter_b = Qwen35BatchAdapter::load(&gguf, device.clone(), 2)
        .expect("load batched");
    let vocab = adapter_b.vocab_size();
    let mut sched_b = BatchScheduler::new(adapter_b, 2, u32::MAX, vocab);
    let batched = sched_b
        .run_with_per_request_max(prompts.clone())
        .expect("batched");

    // --- Sequential: каждый промпт в свой single-slot scheduler (B=1) ---
    let mut seq = Vec::with_capacity(prompts.len());
    for (p, m) in prompts {
        let adapter = Qwen35BatchAdapter::load(&gguf, device.clone(), 1)
            .expect("load seq");
        let mut s = BatchScheduler::new(adapter, 1, u32::MAX, vocab);
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
    eprintln!(
        "[shrink-parity] длинный(B=2→1) и короткий совпали с sequential: BIT-EXACT OK"
    );
    eprintln!("  длинный слот: {} токенов", batched[0].len());
    eprintln!("  короткий слот: {} токенов", batched[1].len());
    eprintln!(
        "  batch shrunk: {} decode_steps (B=2 + B=1 после shrink), max_concurrent={}",
        sched_b.stats().decode_steps,
        sched_b.stats().max_concurrent_decode
    );
}

#[test]
#[ignore = "требует GGUF + GPU; долгий parity+bench прогон"]
fn real_qwen35_batched_equals_sequential_parity() {
    let _ = env_logger::try_init();
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);

    let device = accelerator_device();

    let max_new = 16;
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|s| dummy_prompt_ids(s * 13 + 1, 6 + (s as usize)))
        .collect();
    let total_tokens = max_new * prompts.len();
    let decode_only_tokens = (max_new - 1) * prompts.len();

    // B=1 — честный sequential reference: одна модель, один слот, очередь из 4
    // запросов. Затем B=2/B=4, каждый с одной загрузкой вне измеряемого участка.
    let mut results = Vec::new();
    for batch in [1usize, 2, 4] {
        results.push(run_bench(&gguf, &device, &prompts, max_new, batch));
    }

    let reference = &results[0].outputs;
    for result in &results[1..] {
        assert_eq!(
            result.outputs, *reference,
            "greedy parity B={} vs B=1 нарушена",
            result.batch
        );
    }
    eprintln!("[parity] B=1/2/4 greedy outputs: BIT-EXACT OK");

    let baseline_tps = total_tokens as f64 / (results[0].wall_ns as f64 / 1e9);
    eprintln!(
        "[bench] fair matrix: loads excluded; {} prompts × {} generated tokens",
        prompts.len(),
        max_new
    );
    eprintln!(
        "[bench] B | wall_ms | aggregate_tok/s | per_request_tok/s | decode_only_tok/s | vs_B1 | peak_concurrent | RSS_MB | VRAM_MB | VRAM_delta_MB"
    );
    for r in &results {
        let aggregate_tps = total_tokens as f64 / (r.wall_ns as f64 / 1e9);
        let per_request_tps = aggregate_tps / r.max_concurrent.max(1) as f64;
        let decode_tps = decode_only_tokens as f64 / (r.decode_ns as f64 / 1e9);
        eprintln!(
            "[bench] {} | {:.1} | {:.2} | {:.2} | {:.2} | {:.3}x | {} | {:.1} | {:.1} | {:+.1}",
            r.batch,
            r.wall_ns as f64 / 1e6,
            aggregate_tps,
            per_request_tps,
            decode_tps,
            aggregate_tps / baseline_tps,
            r.max_concurrent,
            r.rss_mb,
            r.vram_mb,
            r.vram_delta_mb,
        );
        eprintln!(
            "[bench-detail] B={} prefill_ms={:.1} decode_ms={:.1} decode_steps={}",
            r.batch,
            r.prefill_ns as f64 / 1e6,
            r.decode_ns as f64 / 1e6,
            r.decode_steps,
        );
    }
}
