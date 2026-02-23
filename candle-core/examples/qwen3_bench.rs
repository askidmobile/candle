/// Benchmark для quantized matmul на Metal — диагностика производительности Qwen3 GGUF.
///
/// Запуск:
///   cargo run --example qwen3_bench --release --features metal
///   cargo run --example qwen3_bench --features metal          # debug для сравнения
///
/// Тестирует:
/// 1. Загрузку GGUF файла на Metal device
/// 2. QMatMul forward при m=1 (авторегрессивная генерация, один токен)
/// 3. QMatMul forward при m=32 (prefill, kernel_mul_mm)
/// 4. Потребление памяти
/// 5. Реальную нагрузку на GPU vs CPU
use anyhow::Result;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Tensor};
use std::time::Instant;

const GGUF_PATH: &str = concat!(
    env!("HOME"),
    "/Library/Application Support/com.yttri.app/models/qwen3-0.6b/qwen3-0.6b-q4_0.gguf"
);

fn main() -> Result<()> {
    println!("=== Candle Quantized MatMul Benchmark ===\n");

    // --- Этап 1: Устройство ---
    let device = Device::new_metal(0)?;
    println!("[1] Device: Metal GPU");

    // --- Этап 2: Загрузка GGUF ---
    println!("\n[2] Загрузка GGUF: {}", GGUF_PATH);
    let t0 = Instant::now();
    let mut file = std::fs::File::open(GGUF_PATH)?;
    let ct = gguf_file::Content::read(&mut file)?;
    let load_time = t0.elapsed();
    println!("    GGUF headers за {:.1}мс", load_time.as_millis());
    println!("    Тензоров: {}", ct.tensor_infos.len());

    // Выводим несколько тензоров для понимания архитектуры
    let mut tensor_names: Vec<_> = ct.tensor_infos.keys().collect();
    tensor_names.sort();
    println!("    Первые 10 тензоров:");
    for name in tensor_names.iter().take(10) {
        let info = &ct.tensor_infos[*name];
        println!("      {} : {:?} {:?}", name, info.shape, info.ggml_dtype);
    }

    // --- Этап 3: Загрузка нескольких весовых тензоров ---
    println!("\n[3] Загрузка тензоров на Metal device...");

    // Загрузим один большой слой (attention_wq первого layer)
    let layer0_prefix = "blk.0";
    let wq_name = format!("{}.attn_q.weight", layer0_prefix);
    let wk_name = format!("{}.attn_k.weight", layer0_prefix);
    let wv_name = format!("{}.attn_v.weight", layer0_prefix);
    let wo_name = format!("{}.attn_output.weight", layer0_prefix);
    let ffn_gate_name = format!("{}.ffn_gate.weight", layer0_prefix);
    let ffn_down_name = format!("{}.ffn_down.weight", layer0_prefix);
    let ffn_up_name = format!("{}.ffn_up.weight", layer0_prefix);

    let t0 = Instant::now();
    let wq = ct.tensor(&mut file, &wq_name, &device)?;
    let wk = ct.tensor(&mut file, &wk_name, &device)?;
    let wv = ct.tensor(&mut file, &wv_name, &device)?;
    let wo = ct.tensor(&mut file, &wo_name, &device)?;
    let ffn_gate = ct.tensor(&mut file, &ffn_gate_name, &device)?;
    let ffn_down = ct.tensor(&mut file, &ffn_down_name, &device)?;
    let ffn_up = ct.tensor(&mut file, &ffn_up_name, &device)?;
    let tensor_load_time = t0.elapsed();

    println!(
        "    Загружено 7 тензоров (1 layer) за {:.1}мс",
        tensor_load_time.as_millis()
    );
    println!("    wq: {:?} dtype={:?}", wq.shape(), wq.dtype());
    println!(
        "    ffn_gate: {:?} dtype={:?}",
        ffn_gate.shape(),
        ffn_gate.dtype()
    );

    // Создаём QMatMul из QTensor
    let t0 = Instant::now();
    let wq_mm = QMatMul::from_qtensor(wq)?;
    let wk_mm = QMatMul::from_qtensor(wk)?;
    let wv_mm = QMatMul::from_qtensor(wv)?;
    let wo_mm = QMatMul::from_qtensor(wo)?;
    let ffn_gate_mm = QMatMul::from_qtensor(ffn_gate)?;
    let ffn_down_mm = QMatMul::from_qtensor(ffn_down)?;
    let ffn_up_mm = QMatMul::from_qtensor(ffn_up)?;
    let qmatmul_time = t0.elapsed();
    println!(
        "    QMatMul::from_qtensor за {:.1}мс",
        qmatmul_time.as_millis()
    );

    // --- Этап 4: Загрузка ВСЕЙ модели ---
    println!("\n[4] Загрузка ВСЕХ тензоров модели...");
    let t0 = Instant::now();
    let mut all_tensors = Vec::new();
    // Перечитаем файл
    let mut file2 = std::fs::File::open(GGUF_PATH)?;
    let ct2 = gguf_file::Content::read(&mut file2)?;
    for name in tensor_names.iter() {
        let qt = ct2.tensor(&mut file2, name, &device)?;
        all_tensors.push(qt);
    }
    let full_load_time = t0.elapsed();
    println!(
        "    Загружено {} тензоров за {:.1}сек",
        all_tensors.len(),
        full_load_time.as_secs_f64()
    );

    // Проверим потребление памяти Metal device
    let metal_dev = match &device {
        Device::Metal(m) => m,
        _ => unreachable!(),
    };
    // К сожалению, candle не имеет API для получения memory stats
    // Но мы можем посмотреть общий размер буферов
    let total_bytes: usize = all_tensors.iter().map(|t| t.storage_size_in_bytes()).sum();
    println!(
        "    Общий размер quantized буферов: {:.1}MB",
        total_bytes as f64 / 1024.0 / 1024.0
    );

    // Теперь проверим что происходит с embedding dequantize
    println!("\n[5] Тест embedding dequantize...");
    let t0 = Instant::now();
    let embed_qt = ct2.tensor(
        &mut std::fs::File::open(GGUF_PATH)?,
        "token_embd.weight",
        &device,
    )?;
    let embed_shape = embed_qt.shape().clone();
    let embed_dtype = embed_qt.dtype();
    let embed_bytes = embed_qt.storage_size_in_bytes();
    println!(
        "    token_embd.weight: {:?} dtype={:?} size={:.1}MB",
        embed_shape,
        embed_dtype,
        embed_bytes as f64 / 1024.0 / 1024.0
    );
    let embed_dequant = embed_qt.dequantize(&device)?;
    let dequant_time = t0.elapsed();
    println!(
        "    dequantize → {:?} dtype={:?} size={:.1}MB за {:.1}мс",
        embed_dequant.shape(),
        embed_dequant.dtype(),
        embed_dequant.elem_count() * 4 / 1024 / 1024,
        dequant_time.as_millis()
    );

    // --- Этап 6: Benchmark QMatMul forward ---
    // Qwen3-0.6B: hidden_size=1024, intermediate_size=2816, num_attention_heads=16, num_kv_heads=8
    let hidden = 1024usize;
    let seq_lens = [1, 4, 16, 32, 64, 128];

    println!(
        "\n[6] Benchmark QMatMul.forward() (wq: [{}, {}] Q4_0)",
        hidden, hidden
    );
    println!(
        "    {:>6} | {:>10} | {:>10} | {:>10}",
        "seq_len", "time_ms", "tok/s_eq", "path"
    );

    for &seq_len in &seq_lens {
        let input = Tensor::randn(0f32, 1.0, (seq_len, hidden), &device)?;

        // Прогрев
        let _ = wq_mm.forward(&input)?;
        device.synchronize()?;

        // Замер
        let n_iters = if seq_len <= 4 { 100 } else { 20 };
        let t0 = Instant::now();
        for _ in 0..n_iters {
            let _ = wq_mm.forward(&input)?;
        }
        device.synchronize()?;
        let elapsed = t0.elapsed();

        let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / n_iters as f64;
        let tok_per_sec = seq_len as f64 / (per_iter_ms / 1000.0);
        let path = if seq_len > 1 {
            "kernel_mul_mm"
        } else {
            "kernel_mul_mv"
        };

        println!(
            "    {:>6} | {:>10.2} | {:>10.1} | {}",
            seq_len, per_iter_ms, tok_per_sec, path
        );
    }

    // --- Этап 7: Симуляция полного forward одного layer ---
    // Qwen3-0.6B: hidden=1024, num_heads=16, kv_heads=8, head_dim=128
    // wq: [2048, 1024] (num_heads * head_dim = 2048)
    // wk: [1024, 1024] (kv_heads * head_dim = 1024)
    // wv: [1024, 1024]
    // wo: [1024, 2048] (hidden, num_heads * head_dim)
    // ffn_gate: [3072, 1024], ffn_up: [3072, 1024], ffn_down: [1024, 3072]
    println!("\n[7] Симуляция полного forward layer 0 (7 matmul) при m=1");
    let input_h = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
    let input_attn_out = Tensor::randn(0f32, 1.0, (1, 2048), &device)?; // num_heads * head_dim
    let input_ffn_down = Tensor::randn(0f32, 1.0, (1, 3072), &device)?; // intermediate_size

    // Прогрев
    let _ = wq_mm.forward(&input_h)?;
    let _ = wk_mm.forward(&input_h)?;
    let _ = wv_mm.forward(&input_h)?;
    let _ = wo_mm.forward(&input_attn_out)?;
    let _ = ffn_gate_mm.forward(&input_h)?;
    let _ = ffn_up_mm.forward(&input_h)?;
    let _ = ffn_down_mm.forward(&input_ffn_down)?;
    device.synchronize()?;

    let n_iters = 200;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = wq_mm.forward(&input_h)?;
        let _ = wk_mm.forward(&input_h)?;
        let _ = wv_mm.forward(&input_h)?;
        let _ = wo_mm.forward(&input_attn_out)?;
        let _ = ffn_gate_mm.forward(&input_h)?;
        let _ = ffn_up_mm.forward(&input_h)?;
        let _ = ffn_down_mm.forward(&input_ffn_down)?;
    }
    device.synchronize()?;
    let elapsed = t0.elapsed();
    let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / n_iters as f64;
    // Qwen3-0.6B: 28 layers
    let estimated_full_forward_ms = per_iter_ms * 28.0;
    let estimated_tok_per_sec = 1000.0 / estimated_full_forward_ms;

    println!("    1 layer (7 matmul): {:.2}мс", per_iter_ms);
    println!(
        "    28 layers (полная модель): {:.1}мс → {:.1} tok/s (оценка, только matmul)",
        estimated_full_forward_ms, estimated_tok_per_sec
    );

    // --- Этап 8: Сравнение CPU vs Metal для одного QMatMul ---
    println!("\n[8] CPU vs Metal: загрузка одного тензора и forward");

    let cpu_device = Device::Cpu;
    let wq_cpu = ct2.tensor(&mut std::fs::File::open(GGUF_PATH)?, &wq_name, &cpu_device)?;
    let wq_cpu_mm = QMatMul::from_qtensor(wq_cpu)?;

    let input_cpu = Tensor::randn(0f32, 1.0, (1, hidden), &cpu_device)?;
    // Прогрев
    let _ = wq_cpu_mm.forward(&input_cpu)?;

    let n_iters = 100;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = wq_cpu_mm.forward(&input_cpu)?;
    }
    let cpu_elapsed = t0.elapsed();
    let cpu_per_iter = cpu_elapsed.as_secs_f64() * 1000.0 / n_iters as f64;

    let input_metal = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
    let _ = wq_mm.forward(&input_metal)?;
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = wq_mm.forward(&input_metal)?;
    }
    device.synchronize()?;
    let metal_elapsed = t0.elapsed();
    let metal_per_iter = metal_elapsed.as_secs_f64() * 1000.0 / n_iters as f64;

    println!("    CPU:   {:.3}мс/iter", cpu_per_iter);
    println!("    Metal: {:.3}мс/iter", metal_per_iter);
    println!("    Speedup: {:.1}x", cpu_per_iter / metal_per_iter);

    // --- Этап 9: Диагностика QMatMul variant ---
    println!("\n[9] Диагностика QMatMul variant:");
    match &wq_mm {
        QMatMul::QTensor(_) => println!("    wq: QMatMul::QTensor (quantized Metal kernel) ✓"),
        QMatMul::Tensor(_) => {
            println!("    wq: QMatMul::Tensor (DEQUANTIZED f32!) ✗ — это проблема!")
        }
        QMatMul::TensorF16(_) => println!("    wq: QMatMul::TensorF16 (dequantized f16) ✗"),
    }
    match &ffn_gate_mm {
        QMatMul::QTensor(_) => {
            println!("    ffn_gate: QMatMul::QTensor (quantized Metal kernel) ✓")
        }
        QMatMul::Tensor(_) => println!("    ffn_gate: QMatMul::Tensor (DEQUANTIZED f32!) ✗"),
        QMatMul::TensorF16(_) => println!("    ffn_gate: QMatMul::TensorF16 (dequantized f16) ✗"),
    }

    // --- Этап 10: Benchmark типичных не-matmul операций ---
    println!("\n[10] Benchmark не-matmul операций (bottleneck диагностика)");

    // RMSNorm: normalize + mul (element-wise)
    let norm_weight = Tensor::ones((hidden,), DType::F32, &device)?;
    let x = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;

    // Прогрев
    let _ = {
        let var = x.sqr()?.mean_keepdim(1)?;
        let eps = 1e-6;
        let x_norm = x.broadcast_div(&(var + eps)?.sqrt()?)?;
        x_norm.broadcast_mul(&norm_weight)?
    };
    device.synchronize()?;

    let n_iters = 500;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let var = x.sqr()?.mean_keepdim(1)?;
        let eps = 1e-6;
        let x_norm = x.broadcast_div(&(var + eps)?.sqrt()?)?;
        let _ = x_norm.broadcast_mul(&norm_weight)?;
    }
    device.synchronize()?;
    let rmsnorm_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!("    RMSNorm (1, {}): {:.3}мс", hidden, rmsnorm_ms);

    // Softmax (ручная реализация как в candle-nn)
    let attn_scores = Tensor::randn(0f32, 1.0, (16, 1, 128), &device)?; // 16 heads, 1 query, 128 kv
    let softmax_fn = |t: &Tensor| -> Result<Tensor> {
        let max = t.max_keepdim(2)?;
        let shifted = t.broadcast_sub(&max)?;
        let exp = shifted.exp()?;
        let sum = exp.sum_keepdim(2)?;
        Ok(exp.broadcast_div(&sum)?)
    };
    let _ = softmax_fn(&attn_scores)?;
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = softmax_fn(&attn_scores)?;
    }
    device.synchronize()?;
    let softmax_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!("    Softmax (16, 1, 128): {:.3}мс", softmax_ms);

    // Contiguous/Reshape — часто скрытая стоимость
    let y = Tensor::randn(0f32, 1.0, (1, 16, 1, 64), &device)?;
    let _ = y.transpose(1, 2)?.contiguous()?;
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = y.transpose(1, 2)?.contiguous()?;
    }
    device.synchronize()?;
    let transpose_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!(
        "    Transpose+contiguous (1,16,1,64): {:.3}мс",
        transpose_ms
    );

    // Tensor creation overhead (embedding lookup simulation)
    let embed_table = Tensor::randn(0f32, 1.0, (1000, hidden), &device)?;
    let idx = Tensor::new(&[42u32], &device)?;
    let _ = embed_table.index_select(&idx, 0)?;
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = embed_table.index_select(&idx, 0)?;
    }
    device.synchronize()?;
    let embed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!("    Embedding lookup (1 token): {:.3}мс", embed_ms);

    // MatMul (non-quantized, для attention Q*K^T)
    let q = Tensor::randn(0f32, 1.0, (16, 1, 64), &device)?; // heads, seq, head_dim
    let k = Tensor::randn(0f32, 1.0, (16, 64, 128), &device)?; // heads, head_dim, kv_len
    let _ = q.matmul(&k)?;
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = q.matmul(&k)?;
    }
    device.synchronize()?;
    let attn_qk_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!(
        "    Attention Q*K^T (16, 1, 64) x (16, 64, 128): {:.3}мс",
        attn_qk_ms
    );

    // SiLU activation (tensor.silu() — использует Metal шейдер usilu)
    let gate_out = Tensor::randn(0f32, 1.0, (1, 3072), &device)?;
    let _ = gate_out.silu()?;
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = gate_out.silu()?;
    }
    device.synchronize()?;
    let silu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!("    SiLU (1, 3072): {:.3}мс", silu_ms);

    // Element-wise multiply (gate * up)
    let up_out = Tensor::randn(0f32, 1.0, (1, 3072), &device)?;
    let _ = (&gate_out * &up_out)?;
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = (&gate_out * &up_out)?;
    }
    device.synchronize()?;
    let mul_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!("    Mul (1, 3072) * (1, 3072): {:.3}мс", mul_ms);

    // Residual add
    let res = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
    let out = Tensor::randn(0f32, 1.0, (1, hidden), &device)?;
    let _ = (&res + &out)?;
    device.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = (&res + &out)?;
    }
    device.synchronize()?;
    let add_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
    println!("    Add (1, {}): {:.3}мс", hidden, add_ms);

    // Суммарная оценка одного layer
    let total_layer_ms = rmsnorm_ms * 2.0 // pre-attn + pre-ffn norm
        + 0.05 // 7 matmul (из этапа 7, release)
        + softmax_ms
        + attn_qk_ms * 2.0 // Q*K^T + attn*V
        + transpose_ms * 4.0 // Q,K,V reshape + output reshape
        + silu_ms
        + mul_ms
        + add_ms * 2.0; // residual в attn + ffn

    println!(
        "\n    Оценка 1 layer (все операции): {:.2}мс",
        total_layer_ms
    );
    println!(
        "    Оценка 28 layers: {:.1}мс → {:.1} tok/s",
        total_layer_ms * 28.0,
        1000.0 / (total_layer_ms * 28.0)
    );

    println!("\n=== Benchmark завершён ===");
    Ok(())
}
