// Фаза 3: dense FusedMoe (FFI) удалён. Quantized GGUF MoE работает через
// FusedMoeGGUF + QTensor::indexed_moe_forward (PTX-путь). Dense MoE-модели
// (qwen3_moe) используют naive expert-loop (Qwen3SparseMoeBlock, matmul).
//
// ponytail: полный CUDA/GGUF-тест требует GPU + GGUF weights — вне scope self-check.
// Когда появится CUDA CI: добавить тест на Q4K MoE с CPU reference comparison
// через FusedMoeGGUF + QTensor::indexed_moe_forward, и тест на naive dense MoE
// (Qwen3SparseMoeBlock) с CPU reference.

#[test]
fn placeholder_no_dense_fused_moe() {
    // Dense FusedMoe удалён; naive expert-loop тестируется косвенно через
    // qwen3_moe model tests. Этот файл оставлен как маркер для будущих MoE-тестов.
    assert!(true);
}