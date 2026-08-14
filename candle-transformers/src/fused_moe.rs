// Adapted from: https://github.com/guoqingbao/vllm.rs/blob/main/src/models/layers/moe.rs
use candle::Module;
use candle::{quantized::QTensor, DType, Result, Tensor, D};
use candle_nn::{Activation, Linear};
use std::sync::Arc;

pub struct MoeCfg {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub act: Activation,
    pub decoder_sparse_step: Option<usize>,
}

// Dense FusedMoe (FFI moe_gemm) удалён: libmoe.a не собирается под dynamic-loading
// (T-331), альтернативного PTX-пути для dense MoE нет. Dense MoE-модели (qwen3-moe)
// используют naive expert-loop через стандартный matmul (см. Qwen3SparseMoeBlock).
// Quantized GGUF MoE — FusedMoeGGUF ниже (PTX-путь QTensor::indexed_moe_forward).

pub struct FusedMoeGGUF {
    pub gate: Linear,
    pub gate_experts: Arc<QTensor>,
    pub up_experts: Arc<QTensor>,
    pub down_experts: Arc<QTensor>,
    pub act: Activation,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    // all_reduce: AllReduce,
    // world_size: usize,
    pub dtype: DType,
}

impl FusedMoeGGUF {
    pub fn new(
        cfg: &MoeCfg,
        vb: crate::quantized_var_builder::VarBuilder,
        dtype: DType,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts;
        let gate_ws = vb
            .pp("ffn_gate_inp")
            .get((num_experts, cfg.hidden_size), "weight")?
            .dequantize(vb.device())?
            .to_dtype(DType::F32)?;

        let gate = Linear::new(gate_ws, None);

        let (gate_experts, up_experts, down_experts) = {
            (
                vb.pp("ffn_gate_exps").get(
                    (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                )?,
                vb.pp("ffn_up_exps").get(
                    (num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                )?,
                vb.pp("ffn_down_exps").get(
                    (num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
                    "weight",
                )?,
            )
        };

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.act,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
            // all_reduce: AllReduce::new(comm),
            // world_size: 1,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, _is_prefill: bool) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs)?;

        let routing_weights =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        let topk_ids = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            // PTX-путь: QTensor::indexed_moe_forward (dynamic-loading-совместимый).
            // ids = topk_ids [num_tokens, topk] — expert id для каждого (token, topk-slot).
            // Сортировка sorted_token_ids не нужна — ядро индексирует weights через ids напрямую.
            // input [num_tokens, 1, hidden] (input_dim1=1 → broadcast: все topk используют тот же input).
            let xs_3d = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self.gate_experts.indexed_moe_forward(&xs_3d, &topk_ids)?;
            let up = self.up_experts.indexed_moe_forward(&xs_3d, &topk_ids)?;

            // down_inputs [num_tokens, topk, intermediate] (input_dim1=topk → каждая
            // (token, topk) позиция использует свой input).
            let down_inputs = (up * gate.apply(&self.act)?)?;
            let down = self.down_experts.indexed_moe_forward(&down_inputs, &topk_ids)?;

            // Применяем topk_weights [num_tokens, topk] поверх [num_tokens, topk, hidden]
            // (FFI-путь делал это внутри ядра). sum по topk dim → [num_tokens, hidden].
            let topk_w = topk_weights.unsqueeze(D::Minus1)?;
            (down * topk_w)?
        };
        let mut ys = ys.sum(D::Minus2)?;
        if ys.dtype() != original_dtype {
            ys = ys.to_dtype(original_dtype)?;
        }
        ys.reshape((batch, seq_len, hidden_dim))
    }
}
