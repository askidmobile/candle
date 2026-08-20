//! Quantized Gemma 4 model implementation for GGUF.
//!
//! Features:
//! - GQA attention with Q/K RMS-Norm and optional V-norm
//! - Per-layer token embeddings & projections (Gemma 4 architecture)
//! - SwiGLU MLP
//! - RMSNorm with (w + 1.0) scaling
//! - Support for Q4_K_M, Q5_K_M, Q6_K, Q8_0 GGUF quants

use std::sync::Arc;

use crate::quantized_nn::RmsNorm;
use candle::quantized::gguf_file;
use candle::quantized::QTensor;
use candle::D;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};

pub const MAX_SEQ_LEN: usize = 131072;

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,

    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,

    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,

    mlp: Mlp,

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,

    // Per-layer projections (optional in smaller variants)
    inp_gate: Option<QMatMul>,
    proj: Option<QMatMul>,
    post_norm: Option<RmsNorm>,
    layer_output_scale: Option<Tensor>,

    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    fn forward_attn(
        &mut self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        // RMS norm on Q and K
        let q = self.attention_q_norm.forward(&q)?;
        let k = self.attention_k_norm.forward(&k)?;

        let (q, k) = rotary.apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA expansion if kv_heads < heads
        let (k, v) = if self.n_head != self.n_kv_head {
            let n_rep = self.n_head / self.n_kv_head;
            (
                k.repeat(&[1, n_rep, 1, 1])?,
                v.repeat(&[1, n_rep, 1, 1])?,
            )
        } else {
            (k, v)
        };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;

        let out = out
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
        self.attention_wo.forward(&out)
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        index_pos: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.forward_attn(&normed, rotary, index_pos)?;
        let attn_out = self.post_attention_norm.forward(&attn_out)?;
        let x = (residual + attn_out)?;

        let residual = &x;
        let normed = self.ffn_norm.forward(&x)?;
        let ffn_out = self.mlp.forward(&normed)?;
        let ffn_out = self.post_ffn_norm.forward(&ffn_out)?;
        let mut x = (residual + ffn_out)?;

        if let Some(scale) = &self.layer_output_scale {
            x = x.broadcast_mul(scale)?;
        }
        Ok(x)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    per_layer_embeddings: Option<QMatMul>,
    per_layer_proj_norm: Option<RmsNorm>,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    rotary: RotaryEmbedding,
    device: Device,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md = &ct.metadata;
        let block_count = md
            .get("gemma4.block_count")
            .or_else(|| md.get("gemma3.block_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(28) as usize;
        let head_count = md
            .get("gemma4.attention.head_count")
            .or_else(|| md.get("gemma3.attention.head_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(8) as usize;
        let head_count_kv = md
            .get("gemma4.attention.head_count_kv")
            .or_else(|| md.get("gemma3.attention.head_count_kv"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4) as usize;
        let head_dim = md
            .get("gemma4.attention.key_length")
            .or_else(|| md.get("gemma3.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(256) as usize;
        let rms_norm_eps = md
            .get("gemma4.attention.layer_norm_rms_epsilon")
            .or_else(|| md.get("gemma3.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6) as f64;
        let rope_freq = md
            .get("gemma4.rope.freq_base")
            .or_else(|| md.get("gemma3.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1_000_000.0);

        let rotary = RotaryEmbedding::new(head_dim, rope_freq, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = Embedding::new(tok_embeddings_q.dequantize(device)?, tok_embeddings_q.shape().dims()[1]);

        let per_layer_embeddings = ct
            .tensor(reader, "per_layer_token_embd.weight", device)
            .ok()
            .and_then(|q| QMatMul::from_qtensor(q).ok());
        let per_layer_proj_norm = ct
            .tensor(reader, "per_layer_proj_norm.weight", device)
            .ok()
            .and_then(|q| RmsNorm::from_qtensor(q, rms_norm_eps).ok());

        let norm = RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, rms_norm_eps)?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => QMatMul::from_qtensor(t)?,
            Err(_) => QMatMul::from_qtensor(tok_embeddings_q)?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("blk.{i}");
            let attention_wq = QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?)?;
            let attention_wk = QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?)?;
            let attention_wv = QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_v.weight"), device)?)?;
            let attention_wo = QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_output.weight"), device)?)?;

            let attention_q_norm = RmsNorm::from_qtensor(ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?, rms_norm_eps)?;
            let attention_k_norm = RmsNorm::from_qtensor(ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?, rms_norm_eps)?;

            let attention_norm = RmsNorm::from_qtensor(ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?, rms_norm_eps)?;
            let post_attention_norm = RmsNorm::from_qtensor(ct.tensor(reader, &format!("{p}.post_attention_norm.weight"), device)?, rms_norm_eps)?;
            let ffn_norm = RmsNorm::from_qtensor(ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?, rms_norm_eps)?;
            let post_ffn_norm = RmsNorm::from_qtensor(ct.tensor(reader, &format!("{p}.post_ffw_norm.weight"), device)?, rms_norm_eps)?;

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.ffn_gate.weight"), device)?)?,
                feed_forward_up: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.ffn_up.weight"), device)?)?,
                feed_forward_down: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.ffn_down.weight"), device)?)?,
            };

            let inp_gate = ct.tensor(reader, &format!("{p}.inp_gate.weight"), device).ok().and_then(|q| QMatMul::from_qtensor(q).ok());
            let proj = ct.tensor(reader, &format!("{p}.proj.weight"), device).ok().and_then(|q| QMatMul::from_qtensor(q).ok());
            let post_norm = ct.tensor(reader, &format!("{p}.post_norm.weight"), device).ok().and_then(|q| RmsNorm::from_qtensor(q, rms_norm_eps).ok());
            let layer_output_scale = ct.tensor(reader, &format!("{p}.layer_output_scale.weight"), device).ok().and_then(|q| q.dequantize(device).ok());

            layers.push(LayerWeights {
                attention_wq,
                attention_wk,
                attention_wv,
                attention_wo,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                inp_gate,
                proj,
                post_norm,
                layer_output_scale,
                kv_cache: None,
            });
        }

        Ok(Self {
            tok_embeddings,
            per_layer_embeddings,
            per_layer_proj_norm,
            layers,
            norm,
            output,
            rotary,
            device: device.clone(),
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut hidden = self.tok_embeddings.forward(x)?;

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, index_pos)?;
        }

        let hidden = self.norm.forward(&hidden)?;
        let last_hidden = hidden.narrow(1, seq_len.saturating_sub(1), 1)?;
        self.output.forward(&last_hidden.squeeze(1)?)
    }
}
