//! Model profile and fail-fast GGUF validation (Phase 1).
//!
//! Produces an immutable `ModelProfile` from a `gguf_file::Content` before
//! heavy tensor loading. Validates architecture, block layout, expert config,
//! and tensor contracts for both dense `qwen35` and MoE `qwen35moe`.
//!
//! Errors are aggregated where safe so startup reports all incompatible
//! properties at once rather than failing one-at-a-time.

use candle_core::quantized::gguf_file::{Content, TensorInfo, Value};
use candle_core::quantized::GgmlDType;
use candle_core::Result;
use std::collections::{BTreeMap, HashSet};

/// Clone a TensorInfo manually since gguf_file::TensorInfo does not impl Clone.
fn clone_tensor_info(ti: &TensorInfo) -> TensorInfo {
    TensorInfo {
        ggml_dtype: ti.ggml_dtype,
        shape: ti.shape.clone(),
        offset: ti.offset,
    }
}

/// Runtime architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Dense Qwen3.5/Qwen3.6 hybrid DeltaNet/attention trunk (`general.architecture=qwen35`).
    DenseQwen35,
    /// MoE Qwen3.6 35B-A3B (`general.architecture=qwen35moe`).
    Qwen35Moe,
}

impl Architecture {
    /// Parse from the `general.architecture` GGUF metadata value.
    pub fn from_metadata(md: &BTreeMap<String, Value>) -> Result<Self> {
        let arch = md
            .get("general.architecture")
            .ok_or_else(|| candle_core::Error::Msg("missing general.architecture".into()))?;
        let s = match arch {
            Value::String(s) => s.as_str(),
            _ => return Err(candle_core::Error::Msg("general.architecture not a string".into())),
        };
        match s {
            "qwen35" => Ok(Self::DenseQwen35),
            "qwen35moe" => Ok(Self::Qwen35Moe),
            other => Err(candle_core::Error::Msg(format!(
                "unsupported architecture: '{other}' — supported: qwen35, qwen35moe"
            ))),
        }
    }

    pub fn metadata_prefix(&self) -> &'static str {
        match self {
            Self::DenseQwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
        }
    }
}

/// Stable fingerprint derived from architecture metadata + tensor manifest.
/// Does NOT include the local file path — reproducible across machines.
#[derive(Debug, Clone)]
pub struct ModelFingerprint {
    pub hash: String,
}

impl ModelFingerprint {
    fn compute(
        arch: Architecture,
        metadata: &BTreeMap<String, Value>,
        tensor_manifest: &[(String, GgmlDType, Vec<usize>)],
        file_size: u64,
    ) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        arch.metadata_prefix().hash(&mut h);
        // Architecture-relevant metadata keys (sorted by BTreeMap iteration).
        let prefix = arch.metadata_prefix();
        let arch_keys = [
            "general.architecture",
            "general.file_type",
            "general.name",
            "tokenizer.ggml.model",
        ];
        for key in arch_keys {
            if let Some(v) = metadata.get(key) {
                key.hash(&mut h);
                format!("{v:?}").hash(&mut h);
            }
        }
        // Prefix-specific metadata keys.
        let prefix_keys = [
            format!("{prefix}.block_count"),
            format!("{prefix}.context_length"),
            format!("{prefix}.embedding_length"),
            format!("{prefix}.feed_forward_length"),
            format!("{prefix}.attention.head_count"),
            format!("{prefix}.attention.head_count_kv"),
            format!("{prefix}.attention.key_length"),
            format!("{prefix}.attention.value_length"),
        ];
        for key in prefix_keys {
            if let Some(v) = metadata.get(&key) {
                key.hash(&mut h);
                format!("{v:?}").hash(&mut h);
            }
        }
        // MoE-specific keys.
        if matches!(arch, Architecture::Qwen35Moe) {
            let moe_keys = [
                "qwen35moe.expert_count",
                "qwen35moe.expert_used_count",
                "qwen35moe.shared_expert_count",
            ];
            for key in moe_keys {
                if let Some(v) = metadata.get(key) {
                    key.hash(&mut h);
                    format!("{v:?}").hash(&mut h);
                }
            }
        }
        // Tensor manifest.
        for (name, dtype, shape) in tensor_manifest {
            name.hash(&mut h);
            format!("{dtype:?}").hash(&mut h);
            shape.hash(&mut h);
        }
        file_size.hash(&mut h);
        Self {
            hash: format!("qwen35-profile-{:016x}", h.finish()),
        }
    }
}

/// Immutable model profile produced before heavy tensor loading.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub architecture: Architecture,
    pub block_count: usize,
    pub hidden_size: usize,
    pub context_length: usize,
    pub full_attention_interval: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f64,
    /// MoE-specific fields (None for dense).
    pub routed_experts: Option<usize>,
    pub experts_per_token: Option<usize>,
    pub routed_intermediate: Option<usize>,
    pub shared_intermediate: Option<usize>,
    pub shared_expert_count: Option<usize>,
    pub router_norm_topk: Option<bool>,
    /// Distinct quant dtypes present in the GGUF tensor infos.
    pub quant_set: HashSet<GgmlDType>,
    pub fingerprint: ModelFingerprint,
}

/// Aggregated validation errors — all problems reported at once.
#[derive(Debug, Default)]
pub struct ValidationErrors {
    pub errors: Vec<String>,
}

impl ValidationErrors {
    fn push(&mut self, msg: String) {
        self.errors.push(msg);
    }
    fn push_missing(&mut self, key: &str) {
        self.errors.push(format!("missing metadata key: {key}"));
    }
    fn push_tensor_missing(&mut self, name: &str) {
        self.errors.push(format!("missing required tensor: {name}"));
    }
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
    pub fn into_result(self) -> Result<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            let joined = self.errors.join("; ");
            Err(candle_core::Error::Msg(format!(
                "GGUF validation failed ({} error(s)): {joined}",
                self.errors.len()
            )))
        }
    }
}

/// Helper to read a u32 metadata value.
fn md_u32(md: &BTreeMap<String, Value>, key: &str, errs: &mut ValidationErrors) -> Option<usize> {
    match md.get(key) {
        Some(v) => match v.to_u32() {
            Ok(n) => Some(n as usize),
            Err(_) => {
                errs.push(format!("metadata key {key} is not a u32"));
                None
            }
        },
        None => {
            errs.push_missing(key);
            None
        }
    }
}

/// Helper to read an f32 metadata value (optional with default).
fn md_f32_opt(md: &BTreeMap<String, Value>, key: &str, default: f32) -> f32 {
    md.get(key)
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(default)
}

/// Helper to read an f32 metadata value (required).
fn md_f32_req(md: &BTreeMap<String, Value>, key: &str, errs: &mut ValidationErrors) -> Option<f32> {
    match md.get(key) {
        Some(v) => match v.to_f32() {
            Ok(n) => Some(n),
            Err(_) => {
                errs.push(format!("metadata key {key} is not an f32"));
                None
            }
        },
        None => {
            errs.push_missing(key);
            None
        }
    }
}

/// Helper to read a bool metadata value (optional).
fn md_bool_opt(md: &BTreeMap<String, Value>, key: &str) -> Option<bool> {
    md.get(key).and_then(|v| v.to_bool().ok())
}

/// Check that a required tensor exists in the GGUF tensor_infos.
fn require_tensor(
    tensor_infos: &BTreeMap<String, TensorInfo>,
    name: &str,
    errs: &mut ValidationErrors,
) {
    if !tensor_infos.contains_key(name) {
        errs.push_tensor_missing(name);
    }
}

/// Check tensors for a single block at the given layer index.
/// GGUF tensor names have NO architecture prefix (`blk.N....`); the prefix is
/// only for metadata keys. Hybrid trunk: DeltaNet blocks (linear recurrence)
/// interleave with full-attention blocks every `full_attention_interval`.
fn require_block_tensors(
    tensor_infos: &BTreeMap<String, TensorInfo>,
    arch: Architecture,
    layer: usize,
    full_attention_interval: usize,
    errs: &mut ValidationErrors,
) {
    let prefix = format!("blk.{layer}");
    // Common norms for both block types.
    require_tensor(tensor_infos, &format!("{prefix}.attn_norm.weight"), errs);
    require_tensor(
        tensor_infos,
        &format!("{prefix}.post_attention_norm.weight"),
        errs,
    );
    // Sequence mixer depends on hybrid layout.
    let interval = full_attention_interval.max(1);
    if (layer + 1) % interval == 0 {
        // Full attention block.
        for t in [
            format!("{prefix}.attn_q.weight"),
            format!("{prefix}.attn_k.weight"),
            format!("{prefix}.attn_v.weight"),
            format!("{prefix}.attn_output.weight"),
        ] {
            require_tensor(tensor_infos, &t, errs);
        }
    } else {
        // DeltaNet block.
        for t in [
            format!("{prefix}.attn_qkv.weight"),
            format!("{prefix}.attn_gate.weight"),
            format!("{prefix}.ssm_beta.weight"),
            format!("{prefix}.ssm_alpha.weight"),
            format!("{prefix}.ssm_out.weight"),
            format!("{prefix}.ssm_dt.bias"),
            format!("{prefix}.ssm_a"),
            format!("{prefix}.ssm_conv1d.weight"),
            format!("{prefix}.ssm_norm.weight"),
        ] {
            require_tensor(tensor_infos, &t, errs);
        }
    }
    // Feed-forward tensors depend on architecture.
    match arch {
        Architecture::DenseQwen35 => {
            let ffn = [
                format!("{prefix}.ffn_gate.weight"),
                format!("{prefix}.ffn_up.weight"),
                format!("{prefix}.ffn_down.weight"),
            ];
            for t in &ffn {
                require_tensor(tensor_infos, t, errs);
            }
        }
        Architecture::Qwen35Moe => {
            // MoE: router + shared expert + packed routed experts.
            let moe_tensors = [
                format!("{prefix}.ffn_gate_inp.weight"),  // router
                format!("{prefix}.ffn_gate_exps.weight"), // routed gate
                format!("{prefix}.ffn_up_exps.weight"),   // routed up
                format!("{prefix}.ffn_down_exps.weight"), // routed down
                format!("{prefix}.ffn_gate_shexp.weight"), // shared gate
                format!("{prefix}.ffn_up_shexp.weight"),  // shared up
                format!("{prefix}.ffn_down_shexp.weight"), // shared down
            ];
            for t in &moe_tensors {
                require_tensor(tensor_infos, t, errs);
            }
        }
    }
}

impl ModelProfile {
    /// Read and validate a GGUF `Content` before heavy tensor loading.
    ///
    /// Returns a `ModelProfile` on success, or an aggregated error listing
    /// all validation failures.
    pub fn read_and_validate(ct: &Content, file_size: u64) -> Result<Self> {
        let mut errs = ValidationErrors::default();

        // Sorted metadata and tensor_infos for deterministic iteration.
        let metadata: BTreeMap<String, Value> = ct.metadata.clone().into_iter().collect();
        let tensor_infos: BTreeMap<String, TensorInfo> = ct
            .tensor_infos
            .iter()
            .map(|(k, v)| (k.clone(), clone_tensor_info(v)))
            .collect();

        // 1. Architecture.
        let architecture = match Architecture::from_metadata(&metadata) {
            Ok(a) => a,
            Err(e) => {
                errs.push(format!("{e}"));
                // Can't continue without architecture.
                return errs.into_result().map(|_| unreachable!());
            }
        };
        let prefix = architecture.metadata_prefix();

        // 2. Required metadata for both architectures.
        let block_count = md_u32(&metadata, &format!("{prefix}.block_count"), &mut errs)
            .unwrap_or(0);
        let hidden_size = md_u32(&metadata, &format!("{prefix}.embedding_length"), &mut errs)
            .unwrap_or(0);
        let context_length = md_u32(&metadata, &format!("{prefix}.context_length"), &mut errs)
            .unwrap_or(0);
        let feed_forward_length =
            md_u32(&metadata, &format!("{prefix}.feed_forward_length"), &mut errs).unwrap_or(0);
        let attention_head_count =
            md_u32(&metadata, &format!("{prefix}.attention.head_count"), &mut errs).unwrap_or(0);
        let attention_head_count_kv =
            md_u32(&metadata, &format!("{prefix}.attention.head_count_kv"), &mut errs)
                .unwrap_or(0);
        let attention_key_length =
            md_u32(&metadata, &format!("{prefix}.attention.key_length"), &mut errs).unwrap_or(0);
        let attention_value_length =
            md_u32(&metadata, &format!("{prefix}.attention.value_length"), &mut errs)
                .unwrap_or(0);
        let rms_norm_eps = md_f32_req(
            &metadata,
            &format!("{prefix}.attention.layer_norm_rms_epsilon"),
            &mut errs,
        )
        .map(|v| v as f64)
        .unwrap_or(0.0);
        let rope_freq_base = md_f32_opt(&metadata, &format!("{prefix}.rope.freq_base"), 10000.0);
        let full_attention_interval = md_u32(
            &metadata,
            &format!("{prefix}.full_attention_interval"),
            &mut errs,
        )
        .unwrap_or(4);

        // 3. MoE-specific metadata.
        let (routed_experts, experts_per_token, routed_intermediate, shared_intermediate, shared_expert_count, router_norm_topk) =
            match architecture {
                Architecture::Qwen35Moe => {
                    let re = md_u32(&metadata, "qwen35moe.expert_count", &mut errs);
                    let ept = md_u32(&metadata, "qwen35moe.expert_used_count", &mut errs);
                    let ri = md_u32(
                        &metadata,
                        "qwen35moe.feed_forward_length.experts",
                        &mut errs,
                    )
                    .or_else(|| {
                        // Some GGUFs use a different key; try fallback.
                        md_u32(&metadata, "qwen35moe.intermediate_size_experts", &mut errs)
                    });
                    let si = md_u32(
                        &metadata,
                        "qwen35moe.feed_forward_length.shared_expert",
                        &mut errs,
                    )
                    .or_else(|| {
                        md_u32(
                            &metadata,
                            "qwen35moe.intermediate_size_shared_expert",
                            &mut errs,
                        )
                    });
                    let sec =
                        md_u32(&metadata, "qwen35moe.shared_expert_count", &mut errs);
                    let rnt = md_bool_opt(&metadata, "qwen35moe.router_norm_topk");
                    (re, ept, ri, si, sec, rnt)
                }
                Architecture::DenseQwen35 => (None, None, None, None, None, None),
            };

        // 4. Validate expert_used_count <= expert_count (if both present).
        if let (Some(re), Some(ept)) = (routed_experts, experts_per_token) {
            if ept > re {
                errs.push(format!(
                    "expert_used_count ({ept}) > expert_count ({re})"
                ));
            }
            // Target model: top-8.
            if ept != 8 {
                errs.push(format!(
                    "expert_used_count={ept}, expected 8 for Qwen3.6 35B-A3B"
                ));
            }
        }

        // 5. Validate block_count (target: 40 for Qwen3.6 35B-A3B MoE).
        if matches!(architecture, Architecture::Qwen35Moe) && block_count != 40 {
            errs.push(format!(
                "block_count={block_count}, expected 40 for Qwen3.6 35B-A3B"
            ));
        }

        // 6. Required global tensors.
        require_tensor(&tensor_infos, "token_embd.weight", &mut errs);
        require_tensor(&tensor_infos, "output_norm.weight", &mut errs);
        // output.weight may be tied — not strictly required.

        // 7. Per-block tensor validation (sample first + last block; full check
        //    for small models, sampled for large to keep validation fast).
        if block_count > 0 {
            let interval = full_attention_interval.max(1) as usize;
            // Always check block 0 and last block.
            for layer in [0usize, block_count - 1] {
                require_block_tensors(&tensor_infos, architecture, layer, interval, &mut errs);
            }
            // For MoE with <= 48 blocks, check all (40 is the target).
            if matches!(architecture, Architecture::Qwen35Moe) && block_count <= 48 {
                for layer in 0..block_count {
                    require_block_tensors(&tensor_infos, architecture, layer, interval, &mut errs);
                }
            }
        }

        // 8. Collect quant set from tensor infos.
        let quant_set: HashSet<GgmlDType> =
            tensor_infos.values().map(|ti| ti.ggml_dtype).collect();

        // 9. Build tensor manifest for fingerprint (name, dtype, shape).
        let tensor_manifest: Vec<(String, GgmlDType, Vec<usize>)> = tensor_infos
            .iter()
            .map(|(name, ti)| (name.clone(), ti.ggml_dtype, ti.shape.dims().to_vec()))
            .collect();

        let fingerprint = ModelFingerprint::compute(
            architecture,
            &metadata,
            &tensor_manifest,
            file_size,
        );

        // 10. Return aggregated errors or success.
        errs.into_result()?;

        Ok(Self {
            architecture,
            block_count,
            hidden_size,
            context_length,
            full_attention_interval,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            attention_key_length,
            attention_value_length,
            rope_freq_base,
            rms_norm_eps,
            routed_experts,
            experts_per_token,
            routed_intermediate,
            shared_intermediate,
            shared_expert_count,
            router_norm_topk,
            quant_set,
            fingerprint,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_content() -> Content {
        Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV2,
            metadata: std::collections::HashMap::new(),
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn test_missing_architecture_fails() {
        let ct = empty_content();
        let result = ModelProfile::read_and_validate(&ct, 0);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("general.architecture"));
    }

    #[test]
    fn test_unsupported_architecture_fails() {
        let mut ct = empty_content();
        ct.metadata.insert(
            "general.architecture".into(),
            Value::String("llama".into()),
        );
        let result = ModelProfile::read_and_validate(&ct, 0);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unsupported architecture"));
    }

    #[test]
    fn test_dense_missing_metadata_aggregates_errors() {
        let mut ct = empty_content();
        ct.metadata.insert(
            "general.architecture".into(),
            Value::String("qwen35".into()),
        );
        // Deliberately missing all qwen35.* keys.
        let result = ModelProfile::read_and_validate(&ct, 0);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        // Should report multiple missing keys at once.
        assert!(msg.contains("qwen35.block_count"));
        assert!(msg.contains("qwen35.embedding_length"));
        assert!(msg.contains("multiple") || msg.contains("error(s)"));
    }
}