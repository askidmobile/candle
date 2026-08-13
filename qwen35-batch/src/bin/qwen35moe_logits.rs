use anyhow::{bail, Context, Result};
use candle_core::Device;
use qwen35_batch::model::{DecodeBatch, DecodeItem, PrefillChunk};
use qwen35_batch::real::{tokenizer, Qwen35BatchAdapter};
use qwen35_batch::BatchModel;
use serde_json::json;
use std::cmp::Ordering;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best = index as u32;
            best_value = value;
        }
    }
    best
}

fn record(step: usize, logits: &[f32], include_values: bool) -> serde_json::Value {
    let mut indices: Vec<usize> = (0..logits.len())
        .filter(|&index| logits[index].is_finite())
        .collect();
    let top_len = 10.min(indices.len());
    if top_len > 0 {
        indices.select_nth_unstable_by(top_len - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        indices.truncate(top_len);
    }
    indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut finite = 0usize;
    let mut checksum = 0xcbf29ce484222325u64;
    for &value in logits {
        checksum ^= value.to_bits() as u64;
        checksum = checksum.wrapping_mul(0x100000001b3);
        if value.is_finite() {
            finite += 1;
            let value = value as f64;
            sum += value;
            sum_sq += value * value;
        }
    }
    let top: Vec<_> = indices
        .iter()
        .map(|&index| json!({"id": index, "logit": logits[index]}))
        .collect();
    let margin = if indices.len() >= 2 {
        logits[indices[0]] - logits[indices[1]]
    } else {
        f32::NAN
    };
    let mut record = json!({
        "type": "logits",
        "step": step,
        "argmax": argmax(logits),
        "margin": margin,
        "finite": finite,
        "sum": sum,
        "sum_sq": sum_sq,
        "checksum": format!("{checksum:016x}"),
        "top": top,
    });
    if include_values {
        record["values"] = json!(logits);
    }
    record
}

fn read_forced_tokens(path: &Path) -> Result<Vec<u32>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open forced-token JSONL: {}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if !line.starts_with('{') {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(&line)?;
        if record["type"] == "tokens" {
            return record["ids"]
                .as_array()
                .context("forced-token record missing ids")?
                .iter()
                .map(|id| {
                    id.as_u64()
                        .map(|id| id as u32)
                        .context("forced token is not u32")
                })
                .collect();
        }
    }
    bail!(
        "forced-token JSONL has no tokens record: {}",
        path.display()
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model = PathBuf::from(
        args.next()
            .context("usage: qwen35moe_logits <model.gguf> [steps]")?,
    );
    let steps = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()
        .context("steps must be an integer")?
        .unwrap_or(8);
    let fixture = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "capital".into());
    let forced_path = args.next().map(PathBuf::from);
    if args.next().is_some() || steps == 0 || !matches!(fixture.as_str(), "capital" | "rust") {
        bail!("usage: qwen35moe_logits <model.gguf> [steps] [capital|rust] [forced.jsonl]")
    }
    let forced_tokens = forced_path.as_deref().map(read_forced_tokens).transpose()?;
    if let Some(tokens) = &forced_tokens {
        if tokens.len() < steps {
            bail!(
                "forced-token file has {} tokens, need {steps}",
                tokens.len()
            )
        }
    }

    let device = Device::new_cuda(0).context("open CUDA device 0")?;
    let tokenizer = tokenizer::load_from_gguf_path(&model)?;
    let messages = match fixture.as_str() {
        "capital" => vec![
            tokenizer::ChatMsg {
                role: "system",
                content: "You are a helpful assistant. Answer concisely.",
            },
            tokenizer::ChatMsg {
                role: "user",
                content: "What is the capital of France? Answer with one word.",
            },
        ],
        "rust" => vec![tokenizer::ChatMsg {
            role: "user",
            content: "Write a numbered list of twenty concise Rust memory-safety facts. Do not stop before item twenty.",
        }],
        _ => unreachable!(),
    };
    let prompt = tokenizer::build_chatml_text(&messages);
    let prompt_tokens = tokenizer::encode_no_think(&tokenizer, &prompt)?;
    let backend = std::env::var("QWEN36_MOE_BACKEND").unwrap_or_else(|_| "reference".into());
    println!(
        "{}",
        json!({
            "type": "run",
            "schema": "qwen35moe-logits-v1",
            "backend": backend,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "fixture": fixture,
            "forced_tokens": forced_path,
            "steps": steps,
        })
    );

    let mut adapter = Qwen35BatchAdapter::load(&model, device, 1)?;
    let mut logits = adapter.prefill_chunk(&PrefillChunk {
        slot_idx: 0,
        reset_first: true,
        tokens: prompt_tokens.clone(),
        start_pos: 0,
    })?;

    let full_step = std::env::var("QWEN36_LOGITS_FULL_STEP")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("QWEN36_LOGITS_FULL_STEP must be an integer")?;
    let include_all_values = std::env::var_os("QWEN36_LOGITS_FULL").is_some();
    let mut predicted = Vec::with_capacity(steps);
    let mut fed = Vec::with_capacity(steps);
    for step in 0..steps {
        let prediction = argmax(&logits);
        println!(
            "{}",
            record(step, &logits, include_all_values || full_step == Some(step))
        );
        let token = forced_tokens
            .as_ref()
            .map(|tokens| tokens[step])
            .unwrap_or(prediction);
        predicted.push(prediction);
        fed.push(token);
        if token == adapter.eos() || step + 1 == steps {
            break;
        }
        logits = adapter
            .decode_batch(&DecodeBatch {
                items: vec![DecodeItem {
                    slot_idx: 0,
                    token,
                    pos: prompt_tokens.len() + step,
                }],
            })?
            .pop()
            .context("decode returned no logits")?;
    }
    println!(
        "{}",
        json!({
            "type": "tokens",
            "ids": predicted,
            "fed_ids": fed,
            "text": tokenizer::decode_text(&tokenizer, &predicted)?,
        })
    );
    Ok(())
}
