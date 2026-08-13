use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn read_records(path: &str) -> Result<(BTreeMap<usize, Value>, Value)> {
    let file = File::open(path).with_context(|| format!("open {path}"))?;
    let mut logits = BTreeMap::new();
    let mut tokens = Value::Null;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if !line.starts_with('{') {
            continue;
        }
        let record: Value = serde_json::from_str(&line)?;
        match record["type"].as_str() {
            Some("logits") => {
                logits.insert(
                    record["step"].as_u64().context("missing logits step")? as usize,
                    record,
                );
            }
            Some("tokens") => tokens = record,
            _ => {}
        }
    }
    Ok((logits, tokens))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let reference_path = args
        .next()
        .context("usage: qwen35moe_compare <reference.jsonl> <candidate.jsonl>")?;
    let candidate_path = args
        .next()
        .context("usage: qwen35moe_compare <reference.jsonl> <candidate.jsonl>")?;
    if args.next().is_some() {
        bail!("usage: qwen35moe_compare <reference.jsonl> <candidate.jsonl>")
    }
    let (reference, reference_tokens) = read_records(&reference_path)?;
    let (candidate, candidate_tokens) = read_records(&candidate_path)?;
    let mut first_argmax_divergence = None;
    let mut compared = 0usize;

    for (step, r) in &reference {
        let Some(c) = candidate.get(step) else {
            continue;
        };
        compared += 1;
        let metrics = match (r["values"].as_array(), c["values"].as_array()) {
            (Some(rv), Some(cv)) => {
                if rv.len() != cv.len() {
                    bail!("step {step}: value length {} != {}", rv.len(), cv.len())
                }
                let mut dot = 0.0f64;
                let mut ref_sq = 0.0f64;
                let mut candidate_sq = 0.0f64;
                let mut diff_sq = 0.0f64;
                let mut max_abs = 0.0f64;
                for (r, c) in rv.iter().zip(cv) {
                    let (Some(r), Some(c)) = (r.as_f64(), c.as_f64()) else {
                        continue;
                    };
                    dot += r * c;
                    ref_sq += r * r;
                    candidate_sq += c * c;
                    let diff = c - r;
                    diff_sq += diff * diff;
                    max_abs = max_abs.max(diff.abs());
                }
                json!({
                    "cosine": dot / (ref_sq.sqrt() * candidate_sq.sqrt()),
                    "nrmse": (diff_sq / ref_sq).sqrt(),
                    "max_abs": max_abs,
                })
            }
            _ => Value::Null,
        };
        let argmax_equal = r["argmax"] == c["argmax"];
        if !argmax_equal && first_argmax_divergence.is_none() {
            first_argmax_divergence = Some(*step);
        }
        println!(
            "{}",
            json!({
                "type": "comparison",
                "step": step,
                "argmax_equal": argmax_equal,
                "reference_argmax": r["argmax"],
                "candidate_argmax": c["argmax"],
                "reference_margin": r["margin"],
                "candidate_margin": c["margin"],
                "metrics": metrics,
            })
        );
    }
    println!(
        "{}",
        json!({
            "type": "summary",
            "steps_compared": compared,
            "first_argmax_divergence": first_argmax_divergence,
            "tokens_equal": reference_tokens["ids"] == candidate_tokens["ids"],
            "reference_tokens": reference_tokens["ids"],
            "candidate_tokens": candidate_tokens["ids"],
        })
    );
    Ok(())
}
