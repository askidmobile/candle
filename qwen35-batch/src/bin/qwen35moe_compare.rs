use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

const GATE_STEPS: usize = 128;
const GATE_FULL_STEPS: [usize; 5] = [16, 45, 50, 92, 111];
const GATE_MAX_ARGMAX_DIVERGENCES: usize = 5;
const GATE_MIN_COSINE: f64 = 0.997;
const GATE_MAX_NRMSE: f64 = 0.07;
const GATE_MAX_ABS: f64 = 1.3;
const GATE_MAX_REFERENCE_MARGIN_FOR_ARGMAX_DRIFT: f64 = 0.30;

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
    let first = args
        .next()
        .context("usage: qwen35moe_compare [--gate] <reference.jsonl> <candidate.jsonl>")?;
    let gate = first == "--gate";
    let reference_path = if gate {
        args.next()
            .context("usage: qwen35moe_compare [--gate] <reference.jsonl> <candidate.jsonl>")?
    } else {
        first
    };
    let candidate_path = args
        .next()
        .context("usage: qwen35moe_compare [--gate] <reference.jsonl> <candidate.jsonl>")?;
    if args.next().is_some() {
        bail!("usage: qwen35moe_compare [--gate] <reference.jsonl> <candidate.jsonl>")
    }
    let (reference, reference_tokens) = read_records(&reference_path)?;
    let (candidate, candidate_tokens) = read_records(&candidate_path)?;
    if gate {
        if reference.len() != GATE_STEPS || candidate.len() != GATE_STEPS {
            bail!(
                "gate needs exactly {GATE_STEPS} logits records, got reference={} candidate={}",
                reference.len(),
                candidate.len()
            )
        }
        for step in 0..GATE_STEPS {
            if !reference.contains_key(&step) || !candidate.contains_key(&step) {
                bail!("gate missing logits step {step}")
            }
        }
    }
    let mut first_argmax_divergence = None;
    let mut first_gate_failure = None;
    let mut compared = 0usize;
    let mut full_vectors_compared = Vec::new();
    let mut argmax_divergences = 0usize;

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
                if gate {
                    let expected = r["finite"]
                        .as_u64()
                        .context("full-logit record missing finite count")?
                        as usize;
                    if rv.len() != expected || c["finite"] != r["finite"] {
                        bail!(
                            "step {step}: full-logit length/count mismatch: values={}, reference finite={}, candidate finite={}",
                            rv.len(),
                            r["finite"],
                            c["finite"]
                        )
                    }
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
                let cosine = dot / (ref_sq.sqrt() * candidate_sq.sqrt());
                let nrmse = (diff_sq / ref_sq).sqrt();
                full_vectors_compared.push(*step);
                if gate
                    && first_gate_failure.is_none()
                    && (!cosine.is_finite()
                        || !nrmse.is_finite()
                        || cosine < GATE_MIN_COSINE
                        || nrmse > GATE_MAX_NRMSE
                        || max_abs > GATE_MAX_ABS)
                {
                    first_gate_failure = Some(format!(
                        "numerical gate failed at step {step}: cosine={cosine:.6}, nrmse={nrmse:.6}, max_abs={max_abs:.6}"
                    ));
                }
                json!({
                    "cosine": cosine,
                    "nrmse": nrmse,
                    "max_abs": max_abs,
                })
            }
            _ => Value::Null,
        };
        let argmax_equal = r["argmax"] == c["argmax"];
        if !argmax_equal {
            argmax_divergences += 1;
            if first_argmax_divergence.is_none() {
                first_argmax_divergence = Some(*step);
            }
            let reference_margin = r["margin"]
                .as_f64()
                .context("argmax divergence missing reference margin")?;
            if gate
                && first_gate_failure.is_none()
                && reference_margin > GATE_MAX_REFERENCE_MARGIN_FOR_ARGMAX_DRIFT
            {
                first_gate_failure = Some(format!(
                    "argmax divergence at step {step} has reference margin {reference_margin:.6} > {GATE_MAX_REFERENCE_MARGIN_FOR_ARGMAX_DRIFT:.2}"
                ));
            }
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
    if gate && first_gate_failure.is_none() && argmax_divergences > GATE_MAX_ARGMAX_DIVERGENCES {
        first_gate_failure = Some(format!(
            "gate allows at most {GATE_MAX_ARGMAX_DIVERGENCES} low-margin argmax divergences, got {argmax_divergences}"
        ));
    }
    if gate && first_gate_failure.is_none() {
        let missing: Vec<_> = GATE_FULL_STEPS
            .iter()
            .filter(|step| !full_vectors_compared.contains(step))
            .collect();
        if !missing.is_empty() {
            first_gate_failure = Some(format!(
                "gate missing full-logit vectors at steps {missing:?}"
            ));
        }
    }
    println!(
        "{}",
        json!({
            "type": "summary",
            "steps_compared": compared,
            "full_vector_steps": full_vectors_compared,
            "argmax_divergences": argmax_divergences,
            "first_argmax_divergence": first_argmax_divergence,
            "gate": gate,
            "gate_passed": gate.then(|| first_gate_failure.is_none()),
            "tokens_equal": reference_tokens["ids"] == candidate_tokens["ids"],
            "reference_tokens": reference_tokens["ids"],
            "candidate_tokens": candidate_tokens["ids"],
        })
    );
    if let Some(failure) = first_gate_failure {
        bail!(failure)
    }
    Ok(())
}
