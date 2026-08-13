use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_file(name: &str, records: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("qwen35moe-{name}-{nonce}.jsonl"));
    fs::write(&path, records).unwrap();
    path
}

fn record(step: usize, argmax: usize, margin: f64, values: Option<&str>) -> String {
    format!(
        "{{\"type\":\"logits\",\"step\":{step},\"argmax\":{argmax},\"margin\":{margin},\"finite\":3,\"values\":{}}}\n",
        values.unwrap_or("null")
    )
}

fn run(reference: &std::path::Path, candidate: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_qwen35moe_compare"))
        .args([
            "--gate",
            reference.to_str().unwrap(),
            candidate.to_str().unwrap(),
        ])
        .output()
        .unwrap()
}

#[test]
fn gate_accepts_low_margin_argmax_drift_with_close_logits() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..128 {
        let argmax = usize::from(step == 16);
        let margin = if step == 16 { 0.29 } else { 1.0 };
        let values = matches!(step, 16 | 45 | 50 | 92 | 111).then_some("[1.0,2.0,3.0]");
        reference.push_str(&record(step, argmax, margin, values));
        candidate.push_str(&record(step, 0, 1.0, values));
    }
    let reference = temp_file("reference-pass", &reference);
    let candidate = temp_file("candidate-pass", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn gate_rejects_high_margin_argmax_drift() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..128 {
        let values = matches!(step, 16 | 45 | 50 | 92 | 111).then_some("[1.0,2.0,3.0]");
        reference.push_str(&record(step, usize::from(step == 16), 0.31, values));
        candidate.push_str(&record(step, 0, 1.0, values));
    }
    let reference = temp_file("reference-fail", &reference);
    let candidate = temp_file("candidate-fail", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("argmax divergence at step 16"));
}

#[test]
fn gate_rejects_large_numerical_drift() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..128 {
        let full = matches!(step, 16 | 45 | 50 | 92 | 111);
        reference.push_str(&record(step, 0, 1.0, full.then_some("[1.0,2.0,3.0]")));
        candidate.push_str(&record(step, 0, 1.0, full.then_some("[10.0,0.0,0.0]")));
    }
    let reference = temp_file("reference-drift", &reference);
    let candidate = temp_file("candidate-drift", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("numerical gate failed"));
}

#[test]
fn gate_rejects_missing_required_step() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..127 {
        let values = matches!(step, 16 | 45 | 50 | 92 | 111).then_some("[1.0,2.0,3.0]");
        reference.push_str(&record(step, 0, 1.0, values));
        candidate.push_str(&record(step, 0, 1.0, values));
    }
    let reference = temp_file("reference-short", &reference);
    let candidate = temp_file("candidate-short", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("gate needs exactly 128 logits records")
    );
}
