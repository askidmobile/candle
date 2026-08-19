use qwen35_batch::real::tokenizer;
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("gguf path");
    let tok = tokenizer::load_from_gguf_path(std::path::Path::new(&path))?;
    let text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("ids: {:?}", enc.get_ids());
    println!("expect contains 248045 (im_start), 248068 (think open)");
    Ok(())
}
