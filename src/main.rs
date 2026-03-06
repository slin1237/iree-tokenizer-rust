// CLI binary — print macros, process::exit, and unwrap are acceptable here.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::disallowed_methods
)]

use std::{
    io::{self, BufRead, Write},
    time::Instant,
};

use clap::{Parser, Subcommand};
use iree_tokenizer::Tokenizer;

#[derive(Parser)]
#[command(
    name = "iree-tokenizer",
    about = "Streaming tokenizer CLI backed by IREE"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode text to token IDs.
    Encode {
        /// Path to tokenizer.json or .tiktoken file.
        #[arg(short = 't', long = "tokenizer")]
        tokenizer_path: Option<String>,
        /// Tokenizer JSON string (alternative to -t).
        #[arg(long = "tokenizer-json")]
        tokenizer_json: Option<String>,
        /// Tiktoken encoding name (for .tiktoken files).
        #[arg(long)]
        encoding: Option<String>,
        /// Input mode: line, paragraph, or whole.
        #[arg(long, default_value = "line")]
        input_mode: String,
        /// Include byte offsets in output.
        #[arg(long)]
        rich: bool,
        /// Add special tokens (BOS/EOS).
        #[arg(long)]
        add_special_tokens: bool,
        /// Omit timing/size fields.
        #[arg(long)]
        compact: bool,
    },
    /// Decode token IDs to text.
    Decode {
        /// Path to tokenizer.json or .tiktoken file.
        #[arg(short = 't', long = "tokenizer")]
        tokenizer_path: Option<String>,
        /// Tokenizer JSON string (alternative to -t).
        #[arg(long = "tokenizer-json")]
        tokenizer_json: Option<String>,
        /// Tiktoken encoding name (for .tiktoken files).
        #[arg(long)]
        encoding: Option<String>,
        /// Skip special tokens in output.
        #[arg(long)]
        skip_special_tokens: bool,
        /// Omit timing fields.
        #[arg(long)]
        compact: bool,
    },
    /// Print tokenizer metadata.
    Info {
        /// Path to tokenizer.json or .tiktoken file.
        #[arg(short = 't', long = "tokenizer")]
        tokenizer_path: Option<String>,
        /// Tokenizer JSON string (alternative to -t).
        #[arg(long = "tokenizer-json")]
        tokenizer_json: Option<String>,
        /// Tiktoken encoding name (for .tiktoken files).
        #[arg(long)]
        encoding: Option<String>,
    },
}

fn load_tokenizer(
    path: &Option<String>,
    json_str: &Option<String>,
    encoding: &Option<String>,
) -> Tokenizer {
    if let Some(path) = path {
        if path.ends_with(".tiktoken") {
            let enc = encoding.as_deref().unwrap_or_else(|| {
                eprintln!("Error: --encoding required for .tiktoken files");
                std::process::exit(1);
            });
            return Tokenizer::from_tiktoken_file(path, enc).unwrap_or_else(|e| {
                eprintln!("Error loading tiktoken file: {e}");
                std::process::exit(1);
            });
        }
        return Tokenizer::from_file(path).unwrap_or_else(|e| {
            eprintln!("Error loading tokenizer: {e}");
            std::process::exit(1);
        });
    }
    if let Some(json) = json_str {
        return Tokenizer::from_str(json).unwrap_or_else(|e| {
            eprintln!("Error loading tokenizer from JSON: {e}");
            std::process::exit(1);
        });
    }
    eprintln!("Error: --tokenizer or --tokenizer-json required");
    std::process::exit(1);
}

/// Parse encode input: accepts JSON with "text" field or plain text.
fn parse_encode_input(line: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
        if let Some(text) = val.get("text").and_then(|v| v.as_str()) {
            return text.to_string();
        }
    }
    line.to_string()
}

/// Parse decode input: accepts JSON with "ids" field or JSON array.
fn parse_decode_input(line: &str) -> Vec<i32> {
    let val: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| {
        eprintln!("Error parsing JSON: {e}");
        std::process::exit(1);
    });
    if let Some(obj) = val.as_object() {
        if let Some(ids) = obj.get("ids") {
            return ids
                .as_array()
                .unwrap_or_else(|| {
                    eprintln!("Error: 'ids' must be an array");
                    std::process::exit(1);
                })
                .iter()
                .map(|v| v.as_i64().unwrap_or(0) as i32)
                .collect();
        }
    }
    if let Some(arr) = val.as_array() {
        return arr.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect();
    }
    eprintln!("Expected JSON object with 'ids' or JSON array");
    std::process::exit(1);
}

fn cmd_encode(
    tok: &Tokenizer,
    input_mode: &str,
    rich: bool,
    add_special_tokens: bool,
    compact: bool,
) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut seq = 0u64;
    match input_mode {
        "line" => {
            for line in stdin.lock().lines() {
                let line = line.unwrap_or_default();
                let line = line.trim_end_matches('\n');
                if line.is_empty() {
                    continue;
                }
                let text = parse_encode_input(line);
                encode_one(tok, &text, seq, rich, add_special_tokens, compact, &mut out);
                seq += 1;
            }
        }
        "whole" => {
            let mut text = String::new();
            io::stdin().read_line(&mut text).ok();
            let all = stdin.lock().lines().fold(text, |mut acc, l| {
                acc.push('\n');
                acc.push_str(&l.unwrap_or_default());
                acc
            });
            if !all.trim().is_empty() {
                encode_one(tok, &all, 0, rich, add_special_tokens, compact, &mut out);
            }
        }
        _ => {
            // paragraph mode
            let mut buf = Vec::new();
            for line in stdin.lock().lines() {
                let line = line.unwrap_or_default();
                let stripped = line.trim_end_matches('\n');
                if stripped.is_empty() && !buf.is_empty() {
                    let text = buf.join("\n");
                    encode_one(tok, &text, seq, rich, add_special_tokens, compact, &mut out);
                    seq += 1;
                    buf.clear();
                } else {
                    buf.push(stripped.to_string());
                }
            }
            if !buf.is_empty() {
                let text = buf.join("\n");
                encode_one(tok, &text, seq, rich, add_special_tokens, compact, &mut out);
            }
        }
    }
}

fn encode_one(
    tok: &Tokenizer,
    text: &str,
    seq: u64,
    rich: bool,
    add_special_tokens: bool,
    compact: bool,
    out: &mut impl Write,
) {
    let t0 = Instant::now();
    if rich {
        match tok.encode_rich(text, add_special_tokens, true) {
            Ok(enc) => {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let mut record = serde_json::json!({
                    "seq": seq,
                    "text": text,
                    "ids": enc.ids,
                    "n_tokens": enc.ids.len(),
                });
                if !compact {
                    record["input_chars"] = serde_json::json!(text.len());
                    record["elapsed_ms"] =
                        serde_json::json!((elapsed_ms * 1000.0).round() / 1000.0);
                }
                if let Some(offsets) = &enc.offsets {
                    let offsets_arr: Vec<[usize; 2]> =
                        offsets.iter().map(|(s, e)| [*s, *e]).collect();
                    record["offsets"] = serde_json::json!(offsets_arr);
                }
                writeln!(out, "{}", serde_json::to_string(&record).unwrap()).ok();
            }
            Err(e) => eprintln!("Encode error: {e}"),
        }
    } else {
        match tok.encode(text, add_special_tokens) {
            Ok(ids) => {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let mut record = serde_json::json!({
                    "seq": seq,
                    "text": text,
                    "ids": ids,
                    "n_tokens": ids.len(),
                });
                if !compact {
                    record["input_chars"] = serde_json::json!(text.len());
                    record["elapsed_ms"] =
                        serde_json::json!((elapsed_ms * 1000.0).round() / 1000.0);
                }
                writeln!(out, "{}", serde_json::to_string(&record).unwrap()).ok();
            }
            Err(e) => eprintln!("Encode error: {e}"),
        }
    }
    out.flush().ok();
}

fn cmd_decode(tok: &Tokenizer, skip_special_tokens: bool, compact: bool) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for (seq, line) in stdin.lock().lines().enumerate() {
        let line = line.unwrap_or_default();
        let line = line.trim_end_matches('\n');
        if line.is_empty() {
            continue;
        }
        let ids = parse_decode_input(line);
        let t0 = Instant::now();
        match tok.decode(&ids, skip_special_tokens) {
            Ok(text) => {
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let mut record = serde_json::json!({
                    "seq": seq,
                    "ids": ids,
                    "text": text,
                    "n_tokens": ids.len(),
                });
                if !compact {
                    record["elapsed_ms"] =
                        serde_json::json!((elapsed_ms * 1000.0).round() / 1000.0);
                }
                writeln!(out, "{}", serde_json::to_string(&record).unwrap()).ok();
            }
            Err(e) => eprintln!("Decode error: {e}"),
        }
        out.flush().ok();
    }
}

fn cmd_info(tok: &Tokenizer) {
    let mut info = serde_json::json!({
        "vocab_size": tok.vocab_size(),
        "model_type": tok.model_type(),
    });
    if let Some(id) = tok.bos_token_id() {
        info["bos_token_id"] = serde_json::json!(id);
    }
    if let Some(id) = tok.eos_token_id() {
        info["eos_token_id"] = serde_json::json!(id);
    }
    if let Some(id) = tok.unk_token_id() {
        info["unk_token_id"] = serde_json::json!(id);
    }
    if let Some(id) = tok.pad_token_id() {
        info["pad_token_id"] = serde_json::json!(id);
    }
    if let Some(id) = tok.sep_token_id() {
        info["sep_token_id"] = serde_json::json!(id);
    }
    if let Some(id) = tok.cls_token_id() {
        info["cls_token_id"] = serde_json::json!(id);
    }
    if let Some(id) = tok.mask_token_id() {
        info["mask_token_id"] = serde_json::json!(id);
    }
    println!("{}", serde_json::to_string_pretty(&info).unwrap());
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Encode {
            tokenizer_path,
            tokenizer_json,
            encoding,
            input_mode,
            rich,
            add_special_tokens,
            compact,
        } => {
            let tok = load_tokenizer(tokenizer_path, tokenizer_json, encoding);
            cmd_encode(&tok, input_mode, *rich, *add_special_tokens, *compact);
        }
        Commands::Decode {
            tokenizer_path,
            tokenizer_json,
            encoding,
            skip_special_tokens,
            compact,
        } => {
            let tok = load_tokenizer(tokenizer_path, tokenizer_json, encoding);
            cmd_decode(&tok, *skip_special_tokens, *compact);
        }
        Commands::Info {
            tokenizer_path,
            tokenizer_json,
            encoding,
        } => {
            let tok = load_tokenizer(tokenizer_path, tokenizer_json, encoding);
            cmd_info(&tok);
        }
    }
}
