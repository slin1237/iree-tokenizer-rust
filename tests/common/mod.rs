#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    dead_code,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::{fs, path::PathBuf};

use iree_tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Tokenizer download + cache
// ---------------------------------------------------------------------------

const QWEN_TOKENIZER_URL: &str =
    "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer.json";
const QWEN_CACHE_FILE: &str = "qwen2.5_tokenizer.json";

fn cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/.cache")
}

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata")
}

/// Download a file from `url` and cache it locally. Returns the cached path.
fn download_or_cache(url: &str, filename: &str) -> PathBuf {
    let dir = cache_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(filename);
    if path.exists() {
        return path;
    }
    println!("Downloading {filename} from {url} ...");
    let resp = ureq::get(url).call().expect("failed to download tokenizer");
    let body = resp
        .into_body()
        .read_to_vec()
        .expect("failed to read response body");
    fs::write(&path, &body).unwrap();
    println!("Cached at {}", path.display());
    path
}

// ---------------------------------------------------------------------------
// Tokenizer constructors
// ---------------------------------------------------------------------------

/// Qwen2.5-7B tokenizer (151K+ vocab, production-scale BPE).
pub fn qwen_tokenizer() -> Tokenizer {
    let path = download_or_cache(QWEN_TOKENIZER_URL, QWEN_CACHE_FILE);
    Tokenizer::from_file(&path).unwrap()
}

/// Minimal 112-token BPE tokenizer for edge case tests.
pub fn minimal_tokenizer() -> Tokenizer {
    let path = testdata_dir().join("bpe_bytelevel_minimal.json");
    Tokenizer::from_file(path).unwrap()
}

/// GPT-2 tiktoken tokenizer (261 tokens).
pub fn tiktoken_tokenizer() -> Tokenizer {
    let path = testdata_dir().join("tiktoken_gpt2.tiktoken");
    Tokenizer::from_tiktoken_file(path, "gpt2").unwrap()
}

/// Returns the path to the cached Qwen tokenizer (downloading if needed).
pub fn qwen_tokenizer_path() -> PathBuf {
    download_or_cache(QWEN_TOKENIZER_URL, QWEN_CACHE_FILE)
}

// ---------------------------------------------------------------------------
// Test corpus — diverse, realistic text at multiple scales
// ---------------------------------------------------------------------------

pub const SHORT_TEXT: &str =
    "The Rust programming language helps you write faster, more reliable software.";

pub const MEDIUM_TEXT: &str = "\
Transformers have revolutionized natural language processing since the publication \
of \"Attention Is All You Need\" in 2017. The key innovation is the self-attention \
mechanism, which allows the model to weigh the importance of different parts of the \
input sequence when producing each output token. Unlike RNNs, transformers process \
all positions in parallel, leading to significant speedups during training. Modern \
LLMs like GPT-4, Claude, and Qwen2.5 are all based on this architecture, typically \
using decoder-only designs with billions of parameters. Fine-tuning techniques such \
as LoRA (Low-Rank Adaptation) and RLHF (Reinforcement Learning from Human Feedback) \
have made these models increasingly capable at following instructions and generating \
high-quality text. The tokenizer is a critical component: it converts raw text into \
integer token IDs that the model can process, and its vocabulary size directly affects \
both model capacity and inference speed.";

pub const LONG_TEXT: &str = "\
# Understanding Tokenization in Large Language Models

Tokenization is the process of converting raw text into a sequence of integer token \
IDs that a language model can process. It is one of the most fundamental yet often \
overlooked components of the modern NLP pipeline. The choice of tokenizer affects \
model quality, inference latency, and the effective context window available to users.

## Byte-Pair Encoding (BPE)

BPE is the most widely used tokenization algorithm in modern LLMs. Originally \
proposed for data compression by Philip Gage in 1994, it was adapted for NLP by \
Sennrich et al. in 2016. The algorithm works by iteratively merging the most \
frequent pair of adjacent tokens in a training corpus until the desired vocabulary \
size is reached.

For example, starting with individual characters, BPE might first merge 't' and 'h' \
into 'th', then merge 'th' and 'e' into 'the'. This creates a vocabulary that \
efficiently represents common subwords while still being able to encode any text \
by falling back to individual bytes.

## Vocabulary Design

The vocabulary size is a critical hyperparameter. Smaller vocabularies (e.g., 32K \
tokens) produce longer sequences but generalize better to rare words. Larger \
vocabularies (e.g., 150K+ tokens like Qwen2.5) produce shorter sequences and can \
represent more words as single tokens, but require more model parameters in the \
embedding layer.

Modern tokenizers like those used in GPT-4 and Qwen2.5 use byte-level BPE, meaning \
they can represent any byte sequence. This eliminates unknown tokens entirely — a \
significant improvement over earlier approaches that used a fixed character set.

## Performance Considerations

Tokenizer performance matters more than many practitioners realize. In a typical \
LLM inference pipeline, tokenization happens on the critical path before any GPU \
computation begins. A slow tokenizer can add significant latency, especially for \
streaming applications where the first token latency (TTFT) is critical.

The IREE tokenizer addresses this by implementing the entire tokenization pipeline \
in optimized C code with zero allocations per token. It uses a pull-based streaming \
architecture that processes text incrementally, making it ideal for real-time \
applications. Benchmarks show it is 3-15x faster than HuggingFace tokenizers and \
2-12x faster than tiktoken for encoding, with even larger speedups for decoding.

## Streaming Tokenization

Traditional tokenizers process text in a batch: you provide the full input text and \
receive all token IDs at once. Streaming tokenization allows you to feed text \
incrementally and receive tokens as they become available. This is particularly \
useful for:

1. Real-time transcription: Tokenize speech-to-text output as it arrives
2. Interactive editors: Tokenize as the user types for live token counting
3. Pipeline processing: Avoid buffering entire documents before tokenization
4. Memory efficiency: Process large documents without loading them entirely

The IREE tokenizer provides first-class streaming support through its EncodeStream \
and DecodeStream APIs, which maintain internal state across feed() calls and produce \
tokens as soon as they are determined.

## Special Tokens

Most tokenizers define special tokens for model control: BOS (beginning of sequence), \
EOS (end of sequence), PAD (padding), UNK (unknown), SEP (separator), CLS \
(classification), and MASK (masked language modeling). These tokens have dedicated \
IDs in the vocabulary and can be optionally added during encoding.

Qwen2.5 uses a vocabulary of 151,643 tokens with special tokens including \
<|endoftext|> (EOS), <|im_start|>, and <|im_end|> for chat formatting.";

pub fn very_long_text() -> String {
    // ~50K chars: repeat the long text ~10 times with variation
    let mut text = String::with_capacity(55000);
    for i in 0..10 {
        if i > 0 {
            text.push_str("\n\n---\n\n");
        }
        text.push_str(&format!("## Section {}\n\n", i + 1));
        text.push_str(LONG_TEXT);
    }
    text
}

pub const MULTILINGUAL_TEXT: &str = "\
Large language models have made remarkable progress in recent years. \
大型语言模型在近年来取得了显著的进展。\
Qwen2.5 supports both English and Chinese natively with strong performance. \
Qwen2.5原生支持中英双语，表现出色。\
Tokenization must handle mixed scripts efficiently: \
分词器必须高效处理混合脚本：\
Unicode normalization, byte-level fallback, and script detection are all essential. \
Unicode标准化、字节级回退和脚本检测都是必不可少的。\
こんにちは世界 — Japanese text is also common in multilingual models. \
한국어도 지원됩니다 — Korean support matters too.";

pub const CODE_TEXT: &str = r#"
use std::collections::HashMap;

/// A simple LRU cache implementation.
pub struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, (V, u64)>,
    counter: u64,
}

impl<K: Eq + std::hash::Hash, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            counter: 0,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.counter += 1;
        if let Some(entry) = self.map.get_mut(key) {
            entry.1 = self.counter;
            Some(&entry.0)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Evict least recently used
            let lru_key = self.map.iter()
                .min_by_key(|(_, (_, ts))| *ts)
                .map(|(k, _)| k as *const K)
                .unwrap();
            let lru_key = unsafe { &*lru_key }.clone();
            self.map.remove(&lru_key);
        }
        self.counter += 1;
        self.map.insert(key, (value, self.counter));
    }
}

fn main() {
    let mut cache = LruCache::new(3);
    cache.insert("hello", 42);
    cache.insert("world", 99);
    println!("hello = {:?}", cache.get(&"hello"));
}
"#;
