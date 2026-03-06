# iree-tokenizer

Rust bindings for the [IREE tokenizer](https://github.com/iree-org/iree/blob/main/runtime/src/iree/tokenizer/README.md) —
a high-performance C tokenizer with full HuggingFace `tokenizer.json` and
OpenAI tiktoken compatibility.

- **Fast.** 3-15x faster encode, 25-49x faster decode vs HuggingFace
  tokenizers. Pure C hot path with zero allocations per token.
- **Safe.** Idiomatic Rust API with RAII, `Result` error handling, and
  `Send + Sync` thread safety. No `unsafe` in user-facing code.
- **Streaming encode/decode.** First-class support for incremental
  tokenization — feed chunks in, get tokens out. Ideal for LLM inference.
- **Drop-in compatible.** Loads any HuggingFace `tokenizer.json` or OpenAI
  `.tiktoken` vocabulary. Supports BPE, WordPiece, and Unigram models.

Based on the [IREE high-speed tokenizer library](https://github.com/iree-org/iree/blob/main/runtime/src/iree/tokenizer/README.md):

- **Optimized for cache utilization.** Efficiently utilizes cache on both
  large and small CPUs. No dependencies and small footprint make it ideal
  for embedded/client and inclusion into other projects.
- **Unique algorithmic optimizations.** Pull-based streaming processor with
  bounded/small, deterministic memory usage. Various novel optimizations not
  seen elsewhere.
- **GPU-ready.** Designed to be compatible with executing tiled on the GPU,
  not just the host.

## Performance

Criterion benchmarks comparing IREE against HuggingFace
[tokenizers](https://github.com/huggingface/tokenizers) on the same
112-token BPE vocabulary (Apple M-series, single-threaded):

### Encode

| Corpus | IREE | HuggingFace | Speedup |
|--------|-----:|------------:|--------:|
| short (44 chars) | 3.1 us | 10.4 us | **3.3x** |
| medium (880 chars) | 13.5 us | 155.9 us | **11.6x** |
| long (22K chars) | 253.9 us | 3807.8 us | **15.0x** |

### Decode

| Corpus | IREE | HuggingFace | Speedup |
|--------|-----:|------------:|--------:|
| short (38 tokens) | 106 ns | 2.6 us | **24.5x** |
| medium (456 tokens) | 987 ns | 46.3 us | **46.9x** |
| long (11K tokens) | 23.7 us | 1152.2 us | **48.6x** |

Run benchmarks yourself:

```bash
cargo bench
```

## Prerequisites

- CMake and Ninja
- A C/C++ compiler (clang or gcc)
- libclang (for bindgen)

## Building

The IREE C library source is included as a git submodule and compiled
automatically via CMake during `cargo build`.

```bash
git clone https://github.com/lightseekorg/iree-tokenizer-rust.git
cd iree-tokenizer-rust

# Initialize submodules (shallow clone to minimize download)
git submodule update --init --depth 1
cd third_party/iree && git submodule update --init --depth 1 third_party/flatcc && cd ../..

cargo build
```

To build against a local IREE source tree instead of the submodule:

```bash
IREE_SOURCE_DIR=/path/to/iree cargo build
```

## Quick Start

```rust
use iree_tokenizer::Tokenizer;

// Load from a HuggingFace tokenizer.json
let tok = Tokenizer::from_file("tokenizer.json")?;

// Encode / decode
let ids = tok.encode("Hello world", false)?;     // vec![15496, 995]
let text = tok.decode(&ids, false)?;              // "Hello world"

// Batch encode
let batch = tok.encode_batch(&["Hello", "World"], false)?;

// Rich encoding with byte offsets
let enc = tok.encode_rich("Hello world", false, true)?;
// enc.ids, enc.offsets, enc.type_ids

// Streaming encode (feed chunks, get tokens incrementally)
let mut stream = tok.encode_stream(false)?;
stream.feed("Hello ")?;
stream.feed("world")?;
let ids = stream.finalize()?;

// Vocabulary introspection
tok.vocab_size();                // 50257
tok.model_type();                // "BPE"
tok.token_to_id("hello");       // Some(31373)
tok.id_to_token(31373);         // Some("hello")
tok.bos_token_id();             // Some(1)
```

### Tiktoken

```rust
use iree_tokenizer::Tokenizer;

let tok = Tokenizer::from_tiktoken_file("cl100k_base.tiktoken", "cl100k_base")?;
let ids = tok.encode("Hello world", false)?;
```

## API

| Method | Returns | Description |
|--------|---------|-------------|
| `Tokenizer::from_file(path)` | `Result<Tokenizer>` | Load from `tokenizer.json` |
| `Tokenizer::from_str(json)` | `Result<Tokenizer>` | Load from JSON string |
| `Tokenizer::from_tiktoken_file(path, enc)` | `Result<Tokenizer>` | Load from `.tiktoken` file |
| `tok.encode(text, add_special)` | `Result<Vec<i32>>` | Encode text to token IDs |
| `tok.encode_rich(text, special, offsets)` | `Result<Encoding>` | IDs + byte offsets + type IDs |
| `tok.decode(ids, skip_special)` | `Result<String>` | Decode token IDs to text |
| `tok.encode_batch(texts, add_special)` | `Result<Vec<Vec<i32>>>` | Batch encode |
| `tok.decode_batch(id_lists, skip_special)` | `Result<Vec<String>>` | Batch decode |
| `tok.encode_stream(add_special)` | `Result<EncodeStream>` | Streaming encoder |
| `tok.decode_stream(skip_special)` | `Result<DecodeStream>` | Streaming decoder |
| `tok.vocab_size()` | `usize` | Vocabulary size |
| `tok.model_type()` | `String` | `"BPE"`, `"WordPiece"`, or `"Unigram"` |
| `tok.token_to_id(token)` | `Option<i32>` | Look up token ID |
| `tok.id_to_token(id)` | `Option<String>` | Look up token text |

## CLI

A streaming `iree-tokenizer` command is included. It reads from stdin, writes
JSONL to stdout.

```bash
# Encode text to token IDs
echo "Hello world" | iree-tokenizer encode -t tokenizer.json
# {"seq":0,"text":"Hello world","ids":[15496,995],"n_tokens":2,...}

# Encode with a tiktoken vocabulary
echo "Hello world" | iree-tokenizer encode -t cl100k_base.tiktoken --encoding cl100k_base

# Decode token IDs back to text
echo '[15496, 995]' | iree-tokenizer decode -t tokenizer.json
# {"seq":0,"ids":[15496,995],"text":"Hello world","n_tokens":2,...}

# Chain encode -> decode (round-trip)
cat corpus.txt | iree-tokenizer encode -t tokenizer.json | iree-tokenizer decode -t tokenizer.json

# Tokenizer info
iree-tokenizer info -t tokenizer.json
```

Use `--compact` to omit timing fields, `--rich` for byte offsets.

## Testing

```bash
cargo test
```

44 integration tests across 7 test files covering loading, encoding,
decoding, batching, streaming, vocabulary introspection, and tiktoken support.

## Project Structure

```
src/
  lib.rs            Public API re-exports
  tokenizer.rs      Safe Tokenizer wrapper over FFI
  stream.rs         EncodeStream / DecodeStream (RAII)
  encoding.rs       Encoding result type (ids, offsets, type_ids)
  error.rs          Error types mapping IREE status codes
  ffi.rs            Bindgen output + manual inline reimplementations
  main.rs           CLI binary
build.rs            CMake + bindgen build script
third_party/iree/   IREE C library (git submodule)
benches/
  bench_comparison.rs   Criterion benchmarks (IREE vs HuggingFace)
tests/
  test_load.rs      Tokenizer construction
  test_encode.rs    Encoding correctness
  test_decode.rs    Decoding correctness
  test_batch.rs     Batch encode/decode
  test_streaming.rs Streaming encode/decode
  test_vocab.rs     Vocabulary introspection
  test_tiktoken.rs  Tiktoken format support
```

## Acknowledgments

This project wraps the [IREE tokenizer C library](https://github.com/iree-org/iree/tree/main/runtime/src/iree/tokenizer),
developed as part of the [IREE](https://github.com/iree-org/iree) project.
The API design follows the [iree-tokenizer-py](https://github.com/iree-org/iree-tokenizer-py)
Python bindings, aiming for feature parity with the Rust ecosystem.

## License

Apache-2.0 WITH LLVM-exception -- see [LICENSE](LICENSE).
