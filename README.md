# iree-tokenizer

Rust bindings for the [IREE](https://github.com/iree-org/iree) tokenizer C library. Provides a safe, ergonomic API for BPE and tiktoken tokenization with full feature parity to the [Python binding](https://github.com/iree-org/iree-tokenizer-py).

## Features

- **Encode/decode** text with HuggingFace `tokenizer.json` or tiktoken formats
- **Batch encoding** with native C-level batching
- **Streaming** encode/decode for incremental processing
- **Rich encoding** with byte offset tracking
- **Vocabulary introspection** — token lookup, special token IDs, model type
- **Thread-safe** — `Send + Sync` (immutable after construction)
- **CLI** for encode, decode, and tokenizer info

## Prerequisites

- [IREE source tree](https://github.com/iree-org/iree) checked out locally
- CMake and Ninja
- A C/C++ compiler (clang or gcc)
- libclang (for bindgen)

## Building

The build system compiles the IREE tokenizer C library from source via CMake, then generates Rust FFI bindings with bindgen.

```bash
# Point to your IREE checkout (defaults to /Users/simolin/opensource/iree)
export IREE_SOURCE_DIR=/path/to/iree

cargo build
```

## Usage

### Library

```rust
use iree_tokenizer::Tokenizer;

// Load from a HuggingFace tokenizer.json file
let tok = Tokenizer::from_file("tokenizer.json").unwrap();

// Encode
let ids = tok.encode("Hello world", false).unwrap();
println!("{:?}", ids);

// Decode
let text = tok.decode(&ids, false).unwrap();
assert_eq!(text, "Hello world");

// Rich encode with byte offsets
let enc = tok.encode_rich("Hello world", false, true).unwrap();
println!("ids: {:?}, offsets: {:?}", enc.ids, enc.offsets);

// Batch encode
let batch = tok.encode_batch(&["Hello", "World"], false).unwrap();

// Streaming encode
let mut stream = tok.encode_stream(false).unwrap();
stream.feed("Hello ")?;
stream.feed("world")?;
let ids = stream.finalize()?;

// Vocabulary info
println!("vocab size: {}", tok.vocab_size());
println!("model type: {}", tok.model_type());
println!("BOS token: {:?}", tok.bos_token_id());
```

### Tiktoken

```rust
use iree_tokenizer::Tokenizer;

// Load from a .tiktoken file with a named encoding
let tok = Tokenizer::from_tiktoken_file("vocab.tiktoken", "gpt2").unwrap();
let ids = tok.encode("Hello world", false).unwrap();
```

### CLI

```bash
# Encode (reads lines from stdin, outputs JSONL)
echo "Hello world" | cargo run -- encode -t tokenizer.json

# Decode (reads JSON arrays from stdin)
echo '[72, 101, 108, 108, 111]' | cargo run -- decode -t tokenizer.json

# Tokenizer info
cargo run -- info -t tokenizer.json
```

## Testing

```bash
cargo test
```

Runs 44 integration tests across 7 test files covering loading, encoding, decoding, batching, streaming, vocabulary, and tiktoken support.

## Benchmarks

Criterion benchmarks compare IREE against HuggingFace `tokenizers` on the same vocabulary:

```bash
cargo bench
```

Results on Apple M-series (same 112-token BPE vocabulary):

| Operation | Corpus | IREE | HuggingFace | Speedup |
|-----------|--------|-----:|------------:|--------:|
| Encode | short (44 chars) | 3.1 us | 10.4 us | 3.3x |
| Encode | medium (880 chars) | 13.5 us | 155.9 us | 11.6x |
| Encode | long (22K chars) | 253.9 us | 3807.8 us | 15.0x |
| Decode | short (38 tokens) | 106 ns | 2.6 us | 24.5x |
| Decode | medium (456 tokens) | 987 ns | 46.3 us | 46.9x |
| Decode | long (11K tokens) | 23.7 us | 1152.2 us | 48.6x |

## Project Structure

```
src/
  lib.rs          Public API re-exports
  tokenizer.rs    Safe Tokenizer wrapper
  stream.rs       EncodeStream / DecodeStream RAII wrappers
  encoding.rs     Encoding result type (ids, offsets, type_ids)
  error.rs        Error types mapping IREE status codes
  ffi.rs          Bindgen output + manual inline function reimplementations
  main.rs         CLI binary
build.rs          CMake + bindgen build script
benches/
  bench_comparison.rs   Criterion benchmarks (IREE vs HuggingFace)
tests/
  test_load.rs          Tokenizer construction
  test_encode.rs        Encoding correctness
  test_decode.rs        Decoding correctness
  test_batch.rs         Batch encode/decode
  test_streaming.rs     Streaming encode/decode
  test_vocab.rs         Vocabulary introspection
  test_tiktoken.rs      Tiktoken format support
```

## License

Apache-2.0 WITH LLVM-exception
