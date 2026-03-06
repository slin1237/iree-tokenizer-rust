# IREE Tokenizer Rust Binding

Build a Rust binding crate (`iree-tokenizer`) that wraps the IREE tokenizer C library,
providing feature parity with the Python binding at `/Users/simolin/opensource/iree-tokenizer-py`.

## Reference Materials

- **C API headers**: `/Users/simolin/opensource/iree/runtime/src/iree/tokenizer/tokenizer.h`, `types.h`
- **Python binding source**: `/Users/simolin/opensource/iree-tokenizer-py/src/bindings/` (nanobind C++)
- **Python public API**: `/Users/simolin/opensource/iree-tokenizer-py/src/iree/tokenizer/__init__.py`
- **Python tests**: `/Users/simolin/opensource/iree-tokenizer-py/tests/`
- **Python benchmarks**: `/Users/simolin/opensource/iree-tokenizer-py/benchmarks/bench_comparison.py`
- **Test data**: `/Users/simolin/opensource/iree-tokenizer-py/tests/testdata/`
- **Working directory**: `/Users/simolin/opensource/slin1237/iree-tokenizer-rust`

## Git Discipline

**Commit early and often.** After every meaningful change, create a git commit:
- Use conventional commit format: `feat:`, `fix:`, `build:`, `test:`, `bench:`, `refactor:`, `docs:`
- Use `git commit -s` to add DCO Signed-off-by line
- Write detailed multi-line commit messages (what changed, why, how)
- Never include AI attribution in commit messages
- Examples of commit points:
  - `build.rs` compiles IREE C lib successfully
  - FFI bindings generated and compiling
  - Each major API surface area implemented (Tokenizer, Encoding, streams)
  - Each test file added and passing
  - Benchmark suite added
  - CLI implemented
- Do NOT batch all work into a single commit. Each logical unit of work gets its own commit.

## Architecture

### Crate Structure
```
iree-tokenizer-rust/
├── Cargo.toml
├── build.rs                    # Build script: compile IREE C lib, generate FFI bindings
├── src/
│   ├── lib.rs                  # Public API re-exports
│   ├── ffi.rs                  # Raw unsafe FFI bindings (bindgen or manual)
│   ├── tokenizer.rs            # Safe Tokenizer wrapper
│   ├── encoding.rs             # Encoding result type (ids, offsets, type_ids)
│   ├── stream.rs               # EncodeStream, DecodeStream (RAII wrappers)
│   ├── error.rs                # Error types mapping IREE status codes
│   └── main.rs                 # CLI binary (encode/decode/info subcommands)
├── benches/
│   └── bench_comparison.rs     # Criterion benchmarks vs HF tokenizers & tiktoken
├── tests/
│   ├── test_load.rs
│   ├── test_encode.rs
│   ├── test_decode.rs
│   ├── test_batch.rs
│   ├── test_streaming.rs
│   ├── test_vocab.rs
│   └── test_tiktoken.rs
└── testdata/ -> symlink or copy from iree-tokenizer-py/tests/testdata/
```

### Build System (`build.rs`)
- Use `IREE_SOURCE_DIR` env var (default: `/Users/simolin/opensource/iree`) to locate IREE source
- Compile the IREE tokenizer C library via `cmake` (same approach as Python binding's CMakeLists.txt)
- Link against: `iree_tokenizer_tokenizer`, `iree_tokenizer_format_huggingface_tokenizer_json`
- Generate FFI bindings with `bindgen` from `tokenizer.h` and `types.h`
- Set `cargo:rerun-if-changed` for header files

### FFI Layer (`ffi.rs`)
Raw bindings to the C API. Key functions to bind:
```rust
// Lifecycle
iree_tokenizer_builder_initialize / build / deinitialize
iree_tokenizer_free

// One-shot encode/decode
iree_tokenizer_encode / iree_tokenizer_decode

// Batch
iree_tokenizer_encode_batch / iree_tokenizer_decode_batch

// Streaming encode
iree_tokenizer_encode_state_calculate_size
iree_tokenizer_transform_buffer_recommended_size
iree_tokenizer_encode_state_initialize / feed / finalize / deinitialize / reset

// Streaming decode
iree_tokenizer_decode_state_calculate_size
iree_tokenizer_decode_state_initialize / feed / finalize / deinitialize

// Format loaders
iree_tokenizer_load_huggingface_tokenizer_json
iree_tokenizer_load_tiktoken
```

### Error Handling (`error.rs`)
Map IREE status codes to a Rust `Error` enum:
```rust
pub enum Error {
    InvalidArgument(String),   // IREE_STATUS_INVALID_ARGUMENT
    NotFound(String),          // IREE_STATUS_NOT_FOUND
    Unimplemented(String),     // IREE_STATUS_UNIMPLEMENTED
    ResourceExhausted(String), // IREE_STATUS_RESOURCE_EXHAUSTED
    Internal(String),          // All others
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
```

## Required Public API (Feature Parity with Python)

### `Tokenizer` (thread-safe: `Send + Sync`)
```rust
impl Tokenizer {
    // Construction
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self>;
    pub fn from_str(json: &str) -> Result<Self>;
    pub fn from_bytes(data: &[u8]) -> Result<Self>;
    pub fn from_tiktoken_file(path: impl AsRef<Path>, encoding: &str) -> Result<Self>;
    pub fn from_tiktoken_str(data: &str, encoding: &str) -> Result<Self>;
    pub fn from_tiktoken_bytes(data: &[u8], encoding: &str) -> Result<Self>;

    // Encoding
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>>;
    pub fn encode_rich(&self, text: &str, add_special_tokens: bool, track_offsets: bool) -> Result<Encoding>;
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<i32>>>;
    pub fn encode_batch_flat(&self, texts: &[&str], add_special_tokens: bool) -> Result<(Vec<i32>, Vec<u64>)>;

    // Decoding
    pub fn decode(&self, token_ids: &[i32], skip_special_tokens: bool) -> Result<String>;
    pub fn decode_batch(&self, token_id_lists: &[&[i32]], skip_special_tokens: bool) -> Result<Vec<String>>;

    // Streaming
    pub fn encode_stream(&self, add_special_tokens: bool) -> Result<EncodeStream>;
    pub fn decode_stream(&self, skip_special_tokens: bool) -> Result<DecodeStream>;

    // Vocabulary
    pub fn vocab_size(&self) -> usize;
    pub fn model_type(&self) -> &str;  // "BPE", "WordPiece", or "Unigram"
    pub fn token_to_id(&self, token: &str) -> Option<i32>;
    pub fn id_to_token(&self, id: i32) -> Option<&str>;

    // Special token IDs (None if not configured)
    pub fn bos_token_id(&self) -> Option<i32>;
    pub fn eos_token_id(&self) -> Option<i32>;
    pub fn unk_token_id(&self) -> Option<i32>;
    pub fn pad_token_id(&self) -> Option<i32>;
    pub fn sep_token_id(&self) -> Option<i32>;
    pub fn cls_token_id(&self) -> Option<i32>;
    pub fn mask_token_id(&self) -> Option<i32>;
}
impl fmt::Display for Tokenizer { /* Tokenizer(model_type='BPE', vocab_size=50257) */ }
impl Drop for Tokenizer { /* calls iree_tokenizer_free */ }
// SAFETY: iree_tokenizer_t is immutable after construction
unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}
```

### `Encoding`
```rust
pub struct Encoding {
    pub ids: Vec<i32>,
    pub offsets: Option<Vec<(usize, usize)>>,  // (start, end) byte offsets
    pub type_ids: Vec<u8>,
}
impl Encoding {
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

### `EncodeStream`
```rust
impl EncodeStream<'_> {
    pub fn feed(&mut self, text: &str) -> Result<Vec<i32>>;
    pub fn finalize(&mut self) -> Result<Vec<i32>>;
    pub fn is_open(&self) -> bool;
}
impl Drop for EncodeStream<'_> { /* releases state */ }
```

### `DecodeStream`
```rust
impl DecodeStream<'_> {
    pub fn feed(&mut self, token_ids: &[i32]) -> Result<String>;
    pub fn finalize(&mut self) -> Result<String>;
    pub fn is_open(&self) -> bool;
}
impl Drop for DecodeStream<'_> { /* releases state */ }
```

### `decode_stream_iter`
```rust
pub fn decode_stream_iter<I>(
    tokenizer: &Tokenizer,
    token_iter: I,
    skip_special_tokens: bool,
) -> Result<impl Iterator<Item = Result<String>> + '_>
where
    I: IntoIterator<Item = i32> + '_;
```

### CLI (`main.rs`)
Subcommands matching the Python CLI:
```
iree-tokenizer encode -t <tokenizer.json> [--add-special-tokens] [--rich] [--input-mode line|paragraph|whole]
iree-tokenizer decode -t <tokenizer.json> [--skip-special-tokens]
iree-tokenizer info -t <tokenizer.json>
```
- Read from stdin, write JSONL to stdout
- Support `--tokenizer-json` for inline JSON
- Support `--encoding` for tiktoken files
- Support `--compact` to omit timing fields

## Benchmark Suite

### Overview
Port the Python benchmark (`/Users/simolin/opensource/iree-tokenizer-py/benchmarks/bench_comparison.py`)
to Rust using **Criterion** for statistical benchmarking, comparing against the Rust crates for
HuggingFace tokenizers and tiktoken.

### Dependencies (Cargo.toml)
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
tokenizers = "0.21"          # HuggingFace tokenizers Rust crate
tiktoken-rs = "0.6"          # tiktoken Rust crate

[[bench]]
name = "bench_comparison"
harness = false
```

### Benchmark File: `benches/bench_comparison.rs`

**Test corpus** (matching Python):
```rust
const SHORT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
// MEDIUM_TEXT = SHORT_TEXT repeated 20x  (~900 chars)
// LONG_TEXT   = SHORT_TEXT repeated 500x (~22K chars)
```

**Benchmark groups** (matching Python's three tables):

1. **Encode Latency** — `encode/{backend}/{corpus_size}`
   - Backends: `iree`, `huggingface`, `tiktoken`
   - Corpus sizes: `short`, `medium`, `long`
   - Measures: single-string encode latency

2. **Decode Latency** — `decode/{backend}/{corpus_size}`
   - Pre-encode with each backend, then benchmark decode
   - Backends: `iree`, `huggingface`, `tiktoken`

3. **Batch Encode** — `batch_encode/{backend}/batch_{n}`
   - Batch sizes: 1, 10, 100
   - Backends: `iree`, `huggingface`, `tiktoken`
   - iree uses native `encode_batch`, others use loops

**Tokenizer loading** (matching Python):
- iree: `Tokenizer::from_file()` with local tokenizer.json
  (download GPT-2 tokenizer.json from HuggingFace or use testdata)
- HuggingFace: `tokenizers::Tokenizer::from_pretrained("openai-community/gpt2")`
- tiktoken: `tiktoken_rs::get_encoding("gpt2")`

**Graceful degradation**: If a backend crate fails to load/initialize, skip its benchmarks
(don't panic, just omit that group).

### Benchmark CLI

Also provide a standalone comparison binary for quick results:
```toml
[[bin]]
name = "bench-compare"
path = "src/bench_compare.rs"
required-features = ["bench"]

[features]
bench = ["dep:tokenizers", "dep:tiktoken-rs"]
```

```
cargo run --release --bin bench-compare --features bench -- [--iterations 50] [--warmup 5] [--validate]
```

Outputs a formatted table (similar to Python's Rich output) showing:
- p50/p99 latency per backend per corpus size
- tokens/sec throughput
- Relative speedup vs HuggingFace baseline

When `--validate` is passed: verify all backends produce identical token IDs for the same input,
and verify encode→decode roundtrip for each backend.

### Running Benchmarks
```bash
# Criterion benchmarks (statistical, with HTML reports)
cargo bench

# Quick comparison table
cargo run --release --bin bench-compare --features bench

# With validation
cargo run --release --bin bench-compare --features bench -- --validate
```

## Testing Requirements

Port all tests from `/Users/simolin/opensource/iree-tokenizer-py/tests/`:
- `test_load`: from_file, from_str, from_bytes, error cases
- `test_encode`: basic encode, empty string, roundtrip, encode_rich with offsets
- `test_decode`: basic decode, empty list, roundtrip, unicode
- `test_batch`: encode_batch, decode_batch, encode_batch_flat shapes
- `test_streaming`: encode_stream/decode_stream feed/finalize, decode_stream_iter
- `test_vocab`: vocab_size, model_type, token_to_id, id_to_token, special tokens
- `test_tiktoken`: load tiktoken files, all encodings, roundtrip

Use test data from `/Users/simolin/opensource/iree-tokenizer-py/tests/testdata/`.
Copy test data files into `tests/testdata/` in this project.

## Implementation Order

1. `build.rs` + `ffi.rs` — compile C lib and generate bindings → **commit**
2. `error.rs` — error types → **commit**
3. `tokenizer.rs` — Tokenizer struct with construction + encode/decode → **commit**
4. `encoding.rs` — Encoding type → **commit**
5. `stream.rs` — EncodeStream, DecodeStream → **commit**
6. Tests (alongside each module) → **commit per test file**
7. `main.rs` — CLI → **commit**
8. `benches/bench_comparison.rs` — Criterion benchmarks → **commit**
9. `src/bench_compare.rs` — Standalone comparison binary → **commit**
10. Final integration testing and polish → **commit**

## Quality Criteria

- `cargo build` succeeds
- `cargo test` passes all tests
- `cargo clippy` has no warnings
- `cargo bench` runs all benchmark groups
- Safe Rust API wrapping unsafe FFI (no unsafe in public API)
- Proper RAII (Drop impls for all C resources)
- Thread safety (Send + Sync on Tokenizer)
- Idiomatic Rust (Result types, iterators, builders, lifetimes)

## Completion

When ALL of the following are true, output `<promise>RUST BINDING COMPLETE</promise>`:
1. `cargo build` compiles successfully
2. `cargo test` — all tests pass
3. `cargo clippy` — no warnings
4. `cargo bench` — all benchmark groups run
5. Full API parity with Python binding (all methods listed above implemented)
6. CLI works for encode/decode/info subcommands
7. All test files ported and passing
8. Benchmark comparison binary works with formatted output
9. Multiple meaningful git commits exist in history
