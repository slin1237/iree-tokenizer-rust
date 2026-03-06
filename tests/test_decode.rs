use iree_tokenizer::Tokenizer;
use std::path::PathBuf;

fn bpe_tokenizer() -> Tokenizer {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata/bpe_bytelevel_minimal.json");
    Tokenizer::from_file(path).unwrap()
}

#[test]
fn test_decode_basic() {
    let tok = bpe_tokenizer();
    let ids = tok.encode("Hello world", false).unwrap();
    let text = tok.decode(&ids, false).unwrap();
    assert_eq!(text, "Hello world");
}

#[test]
fn test_decode_empty() {
    let tok = bpe_tokenizer();
    let text = tok.decode(&[], false).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_decode_roundtrip_unicode() {
    let tok = bpe_tokenizer();
    for text in &["Hello", "abc 123", "foo bar baz"] {
        let ids = tok.encode(text, false).unwrap();
        let decoded = tok.decode(&ids, false).unwrap();
        assert_eq!(&decoded, text);
    }
}
