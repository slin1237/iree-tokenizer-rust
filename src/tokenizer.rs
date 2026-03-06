use std::fmt;
use std::path::Path;

use crate::encoding::Encoding;
use crate::error::Result;
use crate::stream::{DecodeStream, EncodeStream};

/// Placeholder Tokenizer struct. Will wrap iree_tokenizer_t*.
pub struct Tokenizer {
    _placeholder: (),
}

impl Tokenizer {
    pub fn from_file(_path: impl AsRef<Path>) -> Result<Self> {
        todo!("Tokenizer::from_file")
    }

    pub fn from_str(_json: &str) -> Result<Self> {
        todo!("Tokenizer::from_str")
    }

    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        todo!("Tokenizer::from_bytes")
    }

    pub fn from_tiktoken_file(_path: impl AsRef<Path>, _encoding: &str) -> Result<Self> {
        todo!("Tokenizer::from_tiktoken_file")
    }

    pub fn from_tiktoken_str(_data: &str, _encoding: &str) -> Result<Self> {
        todo!("Tokenizer::from_tiktoken_str")
    }

    pub fn from_tiktoken_bytes(_data: &[u8], _encoding: &str) -> Result<Self> {
        todo!("Tokenizer::from_tiktoken_bytes")
    }

    pub fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<i32>> {
        todo!("Tokenizer::encode")
    }

    pub fn encode_rich(
        &self,
        _text: &str,
        _add_special_tokens: bool,
        _track_offsets: bool,
    ) -> Result<Encoding> {
        todo!("Tokenizer::encode_rich")
    }

    pub fn encode_batch(
        &self,
        _texts: &[&str],
        _add_special_tokens: bool,
    ) -> Result<Vec<Vec<i32>>> {
        todo!("Tokenizer::encode_batch")
    }

    pub fn encode_batch_flat(
        &self,
        _texts: &[&str],
        _add_special_tokens: bool,
    ) -> Result<(Vec<i32>, Vec<u64>)> {
        todo!("Tokenizer::encode_batch_flat")
    }

    pub fn decode(&self, _token_ids: &[i32], _skip_special_tokens: bool) -> Result<String> {
        todo!("Tokenizer::decode")
    }

    pub fn decode_batch(
        &self,
        _token_id_lists: &[&[i32]],
        _skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        todo!("Tokenizer::decode_batch")
    }

    pub fn encode_stream(&self, _add_special_tokens: bool) -> Result<EncodeStream<'_>> {
        todo!("Tokenizer::encode_stream")
    }

    pub fn decode_stream(&self, _skip_special_tokens: bool) -> Result<DecodeStream<'_>> {
        todo!("Tokenizer::decode_stream")
    }

    pub fn vocab_size(&self) -> usize {
        todo!("Tokenizer::vocab_size")
    }

    pub fn model_type(&self) -> &str {
        todo!("Tokenizer::model_type")
    }

    pub fn token_to_id(&self, _token: &str) -> Option<i32> {
        todo!("Tokenizer::token_to_id")
    }

    pub fn id_to_token(&self, _id: i32) -> Option<&str> {
        todo!("Tokenizer::id_to_token")
    }

    pub fn bos_token_id(&self) -> Option<i32> {
        todo!("Tokenizer::bos_token_id")
    }

    pub fn eos_token_id(&self) -> Option<i32> {
        todo!("Tokenizer::eos_token_id")
    }

    pub fn unk_token_id(&self) -> Option<i32> {
        todo!("Tokenizer::unk_token_id")
    }

    pub fn pad_token_id(&self) -> Option<i32> {
        todo!("Tokenizer::pad_token_id")
    }

    pub fn sep_token_id(&self) -> Option<i32> {
        todo!("Tokenizer::sep_token_id")
    }

    pub fn cls_token_id(&self) -> Option<i32> {
        todo!("Tokenizer::cls_token_id")
    }

    pub fn mask_token_id(&self) -> Option<i32> {
        todo!("Tokenizer::mask_token_id")
    }
}

impl fmt::Display for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tokenizer(placeholder)")
    }
}

/// Iterate over decoded text chunks from a token iterator.
pub fn decode_stream_iter<I>(
    _tokenizer: &Tokenizer,
    _token_iter: I,
    _skip_special_tokens: bool,
) -> Result<impl Iterator<Item = Result<String>> + '_>
where
    I: IntoIterator<Item = i32> + 'static,
{
    Ok(std::iter::empty())
}
