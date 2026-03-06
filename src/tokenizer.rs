use std::{fmt, os::raw::c_char, path::Path};

use crate::{
    encoding::Encoding,
    error::{check_status, is_resource_exhausted, Error, Result},
    ffi,
    stream::{DecodeStream, EncodeStream},
};

/// Safe wrapper around an IREE tokenizer (`iree_tokenizer_t*`).
///
/// Thread-safe: the underlying C tokenizer is immutable after construction.
pub struct Tokenizer {
    ptr: *mut ffi::iree_tokenizer_t,
    /// Cached model type string (owned).
    model_type_cache: String,
}

// SAFETY: iree_tokenizer_t is immutable after construction.
unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::iree_tokenizer_free(self.ptr) };
        }
    }
}

impl Tokenizer {
    /// Returns the system allocator used by IREE.
    fn system_allocator() -> ffi::iree_allocator_t {
        ffi::iree_allocator_t {
            self_: std::ptr::null_mut(),
            ctl: Some(ffi::iree_allocator_libc_ctl),
        }
    }

    /// Wraps a raw tokenizer pointer. Caches the model type string.
    fn from_raw(ptr: *mut ffi::iree_tokenizer_t) -> Self {
        let model_type_cache = unsafe {
            let sv = ffi::iree_tokenizer_model_type_name(ptr);
            if sv.data.is_null() || sv.size == 0 {
                String::new()
            } else {
                let slice = std::slice::from_raw_parts(sv.data as *const u8, sv.size);
                String::from_utf8_lossy(slice).into_owned()
            }
        };
        Tokenizer {
            ptr,
            model_type_cache,
        }
    }

    // -----------------------------------------------------------------------
    // Construction: HuggingFace JSON
    // -----------------------------------------------------------------------

    /// Load a tokenizer from a HuggingFace `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|e| {
            Error::InvalidArgument(format!(
                "Cannot open file '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Self::from_bytes(&data)
    }

    /// Load a tokenizer from a JSON string.
    #[expect(clippy::should_implement_trait)]
    pub fn from_str(json: &str) -> Result<Self> {
        Self::from_bytes(json.as_bytes())
    }

    /// Load a tokenizer from raw JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let sv = ffi::iree_string_view_t {
            data: data.as_ptr() as *const c_char,
            size: data.len(),
        };
        let allocator = Self::system_allocator();
        let mut tokenizer: *mut ffi::iree_tokenizer_t = std::ptr::null_mut();
        let status =
            unsafe { ffi::iree_tokenizer_from_huggingface_json(sv, allocator, &mut tokenizer) };
        check_status(status)?;
        Ok(Self::from_raw(tokenizer))
    }

    // -----------------------------------------------------------------------
    // Construction: Tiktoken
    // -----------------------------------------------------------------------

    /// Load a tokenizer from a tiktoken file.
    pub fn from_tiktoken_file(path: impl AsRef<Path>, encoding: &str) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|e| {
            Error::InvalidArgument(format!(
                "Cannot open file '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Self::from_tiktoken_bytes(&data, encoding)
    }

    /// Load a tokenizer from tiktoken data as a string.
    pub fn from_tiktoken_str(data: &str, encoding: &str) -> Result<Self> {
        Self::from_tiktoken_bytes(data.as_bytes(), encoding)
    }

    /// Load a tokenizer from raw tiktoken data bytes.
    pub fn from_tiktoken_bytes(data: &[u8], encoding: &str) -> Result<Self> {
        let config = Self::tiktoken_config_by_name(encoding)?;
        let sv = ffi::iree_string_view_t {
            data: data.as_ptr() as *const c_char,
            size: data.len(),
        };
        let allocator = Self::system_allocator();
        let mut tokenizer: *mut ffi::iree_tokenizer_t = std::ptr::null_mut();
        let status =
            unsafe { ffi::iree_tokenizer_from_tiktoken(sv, config, allocator, &mut tokenizer) };
        check_status(status)?;
        Ok(Self::from_raw(tokenizer))
    }

    fn tiktoken_config_by_name(
        encoding: &str,
    ) -> Result<*const ffi::iree_tokenizer_tiktoken_config_t> {
        let sv = ffi::iree_string_view_t {
            data: encoding.as_ptr() as *const c_char,
            size: encoding.len(),
        };
        let config = unsafe { ffi::iree_tokenizer_tiktoken_config_by_name(sv) };
        if config.is_null() {
            return Err(Error::InvalidArgument(format!(
                "Unknown tiktoken encoding '{encoding}'. Supported: cl100k_base, o200k_base, \
                 o200k_harmony, r50k_base, gpt2, p50k_base, p50k_edit"
            )));
        }
        Ok(config)
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>> {
        let mut capacity = text.len() / 2 + 16;
        let flags = Self::encode_flags(add_special_tokens, false);
        let allocator = Self::system_allocator();
        let text_sv = Self::make_string_view(text);

        loop {
            let mut ids = vec![0i32; capacity];
            let output = ffi::iree_tokenizer_token_output_t {
                capacity,
                token_ids: ids.as_mut_ptr(),
                token_offsets: std::ptr::null_mut(),
                type_ids: std::ptr::null_mut(),
            };
            let mut token_count: usize = 0;
            let status = unsafe {
                ffi::iree_tokenizer_encode(
                    self.ptr,
                    text_sv,
                    flags,
                    output,
                    allocator,
                    &mut token_count,
                )
            };
            if is_resource_exhausted(status) {
                unsafe { ffi::iree_status_ignore(status) };
                capacity = text.len() + 64;
                continue;
            }
            check_status(status)?;
            ids.truncate(token_count);
            return Ok(ids);
        }
    }

    /// Encode text with rich output (IDs, optional offsets, type IDs).
    pub fn encode_rich(
        &self,
        text: &str,
        add_special_tokens: bool,
        track_offsets: bool,
    ) -> Result<Encoding> {
        let mut capacity = text.len() / 2 + 16;
        let flags = Self::encode_flags(add_special_tokens, track_offsets);
        let allocator = Self::system_allocator();
        let text_sv = Self::make_string_view(text);

        loop {
            let mut ids = vec![0i32; capacity];
            let mut offsets_buf = if track_offsets {
                vec![ffi::iree_tokenizer_offset_t { start: 0, end: 0 }; capacity]
            } else {
                Vec::new()
            };
            let mut type_ids_buf = vec![0u8; capacity];

            let output = ffi::iree_tokenizer_token_output_t {
                capacity,
                token_ids: ids.as_mut_ptr(),
                token_offsets: if track_offsets {
                    offsets_buf.as_mut_ptr()
                } else {
                    std::ptr::null_mut()
                },
                type_ids: type_ids_buf.as_mut_ptr(),
            };
            let mut token_count: usize = 0;
            let status = unsafe {
                ffi::iree_tokenizer_encode(
                    self.ptr,
                    text_sv,
                    flags,
                    output,
                    allocator,
                    &mut token_count,
                )
            };
            if is_resource_exhausted(status) {
                unsafe { ffi::iree_status_ignore(status) };
                capacity = text.len() + 64;
                continue;
            }
            check_status(status)?;
            ids.truncate(token_count);
            type_ids_buf.truncate(token_count);
            let offsets = if track_offsets {
                offsets_buf.truncate(token_count);
                Some(offsets_buf.iter().map(|o| (o.start, o.end)).collect())
            } else {
                None
            };
            return Ok(Encoding {
                ids,
                offsets,
                type_ids: type_ids_buf,
            });
        }
    }

    /// Encode multiple texts in a single batch call.
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<i32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let flags = Self::encode_flags(add_special_tokens, false);

        // Calculate state and transform buffer sizes.
        let mut state_size: usize = 0;
        let status =
            unsafe { ffi::iree_tokenizer_encode_state_calculate_size(self.ptr, &mut state_size) };
        check_status(status)?;

        let max_text_len = texts.iter().map(|t| t.len()).max().unwrap_or(0);
        let transform_size =
            unsafe { ffi::iree_tokenizer_transform_buffer_oneshot_size(max_text_len) };

        let mut state_storage = vec![0u8; state_size];
        let mut transform_buffer = vec![0u8; transform_size];
        let state_span = ffi::iree_byte_span_t {
            data: state_storage.as_mut_ptr(),
            data_length: state_storage.len(),
        };
        let transform_span = ffi::iree_byte_span_t {
            data: transform_buffer.as_mut_ptr(),
            data_length: transform_buffer.len(),
        };
        let offset_runs = ffi::iree_tokenizer_offset_run_list_t {
            capacity: 0,
            values: std::ptr::null_mut(),
        };

        // Allocate per-item output buffers.
        let mut capacities: Vec<usize> = texts.iter().map(|t| t.len() / 2 + 16).collect();
        let mut all_ids: Vec<Vec<i32>> = capacities.iter().map(|&c| vec![0i32; c]).collect();

        loop {
            let mut items: Vec<ffi::iree_tokenizer_encode_batch_item_t> = texts
                .iter()
                .enumerate()
                .map(|(i, text)| ffi::iree_tokenizer_encode_batch_item_t {
                    text: Self::make_string_view(text),
                    output: ffi::iree_tokenizer_token_output_t {
                        capacity: capacities[i],
                        token_ids: all_ids[i].as_mut_ptr(),
                        token_offsets: std::ptr::null_mut(),
                        type_ids: std::ptr::null_mut(),
                    },
                    out_token_count: 0,
                })
                .collect();

            let status = unsafe {
                ffi::iree_tokenizer_encode_batch(
                    self.ptr,
                    items.as_mut_ptr(),
                    items.len(),
                    flags,
                    state_span,
                    transform_span,
                    offset_runs,
                )
            };
            if is_resource_exhausted(status) {
                unsafe { ffi::iree_status_ignore(status) };
                // Double all capacities.
                for (i, text) in texts.iter().enumerate() {
                    capacities[i] = text.len() + 64;
                    all_ids[i] = vec![0i32; capacities[i]];
                }
                continue;
            }
            check_status(status)?;

            let mut results = Vec::with_capacity(texts.len());
            for (i, item) in items.iter().enumerate() {
                let mut v = std::mem::take(&mut all_ids[i]);
                v.truncate(item.out_token_count);
                results.push(v);
            }
            return Ok(results);
        }
    }

    /// Encode multiple texts returning a flat array of IDs and per-text lengths.
    pub fn encode_batch_flat(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<(Vec<i32>, Vec<u64>)> {
        let batch = self.encode_batch(texts, add_special_tokens)?;
        let lengths: Vec<u64> = batch.iter().map(|v| v.len() as u64).collect();
        let flat: Vec<i32> = batch.into_iter().flatten().collect();
        Ok((flat, lengths))
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    /// Decode token IDs to text.
    pub fn decode(&self, token_ids: &[i32], skip_special_tokens: bool) -> Result<String> {
        let mut capacity = token_ids.len() * 4 + 64;
        let flags = Self::decode_flags(skip_special_tokens);
        let allocator = Self::system_allocator();
        let token_list = ffi::iree_tokenizer_token_id_list_t {
            count: token_ids.len(),
            values: token_ids.as_ptr(),
        };

        loop {
            let mut buf = vec![0u8; capacity];
            let text_output = ffi::iree_mutable_string_view_t {
                data: buf.as_mut_ptr() as *mut c_char,
                size: capacity,
            };
            let mut text_length: usize = 0;
            let status = unsafe {
                ffi::iree_tokenizer_decode(
                    self.ptr,
                    token_list,
                    flags,
                    text_output,
                    allocator,
                    &mut text_length,
                )
            };
            if is_resource_exhausted(status) {
                unsafe { ffi::iree_status_ignore(status) };
                capacity *= 2;
                continue;
            }
            check_status(status)?;
            buf.truncate(text_length);
            return String::from_utf8(buf)
                .map_err(|e| Error::Internal(format!("invalid UTF-8 in decode output: {e}")));
        }
    }

    /// Decode multiple token sequences in a single batch call.
    pub fn decode_batch(
        &self,
        token_id_lists: &[&[i32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        if token_id_lists.is_empty() {
            return Ok(Vec::new());
        }
        let flags = Self::decode_flags(skip_special_tokens);

        // Calculate state size.
        let mut state_size: usize = 0;
        let status =
            unsafe { ffi::iree_tokenizer_decode_state_calculate_size(self.ptr, &mut state_size) };
        check_status(status)?;

        let mut state_storage = vec![0u8; state_size];
        let state_span = ffi::iree_byte_span_t {
            data: state_storage.as_mut_ptr(),
            data_length: state_storage.len(),
        };

        let mut capacities: Vec<usize> = token_id_lists.iter().map(|t| t.len() * 4 + 64).collect();
        let mut all_bufs: Vec<Vec<u8>> = capacities.iter().map(|&c| vec![0u8; c]).collect();

        loop {
            let mut items: Vec<ffi::iree_tokenizer_decode_batch_item_t> = token_id_lists
                .iter()
                .enumerate()
                .map(|(i, tokens)| ffi::iree_tokenizer_decode_batch_item_t {
                    tokens: ffi::iree_tokenizer_token_id_list_t {
                        count: tokens.len(),
                        values: tokens.as_ptr(),
                    },
                    text_output: ffi::iree_mutable_string_view_t {
                        data: all_bufs[i].as_mut_ptr() as *mut c_char,
                        size: capacities[i],
                    },
                    out_text_length: 0,
                })
                .collect();

            let status = unsafe {
                ffi::iree_tokenizer_decode_batch(
                    self.ptr,
                    items.as_mut_ptr(),
                    items.len(),
                    flags,
                    state_span,
                )
            };
            if is_resource_exhausted(status) {
                unsafe { ffi::iree_status_ignore(status) };
                for (i, tokens) in token_id_lists.iter().enumerate() {
                    capacities[i] = tokens.len() * 8 + 128;
                    all_bufs[i] = vec![0u8; capacities[i]];
                }
                continue;
            }
            check_status(status)?;

            let mut results = Vec::with_capacity(token_id_lists.len());
            for (i, item) in items.iter().enumerate() {
                let buf = &all_bufs[i][..item.out_text_length];
                let s = String::from_utf8(buf.to_vec())
                    .map_err(|e| Error::Internal(format!("invalid UTF-8 in decode output: {e}")))?;
                results.push(s);
            }
            return Ok(results);
        }
    }

    // -----------------------------------------------------------------------
    // Streaming
    // -----------------------------------------------------------------------

    /// Create a streaming encoder.
    pub fn encode_stream(&self, add_special_tokens: bool) -> Result<EncodeStream<'_>> {
        EncodeStream::new(self, add_special_tokens)
    }

    /// Create a streaming decoder.
    pub fn decode_stream(&self, skip_special_tokens: bool) -> Result<DecodeStream<'_>> {
        DecodeStream::new(self, skip_special_tokens)
    }

    // -----------------------------------------------------------------------
    // Vocabulary
    // -----------------------------------------------------------------------

    /// Returns the number of active tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        let vocab = unsafe { ffi::iree_tokenizer_vocab(self.ptr) };
        unsafe { ffi::iree_tokenizer_vocab_token_count(vocab) }
    }

    /// Returns the model type name (e.g., "BPE", "WordPiece", "Unigram").
    pub fn model_type(&self) -> &str {
        &self.model_type_cache
    }

    /// Look up a token string and return its ID, or None if not found.
    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        let vocab = unsafe { ffi::iree_tokenizer_vocab(self.ptr) };
        let sv = Self::make_string_view(token);
        let id = unsafe { ffi::iree_tokenizer_vocab_lookup(vocab, sv) };
        if id < 0 {
            None
        } else {
            Some(id)
        }
    }

    /// Look up a token ID and return its string, or None if out of range.
    pub fn id_to_token(&self, id: i32) -> Option<&str> {
        if id < 0 {
            return None;
        }
        let vocab = unsafe { ffi::iree_tokenizer_vocab(self.ptr) };
        let capacity = unsafe { ffi::iree_tokenizer_vocab_capacity(vocab) };
        if (id as usize) >= capacity {
            return None;
        }
        let sv = unsafe { ffi::iree_tokenizer_vocab_token_text(vocab, id) };
        if sv.data.is_null() || sv.size == 0 {
            return None;
        }
        // SAFETY: The vocab string data lives as long as the tokenizer.
        let slice = unsafe { std::slice::from_raw_parts(sv.data as *const u8, sv.size) };
        std::str::from_utf8(slice).ok()
    }

    // -----------------------------------------------------------------------
    // Special token IDs
    // -----------------------------------------------------------------------

    pub fn bos_token_id(&self) -> Option<i32> {
        self.special_id(|ids| ids.bos)
    }

    pub fn eos_token_id(&self) -> Option<i32> {
        self.special_id(|ids| ids.eos)
    }

    pub fn unk_token_id(&self) -> Option<i32> {
        self.special_id(|ids| ids.unk)
    }

    pub fn pad_token_id(&self) -> Option<i32> {
        self.special_id(|ids| ids.pad)
    }

    pub fn sep_token_id(&self) -> Option<i32> {
        self.special_id(|ids| ids.sep)
    }

    pub fn cls_token_id(&self) -> Option<i32> {
        self.special_id(|ids| ids.cls)
    }

    pub fn mask_token_id(&self) -> Option<i32> {
        self.special_id(|ids| ids.mask)
    }

    fn special_id(&self, f: impl Fn(&ffi::iree_tokenizer_special_ids_t) -> i32) -> Option<i32> {
        let vocab = unsafe { ffi::iree_tokenizer_vocab(self.ptr) };
        let ids = unsafe { ffi::iree_tokenizer_vocab_special_ids(vocab) };
        let id = f(&ids);
        if id < 0 {
            None
        } else {
            Some(id)
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    pub(crate) fn raw_ptr(&self) -> *const ffi::iree_tokenizer_t {
        self.ptr
    }

    fn make_string_view(s: &str) -> ffi::iree_string_view_t {
        ffi::iree_string_view_t {
            data: s.as_ptr() as *const c_char,
            size: s.len(),
        }
    }

    fn encode_flags(add_special_tokens: bool, track_offsets: bool) -> u32 {
        let mut flags =
            ffi::iree_tokenizer_encode_flag_bits_e::IREE_TOKENIZER_ENCODE_FLAG_NONE as u32;
        if add_special_tokens {
            flags |= ffi::iree_tokenizer_encode_flag_bits_e::IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS as u32;
        }
        if track_offsets {
            flags |=
                ffi::iree_tokenizer_encode_flag_bits_e::IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS
                    as u32;
        }
        flags
    }

    fn decode_flags(skip_special_tokens: bool) -> u32 {
        if skip_special_tokens {
            ffi::iree_tokenizer_decode_flag_bits_e::IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS
                as u32
        } else {
            ffi::iree_tokenizer_decode_flag_bits_e::IREE_TOKENIZER_DECODE_FLAG_NONE as u32
        }
    }
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("model_type", &self.model_type())
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

impl fmt::Display for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tokenizer(model_type='{}', vocab_size={})",
            self.model_type(),
            self.vocab_size()
        )
    }
}

/// Iterate over decoded text chunks from a token iterator.
pub fn decode_stream_iter<I>(
    tokenizer: &Tokenizer,
    token_iter: I,
    skip_special_tokens: bool,
) -> Result<impl Iterator<Item = Result<String>> + '_>
where
    I: IntoIterator<Item = i32> + 'static,
{
    let mut stream = tokenizer.decode_stream(skip_special_tokens)?;
    let mut iter = token_iter.into_iter();
    let mut finalized = false;

    Ok(std::iter::from_fn(move || {
        if finalized {
            return None;
        }
        // Try to get the next token.
        loop {
            match iter.next() {
                Some(token_id) => {
                    match stream.feed(&[token_id]) {
                        Ok(text) if !text.is_empty() => return Some(Ok(text)),
                        Ok(_) => continue, // No text yet, try next token.
                        Err(e) => return Some(Err(e)),
                    }
                }
                None => {
                    finalized = true;
                    match stream.finalize() {
                        Ok(text) if !text.is_empty() => return Some(Ok(text)),
                        Ok(_) => return None,
                        Err(e) => return Some(Err(e)),
                    }
                }
            }
        }
    }))
}
