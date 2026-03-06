use std::os::raw::c_char;

use crate::{
    error::{check_status, Error, Result},
    ffi,
    tokenizer::Tokenizer,
};

/// Streaming encoder. Feeds text chunks and produces token IDs incrementally.
pub struct EncodeStream<'a> {
    _tokenizer: &'a Tokenizer,
    state: *mut ffi::iree_tokenizer_encode_state_t,
    /// Owned buffers that must outlive the state.
    _state_storage: Vec<u8>,
    _transform_buffer: Vec<u8>,
    open: bool,
}

impl<'a> EncodeStream<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, add_special_tokens: bool) -> Result<Self> {
        let tok_ptr = tokenizer.raw_ptr();

        // Calculate state size.
        let mut state_size: usize = 0;
        let status =
            unsafe { ffi::iree_tokenizer_encode_state_calculate_size(tok_ptr, &mut state_size) };
        check_status(status)?;

        let mut state_storage = vec![0u8; state_size];
        let transform_size = unsafe { ffi::iree_tokenizer_transform_buffer_recommended_size(8192) };
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

        let mut flags =
            ffi::iree_tokenizer_encode_flag_bits_e::IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START
                as u32;
        if add_special_tokens {
            flags |= ffi::iree_tokenizer_encode_flag_bits_e::IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS as u32;
        }

        let mut state: *mut ffi::iree_tokenizer_encode_state_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_tokenizer_encode_state_initialize(
                tok_ptr,
                state_span,
                transform_span,
                offset_runs,
                flags,
                &mut state,
            )
        };
        check_status(status)?;

        Ok(EncodeStream {
            _tokenizer: tokenizer,
            state,
            _state_storage: state_storage,
            _transform_buffer: transform_buffer,
            open: true,
        })
    }

    /// Feed a text chunk and return any tokens produced so far.
    pub fn feed(&mut self, text: &str) -> Result<Vec<i32>> {
        if !self.open {
            return Err(Error::InvalidArgument("stream is closed".to_string()));
        }
        let mut all_ids = Vec::new();
        let batch_size = 1024;
        let mut ids_buf = vec![0i32; batch_size];

        let chunk = ffi::iree_string_view_t {
            data: text.as_ptr() as *const c_char,
            size: text.len(),
        };
        let mut offset = 0usize;

        while offset < text.len() {
            let remaining = ffi::iree_string_view_t {
                data: unsafe { chunk.data.add(offset) },
                size: chunk.size - offset,
            };
            let output = ffi::iree_tokenizer_token_output_t {
                capacity: batch_size,
                token_ids: ids_buf.as_mut_ptr(),
                token_offsets: std::ptr::null_mut(),
                type_ids: std::ptr::null_mut(),
            };
            let mut bytes_consumed: usize = 0;
            let mut token_count: usize = 0;
            let status = unsafe {
                ffi::iree_tokenizer_encode_state_feed(
                    self.state,
                    remaining,
                    output,
                    &mut bytes_consumed,
                    &mut token_count,
                )
            };
            check_status(status)?;
            if token_count > 0 {
                all_ids.extend_from_slice(&ids_buf[..token_count]);
            }
            if bytes_consumed == 0 && token_count == 0 {
                break;
            }
            offset += bytes_consumed;
        }

        Ok(all_ids)
    }

    /// Finalize the stream and return any remaining tokens.
    pub fn finalize(&mut self) -> Result<Vec<i32>> {
        if !self.open {
            return Err(Error::InvalidArgument("stream is closed".to_string()));
        }
        self.open = false;
        let batch_size = 256;
        let mut ids_buf = vec![0i32; batch_size];
        let output = ffi::iree_tokenizer_token_output_t {
            capacity: batch_size,
            token_ids: ids_buf.as_mut_ptr(),
            token_offsets: std::ptr::null_mut(),
            type_ids: std::ptr::null_mut(),
        };
        let mut token_count: usize = 0;
        let status = unsafe {
            ffi::iree_tokenizer_encode_state_finalize(self.state, output, &mut token_count)
        };
        check_status(status)?;
        ids_buf.truncate(token_count);
        Ok(ids_buf)
    }

    /// Returns true if the stream is still open for feeding.
    pub fn is_open(&self) -> bool {
        self.open
    }
}

impl Drop for EncodeStream<'_> {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { ffi::iree_tokenizer_encode_state_deinitialize(self.state) };
        }
    }
}

/// Streaming decoder. Feeds token IDs and produces text incrementally.
pub struct DecodeStream<'a> {
    _tokenizer: &'a Tokenizer,
    state: *mut ffi::iree_tokenizer_decode_state_t,
    /// Owned buffer that must outlive the state.
    _state_storage: Vec<u8>,
    open: bool,
}

impl<'a> DecodeStream<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, skip_special_tokens: bool) -> Result<Self> {
        let tok_ptr = tokenizer.raw_ptr();

        let mut state_size: usize = 0;
        let status =
            unsafe { ffi::iree_tokenizer_decode_state_calculate_size(tok_ptr, &mut state_size) };
        check_status(status)?;

        let mut state_storage = vec![0u8; state_size];
        let state_span = ffi::iree_byte_span_t {
            data: state_storage.as_mut_ptr(),
            data_length: state_storage.len(),
        };

        let flags = if skip_special_tokens {
            ffi::iree_tokenizer_decode_flag_bits_e::IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS
                as u32
        } else {
            ffi::iree_tokenizer_decode_flag_bits_e::IREE_TOKENIZER_DECODE_FLAG_NONE as u32
        };

        let mut state: *mut ffi::iree_tokenizer_decode_state_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_tokenizer_decode_state_initialize(tok_ptr, flags, state_span, &mut state)
        };
        check_status(status)?;

        Ok(DecodeStream {
            _tokenizer: tokenizer,
            state,
            _state_storage: state_storage,
            open: true,
        })
    }

    /// Feed token IDs and return any text produced so far.
    pub fn feed(&mut self, token_ids: &[i32]) -> Result<String> {
        if !self.open {
            return Err(Error::InvalidArgument("stream is closed".to_string()));
        }
        let mut all_text = String::new();
        let buf_size = ffi::IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE as usize;
        let mut text_buf = vec![0u8; buf_size];

        let tokens = ffi::iree_tokenizer_token_id_list_t {
            count: token_ids.len(),
            values: token_ids.as_ptr(),
        };
        let mut offset = 0usize;

        while offset < token_ids.len() {
            let remaining = ffi::iree_tokenizer_token_id_list_t {
                count: tokens.count - offset,
                values: unsafe { tokens.values.add(offset) },
            };
            let text_output = ffi::iree_mutable_string_view_t {
                data: text_buf.as_mut_ptr() as *mut c_char,
                size: buf_size,
            };
            let mut tokens_consumed: usize = 0;
            let mut text_length: usize = 0;
            let status = unsafe {
                ffi::iree_tokenizer_decode_state_feed(
                    self.state,
                    remaining,
                    text_output,
                    &mut tokens_consumed,
                    &mut text_length,
                )
            };
            check_status(status)?;
            if text_length > 0 {
                let chunk = std::str::from_utf8(&text_buf[..text_length])
                    .map_err(|e| Error::Internal(format!("invalid UTF-8: {e}")))?;
                all_text.push_str(chunk);
            }
            if tokens_consumed == 0 && text_length == 0 {
                break;
            }
            offset += tokens_consumed;
        }

        Ok(all_text)
    }

    /// Finalize the stream and return any remaining text.
    pub fn finalize(&mut self) -> Result<String> {
        if !self.open {
            return Err(Error::InvalidArgument("stream is closed".to_string()));
        }
        self.open = false;
        let buf_size = ffi::IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE as usize;
        let mut text_buf = vec![0u8; buf_size];
        let text_output = ffi::iree_mutable_string_view_t {
            data: text_buf.as_mut_ptr() as *mut c_char,
            size: buf_size,
        };
        let mut text_length: usize = 0;
        let status = unsafe {
            ffi::iree_tokenizer_decode_state_finalize(self.state, text_output, &mut text_length)
        };
        check_status(status)?;
        let text = std::str::from_utf8(&text_buf[..text_length])
            .map_err(|e| Error::Internal(format!("invalid UTF-8: {e}")))?;
        Ok(text.to_string())
    }

    /// Returns true if the stream is still open for feeding.
    pub fn is_open(&self) -> bool {
        self.open
    }
}

impl Drop for DecodeStream<'_> {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { ffi::iree_tokenizer_decode_state_deinitialize(self.state) };
        }
    }
}
