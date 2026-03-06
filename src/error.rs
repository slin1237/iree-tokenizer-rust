use std::fmt;

use crate::ffi;

/// Error types for IREE tokenizer operations.
#[derive(Debug, Clone)]
pub enum Error {
    /// Invalid argument (e.g., malformed JSON, bad file path).
    InvalidArgument(String),
    /// Resource not found (e.g., unknown token).
    NotFound(String),
    /// Feature not implemented.
    Unimplemented(String),
    /// Resource exhausted (e.g., output buffer too small).
    ResourceExhausted(String),
    /// Internal error.
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Error::NotFound(msg) => write!(f, "not found: {msg}"),
            Error::Unimplemented(msg) => write!(f, "unimplemented: {msg}"),
            Error::ResourceExhausted(msg) => write!(f, "resource exhausted: {msg}"),
            Error::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Check an iree_status_t and convert it to a Result.
/// Consumes the status (frees it if non-OK).
pub(crate) fn check_status(status: ffi::iree_status_t) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    // Extract the code and free the status.
    let code = unsafe { ffi::iree_status_consume_code(status) };
    let msg = format_status_message(code);
    match code {
        ffi::iree_status_code_e::IREE_STATUS_INVALID_ARGUMENT => {
            Err(Error::InvalidArgument(msg))
        }
        ffi::iree_status_code_e::IREE_STATUS_NOT_FOUND => Err(Error::NotFound(msg)),
        ffi::iree_status_code_e::IREE_STATUS_UNIMPLEMENTED => Err(Error::Unimplemented(msg)),
        ffi::iree_status_code_e::IREE_STATUS_RESOURCE_EXHAUSTED => {
            Err(Error::ResourceExhausted(msg))
        }
        _ => Err(Error::Internal(msg)),
    }
}

/// Returns true if the status represents RESOURCE_EXHAUSTED without consuming it.
pub(crate) fn is_resource_exhausted(status: ffi::iree_status_t) -> bool {
    if status.is_null() {
        return false;
    }
    let code = status as usize
        & ffi::iree_status_code_e::IREE_STATUS_CODE_MASK as usize;
    code == ffi::iree_status_code_e::IREE_STATUS_RESOURCE_EXHAUSTED as usize
}

fn format_status_message(code: ffi::iree_status_code_t) -> String {
    let c_str = unsafe { ffi::iree_status_code_string(code) };
    if c_str.is_null() {
        return format!("status code {}", code as u32);
    }
    let s = unsafe { std::ffi::CStr::from_ptr(c_str) };
    s.to_string_lossy().into_owned()
}
