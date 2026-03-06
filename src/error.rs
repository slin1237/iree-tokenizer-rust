use std::fmt;

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
