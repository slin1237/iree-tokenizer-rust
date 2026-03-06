use crate::error::Result;

/// Placeholder for streaming encode state.
pub struct EncodeStream<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
    open: bool,
}

impl<'a> EncodeStream<'a> {
    pub fn feed(&mut self, _text: &str) -> Result<Vec<i32>> {
        todo!("EncodeStream::feed")
    }

    pub fn finalize(&mut self) -> Result<Vec<i32>> {
        todo!("EncodeStream::finalize")
    }

    pub fn is_open(&self) -> bool {
        self.open
    }
}

impl Drop for EncodeStream<'_> {
    fn drop(&mut self) {
        self.open = false;
    }
}

/// Placeholder for streaming decode state.
pub struct DecodeStream<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
    open: bool,
}

impl<'a> DecodeStream<'a> {
    pub fn feed(&mut self, _token_ids: &[i32]) -> Result<String> {
        todo!("DecodeStream::feed")
    }

    pub fn finalize(&mut self) -> Result<String> {
        todo!("DecodeStream::finalize")
    }

    pub fn is_open(&self) -> bool {
        self.open
    }
}

impl Drop for DecodeStream<'_> {
    fn drop(&mut self) {
        self.open = false;
    }
}
