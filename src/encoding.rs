/// Rich encoding result with token IDs, byte offsets, and type IDs.
#[derive(Debug, Clone)]
pub struct Encoding {
    /// Token IDs (i32 values).
    pub ids: Vec<i32>,
    /// Optional byte offsets mapping each token to original input bytes.
    /// Each element is (start, end) where end is exclusive.
    pub offsets: Option<Vec<(usize, usize)>>,
    /// Type/segment IDs (0 = sequence A, 1 = sequence B).
    pub type_ids: Vec<u8>,
}

impl Encoding {
    /// Returns the number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns true if there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl std::fmt::Display for Encoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Encoding(n_tokens={})", self.len())
    }
}
