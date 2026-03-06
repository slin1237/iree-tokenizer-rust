#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]
#![allow(unused_imports)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// iree_allocator_libc_ctl is the default system allocator implementation.
// It's defined in allocator_libc.c and linked into the IREE static libraries.
// Bindgen can't see it because it's not in a header, so we declare it manually.
unsafe extern "C" {
    pub fn iree_allocator_libc_ctl(
        self_: *mut ::std::os::raw::c_void,
        command: iree_allocator_command_t,
        params: *const ::std::os::raw::c_void,
        inout_ptr: *mut *mut ::std::os::raw::c_void,
    ) -> iree_status_t;
}

/// Returns the IREE transform buffer recommended size (Rust reimplementation
/// of the C inline function which bindgen cannot generate).
pub unsafe fn iree_tokenizer_transform_buffer_recommended_size(
    text_size: usize,
) -> usize {
    let max_size = IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE as usize;
    let expansion = IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR as usize;
    let min_size = IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE as usize;

    let expanded = if text_size <= max_size / expansion {
        text_size * expansion
    } else {
        max_size
    };
    let expanded = expanded.max(min_size);
    let size = expanded.next_power_of_two();
    size.min(max_size)
}

/// Returns the transform buffer size for one-shot encoding (Rust reimplementation).
pub unsafe fn iree_tokenizer_transform_buffer_oneshot_size(
    text_size: usize,
) -> usize {
    let expansion = IREE_TOKENIZER_TRANSFORM_BUFFER_EXPANSION_FACTOR as usize;
    let min_size = IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE as usize;

    let expanded = text_size.saturating_mul(expansion);
    let expanded = expanded.max(min_size);
    expanded.next_power_of_two()
}
