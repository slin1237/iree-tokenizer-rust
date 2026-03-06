use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let iree_source_dir = env::var("IREE_SOURCE_DIR")
        .unwrap_or_else(|_| "/Users/simolin/opensource/iree".to_string());
    let iree_source = Path::new(&iree_source_dir);

    if !iree_source.join("runtime/src/iree/tokenizer/tokenizer.h").exists() {
        panic!(
            "IREE source not found at {}. Set IREE_SOURCE_DIR env var.",
            iree_source.display()
        );
    }

    // Build IREE tokenizer via cmake.
    let build_dir = out_dir.join("iree-build");
    std::fs::create_dir_all(&build_dir).unwrap();

    let cmake_status = Command::new("cmake")
        .args([
            "-S",
            iree_source.to_str().unwrap(),
            "-B",
            build_dir.to_str().unwrap(),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DIREE_BUILD_COMPILER=OFF",
            "-DIREE_BUILD_TESTS=OFF",
            "-DIREE_BUILD_SAMPLES=OFF",
            "-DIREE_ERROR_ON_MISSING_SUBMODULES=OFF",
            "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
            "-DIREE_ENABLE_THREADING=OFF",
            &format!(
                "-DCMAKE_INSTALL_PREFIX={}",
                out_dir.join("iree-install").display()
            ),
        ])
        .status()
        .expect("Failed to run cmake configure");

    if !cmake_status.success() {
        panic!("cmake configure failed");
    }

    // Build only the tokenizer targets we need.
    let targets = [
        "iree_tokenizer_tokenizer",
        "iree_tokenizer_format_huggingface_tokenizer_json",
        "iree_tokenizer_format_tiktoken_tiktoken",
    ];

    for target in &targets {
        let build_status = Command::new("cmake")
            .args([
                "--build",
                build_dir.to_str().unwrap(),
                "--target",
                target,
                "--config",
                "Release",
                "-j",
                &num_cpus().to_string(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("Failed to build target {target}: {e}"));

        if !build_status.success() {
            panic!("cmake build failed for target {target}");
        }
    }

    // Find and link all static libraries produced by the build.
    // IREE's cmake produces libraries in various subdirectories.
    link_iree_libs(&build_dir);

    // Generate FFI bindings with bindgen.
    let tokenizer_h = iree_source.join("runtime/src/iree/tokenizer/tokenizer.h");
    let types_h = iree_source.join("runtime/src/iree/tokenizer/types.h");
    let hf_json_h = iree_source.join(
        "runtime/src/iree/tokenizer/format/huggingface/tokenizer_json.h",
    );
    let tiktoken_h =
        iree_source.join("runtime/src/iree/tokenizer/format/tiktoken/tiktoken.h");
    let vocab_h = iree_source.join("runtime/src/iree/tokenizer/vocab/vocab.h");

    let bindings = bindgen::Builder::default()
        .header(tokenizer_h.to_str().unwrap())
        .header(hf_json_h.to_str().unwrap())
        .header(tiktoken_h.to_str().unwrap())
        .header(vocab_h.to_str().unwrap())
        .clang_arg(format!(
            "-I{}",
            iree_source.join("runtime/src").display()
        ))
        // Allow the tokenizer and base types we need.
        .allowlist_function("iree_tokenizer_.*")
        .allowlist_function("iree_allocator_system")
        .allowlist_function("iree_status_.*")
        .allowlist_type("iree_tokenizer_.*")
        .allowlist_type("iree_string_view_t")
        .allowlist_type("iree_mutable_string_view_t")
        .allowlist_type("iree_byte_span_t")
        .allowlist_type("iree_const_byte_span_t")
        .allowlist_type("iree_allocator_t")
        .allowlist_type("iree_status_code_e")
        .allowlist_var("IREE_STATUS_.*")
        .allowlist_var("IREE_TOKENIZER_.*")
        // Generate types needed for the enums.
        .rustified_enum("iree_status_code_e")
        .rustified_enum("iree_tokenizer_encode_flag_bits_e")
        .rustified_enum("iree_tokenizer_decode_flag_bits_e")
        // Layout tests cause issues with opaque types.
        .layout_tests(false)
        .generate_comments(true)
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Failed to write bindings");

    // Rerun triggers.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=IREE_SOURCE_DIR");
    println!("cargo:rerun-if-changed={}", tokenizer_h.display());
    println!("cargo:rerun-if-changed={}", types_h.display());
    println!("cargo:rerun-if-changed={}", hf_json_h.display());
    println!("cargo:rerun-if-changed={}", tiktoken_h.display());
    println!("cargo:rerun-if-changed={}", vocab_h.display());
}

/// Recursively find all .a files in the build directory and emit linker
/// directives. IREE's cmake produces many small static libraries across
/// the tree; we need to link all of them that were built.
fn link_iree_libs(build_dir: &Path) {
    let mut lib_dirs = std::collections::HashSet::new();
    for entry in walkdir(build_dir) {
        if let Some(ext) = entry.extension() {
            if ext == "a" {
                if let Some(parent) = entry.parent() {
                    if lib_dirs.insert(parent.to_path_buf()) {
                        println!(
                            "cargo:rustc-link-search=native={}",
                            parent.display()
                        );
                    }
                }
                // Extract library name from filename: libfoo.a -> foo
                if let Some(stem) = entry.file_stem() {
                    let stem = stem.to_str().unwrap();
                    if let Some(name) = stem.strip_prefix("lib") {
                        println!("cargo:rustc-link-lib=static={name}");
                    } else {
                        println!("cargo:rustc-link-lib=static={stem}");
                    }
                }
            }
        }
    }
    // Link C++ standard library (needed by IREE internals).
    println!("cargo:rustc-link-lib=c++");
}

/// Simple recursive directory walker.
fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(walkdir(&path));
            } else {
                results.push(path);
            }
        }
    }
    results
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
