# iree-tokenizer Makefile
# Provides convenient shortcuts for common development tasks

# Auto-detect CPU cores and cap at reasonable limit
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
JOBS ?= $(shell echo $$(($(NPROC) > 16 ? 16 : $(NPROC))))

# Check if sccache is available and set RUSTC_WRAPPER accordingly
SCCACHE := $(shell which sccache 2>/dev/null)
ifdef SCCACHE
    export RUSTC_WRAPPER := $(SCCACHE)
    $(info Using sccache for compilation caching)
else
    $(info sccache not found. Install it for faster builds: cargo install sccache)
endif

.PHONY: help build test clean docs check fmt lint pre-commit bench submodule dev-setup \
        setup-sccache sccache-stats sccache-clean sccache-stop

help: ## Show this help message
	@echo "iree-tokenizer Development Commands"
	@echo "===================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

build: ## Build the project in release mode
	@echo "Building iree-tokenizer..."
	@cargo build --release

lint: fmt check ## Run formatting and clippy

test: lint ## Run all tests (lint first)
	@echo "Running tests..."
	@cargo test

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@cargo clean

docs: ## Generate and open documentation
	@echo "Generating documentation..."
	@cargo doc --open

check: ## Run cargo check and clippy
	@echo "Running cargo check..."
	@cargo check
	@echo "Running clippy..."
	@cargo clippy --all-targets -- -D warnings

fmt: ## Format code with nightly rustfmt
	@echo "Formatting code..."
	@cargo +nightly fmt --all

bench: ## Run benchmarks
	@echo "Running benchmarks..."
	@cargo bench

pre-commit: lint test ## Run pre-commit checks (lint + test)
	@echo "Pre-commit checks passed!"

submodule: ## Initialize/update the IREE submodule
	@echo "Initializing IREE submodule (shallow)..."
	@git submodule update --init --depth 1
	@cd third_party/iree && git submodule update --init --depth 1 third_party/flatcc

dev-setup: submodule build test ## Set up development environment
	@echo "Development environment ready!"

# sccache management targets
setup-sccache: ## Install and configure sccache
	@echo "Installing sccache..."
	@cargo install sccache
	@echo "sccache installed. Restart your shell or set RUSTC_WRAPPER=sccache"

sccache-stats: ## Show sccache statistics
	@if [ -n "$(SCCACHE)" ]; then \
		echo "sccache statistics:"; \
		sccache -s; \
	else \
		echo "sccache not installed. Run 'make setup-sccache' to install it."; \
	fi

sccache-clean: ## Clear sccache cache
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Clearing sccache cache..."; \
		sccache -C; \
		echo "sccache cache cleared"; \
	else \
		echo "sccache not installed"; \
	fi

sccache-stop: ## Stop the sccache server
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Stopping sccache server..."; \
		sccache --stop-server || true; \
	else \
		echo "sccache not installed"; \
	fi
