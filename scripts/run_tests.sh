#!/usr/bin/env bash
set -e

# We want the code to panic if there is an integer overflow
export RUSTFLAGS="-C overflow-checks=on"

cargo +nightly test --release -p jf-utils -- -Zunstable-options --report-time
cargo +nightly test --release -p mpc-plonk --lib --bins --all-features -- -Zunstable-options --report-time
cargo +nightly test --release -p jf-primitives --all-features -- -Zunstable-options --report-time # enable test-srs feature for gen_srs_for_testing
cargo +nightly test --release -p mpc-relation --all-features -- -Zunstable-options --report-time
