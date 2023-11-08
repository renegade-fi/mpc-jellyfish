#!/usr/bin/env bash
set -e
set -x

cargo-nono check --no-default-features --package jf-utils
cargo-nono check --no-default-features --package mpc-relation
cargo-nono check --no-default-features --package jf-primitives
cargo-nono check --no-default-features --package jf-plonk
