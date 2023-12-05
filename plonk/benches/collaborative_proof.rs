//! Benchmarks a collaborative proof and a singleprover proof on the same
//! circuit for baselining

use std::time::{Duration, Instant};

use ark_mpc::test_helpers::execute_mock_mpc;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mpc_plonk::{
    multiprover::proof_system::{
        test_helpers::{gen_multiprover_circuit_for_test, setup_snark, TestCurve},
        MultiproverPlonkKzgSnark,
    },
    proof_system::{snark_test_helpers::gen_circuit_for_test, structs::ProvingKey, PlonkKzgSnark},
    transcript::SolidityTranscript,
};
use mpc_relation::PlonkType;
use rand::thread_rng;
use tokio::runtime::Builder as RuntimeBuilder;

const CIRCUIT_SIZING_PARAM: usize = 100;

// -----------
// | Helpers |
// -----------

/// Setup a proving key for the benchmark circuit
fn setup_pk() -> ProvingKey<TestCurve> {
    // Build a circuit and setup the proving key
    let circuit = gen_circuit_for_test(
        CIRCUIT_SIZING_PARAM,
        0, // start_val
        PlonkType::TurboPlonk,
    )
    .unwrap();
    let (pk, _) = setup_snark(&circuit);

    pk
}

// --------------
// | Benchmarks |
// --------------

/// Benchmark a singleprover proof on a test circuit
fn bench_singleprover(c: &mut Criterion) {
    // Build a circuit to prove satisfaction for
    let pk = setup_pk();
    let circuit = gen_circuit_for_test(
        CIRCUIT_SIZING_PARAM,
        0, // start_val
        PlonkType::TurboPlonk,
    )
    .unwrap();

    let mut group = c.benchmark_group("singleprover");
    let id = BenchmarkId::new("prover-latency", CIRCUIT_SIZING_PARAM);
    group.bench_function(id, |b| {
        b.iter(|| {
            let mut rng = thread_rng();
            let res = PlonkKzgSnark::batch_prove::<_, _, SolidityTranscript>(
                &mut rng,
                &[&circuit],
                &[&pk],
            )
            .unwrap();
            black_box(res);
        })
    });
}

/// Benchmark a collaborative proof on a test circuit
fn bench_multiprover(c: &mut Criterion) {
    let runtime = RuntimeBuilder::new_multi_thread()
        .worker_threads(3)
        .enable_all()
        .build()
        .unwrap();

    let pk = setup_pk();

    let mut group = c.benchmark_group("multiprover");
    let id = BenchmarkId::new("prover-latency", CIRCUIT_SIZING_PARAM);
    group.bench_function(id, |b| {
        let mut b = b.to_async(&runtime);
        b.iter_custom(|n_iters| {
            let pk = pk.clone();
            async move {
                let mut total_time = Duration::from_millis(0);
                for _ in 0..n_iters {
                    let elapsed = multiprover_prove(&pk).await;
                    total_time += elapsed;
                }

                total_time
            }
        })
    });
}

/// Prove the test circuit in a multiprover setting using the given key
///
/// Return the latency excluding the MPC setup time
async fn multiprover_prove(pk: &ProvingKey<TestCurve>) -> Duration {
    let (elapsed1, elapsed2) = execute_mock_mpc(|fabric| {
        let pk = pk.clone();
        async move {
            let start = Instant::now();
            let circuit = gen_multiprover_circuit_for_test(
                CIRCUIT_SIZING_PARAM,
                0, // start val
                fabric.clone(),
            )
            .unwrap();

            let proof = MultiproverPlonkKzgSnark::prove(&circuit, &pk, fabric).unwrap();
            black_box(proof.open_authenticated().await.unwrap());

            start.elapsed()
        }
    })
    .await;

    Duration::max(elapsed1, elapsed2)
}

criterion_group! {
    name = prover_latency;
    config = Criterion::default().sample_size(10);
    targets = bench_singleprover, bench_multiprover,
}

#[cfg(not(feature = "stats"))]
criterion_main!(prover_latency);

#[cfg(feature = "stats")]
#[tokio::main]
async fn main() {
    let pk = setup_pk();
    let duration = multiprover_prove(&pk).await;

    // Let the fabric print its stats
    tokio::time::sleep(Duration::from_secs(1)).await;
    println!("\nTook: {duration:?}");
}
