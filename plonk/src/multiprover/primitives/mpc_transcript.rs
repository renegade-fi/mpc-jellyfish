//! The mpc transcript replicates the functionality of the `SolidityTranscript`
//! in an MPC fabric as defined by `ark-mpc`. This means that the inputs,
//! outputs, and state of the transcript are `Result`s in the mpc fabric: i.e.
//! they are handles on yet-to-be computed nodes in the MPC's computation graph

use std::sync::{Arc, Mutex};

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_mpc::{
    algebra::{Scalar, ScalarResult},
    MpcFabric, ResultId, ResultValue,
};

use crate::transcript::{PlonkTranscript, SolidityTranscript};

/// The MPC transcript, implements the same functionality as the
/// `SolidityTranscript` in an async manner evaluated by the MPC fabric's
/// executor
pub struct MpcTranscript<C: CurveGroup> {
    /// The base transcript that the MPC transcript effectively orders accesses
    /// to in a fabric's computation graph
    transcript: Arc<Mutex<SolidityTranscript>>,
    /// The latest op ID that has been submitted for the transcript. Each
    /// successive operation awaits the current value of this field; thus
    /// enqueuing itself behind the last operation. Across all operations,
    /// this gives a total ordering that keeps the transcript in sync between
    /// the parties of the MPC
    latest_op_id: ResultId,
    /// The underlying fabric the MPC runs on
    fabric: MpcFabric<C>,
}

impl<C: CurveGroup> MpcTranscript<C> {
    /// Constructor
    pub fn new(label: &'static [u8], fabric: MpcFabric<C>) -> Self {
        let base_transcript = Arc::new(Mutex::new(
            <SolidityTranscript as PlonkTranscript<()>>::new(label),
        ));
        Self {
            transcript: base_transcript,
            latest_op_id: ResultId::default(),
            fabric,
        }
    }

    /// Append a message to the transcript
    pub fn append_message(&mut self, label: &'static [u8], msg: &[u8]) {
        let transcript_ref = self.transcript.clone();
        let msg_clone = msg.to_vec();
        let res: ScalarResult<C> = self
            .fabric
            .new_gate_op(vec![self.latest_op_id], move |_args| {
                let mut locked_transcript = transcript_ref.lock().unwrap();
                <SolidityTranscript as PlonkTranscript<()>>::append_message(
                    &mut locked_transcript,
                    label,
                    &msg_clone,
                )
                .unwrap();

                // Return a dummy result, this is unused
                ResultValue::Scalar(Scalar::zero())
            });

        self.latest_op_id = res.op_ids()[0];
    }

    /// Generate a challenge for the current transcript, and append it
    ///
    /// Returns a result to the challenge
    pub fn get_and_append_challenge<E: Pairing<ScalarField = C::ScalarField>>(
        &mut self,
        label: &'static [u8],
    ) -> ScalarResult<C> {
        let transcript_ref = self.transcript.clone();
        let res: ScalarResult<C> = self
            .fabric
            .new_gate_op(vec![self.latest_op_id], move |_args| {
                let mut locked_transcript = transcript_ref.lock().unwrap();
                let challenge: C::ScalarField = locked_transcript
                    .get_and_append_challenge::<E>(label)
                    .unwrap();

                ResultValue::Scalar(Scalar::new(challenge))
            });

        self.latest_op_id = res.op_ids()[0];
        res
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_mpc::{algebra::Scalar, test_helpers::execute_mock_mpc};
    use itertools::Itertools;
    use rand::{thread_rng, CryptoRng, Rng, RngCore};

    use crate::{
        multiprover::primitives::MpcTranscript,
        transcript::{PlonkTranscript, SolidityTranscript},
    };

    /// Enumerates the operations on a transcript
    #[derive(Clone)]
    enum TranscriptOp {
        /// Append to transcript
        AppendMessage(&'static [u8], Vec<u8>),
        /// Squeeze a challenge
        GetAndAppendChallenge(&'static [u8]),
    }

    impl TranscriptOp {
        /// Get a random transcript operation
        fn random_op<R: RngCore + CryptoRng>(rng: &mut R) -> Self {
            let is_append = rng.gen_bool(0.5);

            if is_append {
                const N_BYTES: usize = 32;
                let mut random_bytes = vec![0u8; N_BYTES];
                rng.fill_bytes(&mut random_bytes);

                Self::AppendMessage(b"test", random_bytes)
            } else {
                Self::GetAndAppendChallenge(b"test")
            }
        }
    }

    /// Tests coherence on a random transcript between the MPC transcript and
    /// the base transcript
    #[tokio::test]
    async fn test_transcript_coherence() {
        const N_OPS: usize = 100;
        let mut rng = thread_rng();
        let ops = (0..N_OPS)
            .map(|_| TranscriptOp::random_op(&mut rng))
            .collect_vec();

        // Apply the ops to a base transcript
        let mut base_transcript = <SolidityTranscript as PlonkTranscript<()>>::new(b"test");
        for op in ops.iter() {
            match op {
                TranscriptOp::AppendMessage(label, msg) => {
                    <SolidityTranscript as PlonkTranscript<()>>::append_message(
                        &mut base_transcript,
                        label,
                        msg,
                    )
                    .unwrap();
                },
                TranscriptOp::GetAndAppendChallenge(label) => base_transcript
                    .get_and_append_challenge::<ark_bn254::Bn254>(label)
                    .map(|_| ())
                    .unwrap(),
            };
        }

        // Squeeze the expected result after the ops
        let base_res: <Bn254 as Pairing>::ScalarField = base_transcript
            .get_and_append_challenge::<ark_bn254::Bn254>(b"final")
            .unwrap();
        let expected_res = Scalar::new(base_res);

        let (res1, res2) = execute_mock_mpc(|fabric| {
            let ops = ops.clone();
            async move {
                // Build a transcript and apply the operations
                let mut transcript = MpcTranscript::new(b"test", fabric);
                for op in ops.iter() {
                    match op {
                        TranscriptOp::AppendMessage(label, msg) => {
                            transcript.append_message(label, msg);
                        },
                        TranscriptOp::GetAndAppendChallenge(label) => {
                            transcript
                                .get_and_append_challenge::<ark_bn254::Bn254>(label)
                                .await;
                        },
                    };
                }

                // Squeeze a final challenge
                transcript
                    .get_and_append_challenge::<ark_bn254::Bn254>(b"final")
                    .await
            }
        })
        .await;

        assert_eq!(res1, expected_res);
        assert_eq!(res2, expected_res);
    }
}
