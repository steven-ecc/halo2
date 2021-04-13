use ff::Field;

use crate::{
    arithmetic::CurveAffine,
    plonk::{Error, VerifyingKey},
    poly::multiopen::VerifierQuery,
    transcript::{
        read_n_points, read_n_scalars, ChallengeScalar, ChallengeScalarType, ChallengeSpace,
        TranscriptRead,
    },
};

use super::Argument;

pub struct Committed<C: CurveAffine> {
    h_commitments: Vec<C>,
}

pub struct Evaluated<C: CurveAffine> {
    h_commitments: Vec<C>,
    h_evals: Vec<C::Scalar>,
}

impl<C: CurveAffine> Argument<C> {
    pub(in crate::plonk) fn read_commitments<S: ChallengeSpace<C>, T: TranscriptRead<C, S>>(
        vk: &VerifyingKey<C>,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        // Obtain a commitment to h(X) in the form of multiple pieces of degree n - 1
        let h_commitments = read_n_points(transcript, vk.domain.get_quotient_poly_degree())
            .map_err(|_| Error::TranscriptError)?;

        Ok(Committed { h_commitments })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate<S: ChallengeSpace<C>, T: TranscriptRead<C, S>>(
        self,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let h_evals = read_n_scalars(transcript, self.h_commitments.len())
            .map_err(|_| Error::TranscriptError)?;

        Ok(Evaluated {
            h_commitments: self.h_commitments,
            h_evals,
        })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn verify(
        &self,
        expressions: impl Iterator<Item = C::Scalar>,
        y: ChallengeScalar<C>,
        xn: C::Scalar,
    ) -> Result<(), Error> {
        assert!(matches!(y.challenge_type(), ChallengeScalarType::Y));

        let expected_h_eval = expressions.fold(C::Scalar::zero(), |h_eval, v| h_eval * &*y + &v);

        // Compute h(x) from the prover
        let h_eval = self
            .h_evals
            .iter()
            .rev()
            .fold(C::Scalar::zero(), |acc, eval| acc * &xn + eval);

        // Did the prover commit to the correct polynomial?
        if expected_h_eval != (h_eval * &(xn - &C::Scalar::one())) {
            return Err(Error::ConstraintSystemFailure);
        }

        Ok(())
    }

    pub(in crate::plonk) fn queries(
        &self,
        x: ChallengeScalar<C>,
    ) -> impl Iterator<Item = VerifierQuery<'_, C>> + Clone {
        assert!(matches!(x.challenge_type(), ChallengeScalarType::X));

        self.h_commitments
            .iter()
            .zip(self.h_evals.iter())
            .map(move |(commitment, &eval)| VerifierQuery {
                point: *x,
                commitment,
                eval,
            })
    }
}
