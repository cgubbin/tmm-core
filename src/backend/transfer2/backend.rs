//! Internal evaluation kernel for the isotropic 2×2 transfer backend.
//!
//! This module evaluates the native transfer matrix of a finite planar stack.
//! It is responsible for:
//!
//! - evaluating isotropic quantities once per finite layer;
//! - constructing each finite-layer transfer matrix;
//! - composing layer matrices in propagation order;
//! - propagating first- and second-order derivatives;
//! - transforming primitive squared-coordinate derivatives to requested
//!   linear coordinates.
//!
//! The semi-infinite exterior media do not contribute propagation matrices.
//! They are nevertheless evaluated once to construct the exterior context used
//! for amplitude, power, and outgoing-mode projections.
//!
//! If the finite layers are encountered as `L₁, L₂, …, Lₙ`, accumulation is:
//!
//! ```text
//! M = Lₙ … L₂ L₁
//! ```
//!
//! Value-only, first-order, and second-order calculations use distinct return
//! types so derivative arrays are allocated only when requested.

use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{
        ExteriorWavevectors, RunMode,
        isotropic::IsotropicLayerQuantities,
        transfer2::{
            Transfer2Entries, Transfer2Error, TransferStabilityCheck,
            entries::Transfer2ExteriorContext,
        },
    },
    input::{CanonicalCoordinates, CanonicalStack},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use super::{Transfer2, Transfer2Workspace};

impl Transfer2 {
    pub(crate) fn accumulate<J, E, M>(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        stack: &CanonicalStack<M, J>,
        polarisation: Polarisation,
        exterior: &ExteriorWavevectors<J>,
        request: RunMode,
    ) -> Result<Transfer2Workspace<J>, Transfer2Error>
    where
        J: ScalarAlgebra + ConstitutiveLift<E, M> + Clone,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Copy,
        J::Dimension: Dimension,
        E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
    {
        let context = Transfer2ExteriorContext::new(
            coordinates,
            stack.left_exterior(),
            stack.right_exterior(),
            exterior,
            polarisation,
        );

        let mut workspace = Transfer2Workspace::new(
            coordinates.vacuum_angular_wavenumber().value(),
            context,
            request,
            stack.layer_count(),
        );

        for (layer_index, layer) in stack.layers().iter().enumerate() {
            let quantities = IsotropicLayerQuantities::evaluate::<E, M>(
                layer.material(),
                coordinates,
                polarisation,
            );

            let layer_matrix = Transfer2Entries::from_layer(&quantities, layer.thickness_cm());

            if self.stability_check == TransferStabilityCheck::PerLayer {
                check_layer_matrix(&layer_matrix, layer_index)?;
            }

            workspace.append(layer_matrix, quantities, layer.thickness_cm().clone());

            if self.stability_check == TransferStabilityCheck::PerLayer {
                check_accumulation(workspace.entries(), layer_index)?;
            }
        }

        if self.stability_check == TransferStabilityCheck::Final {
            if let Some((entry, index)) = workspace.entries().first_non_finite() {
                return Err(Transfer2Error::NonFiniteFinalMatrix { entry, index });
            }
        }

        Ok(workspace)
    }
}

fn check_layer_matrix<A>(matrix: &Transfer2Entries<A>, layer: usize) -> Result<(), Transfer2Error>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexField,
    A::Dimension: Dimension,
{
    if let Some((entry, index)) = matrix.first_non_finite() {
        return Err(Transfer2Error::NonFiniteLayerMatrix {
            layer,
            entry,
            index,
        });
    }

    Ok(())
}

fn check_accumulation<A>(matrix: &Transfer2Entries<A>, layer: usize) -> Result<(), Transfer2Error>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexField,
    A::Dimension: Dimension,
{
    if let Some((entry, index)) = matrix.first_non_finite() {
        return Err(Transfer2Error::NonFiniteAccumulation {
            layer,
            entry,
            index,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::Ix0;

    use super::*;
    use crate::{
        RealAxis,
        algebra::{ArrayJet0, RealParameter},
        backend::RunMode,
        input::canonical::{CanonicalCoordinates, CanonicalLayer, CanonicalStack},
        test_support::{
            C, c,
            jet::zero_jet_from_value,
            materials::linear,
            stack::{empty_stack, single_layer_stack, two_layer_stack},
        },
    };

    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn coordinates() -> CanonicalCoordinates<A> {
        CanonicalCoordinates::new(zero_jet_from_value(c(2.0)), zero_jet_from_value(c(0.1)))
    }

    #[test]
    fn empty_stack_accumulates_identity() {
        let backend = Transfer2::new();

        let workspace = backend
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates(),
                &empty_stack(),
                Polarisation::TransverseElectric,
                &ExteriorWavevectors::new(
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        empty_stack().left_exterior(),
                        &coordinates(),
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        empty_stack().right_exterior(),
                        &coordinates(),
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                ),
                RunMode::ResponseOnly,
            )
            .unwrap();

        let identity =
            Transfer2Entries::<A>::identity_like(coordinates().vacuum_angular_wavenumber().value());

        assert_eq!(workspace.entries(), &identity);
        assert!(!workspace.retains_layers());
    }

    #[test]
    fn single_layer_accumulation_matches_layer_matrix() {
        let backend = Transfer2::new();
        let stack = single_layer_stack();
        let coordinates = coordinates();

        let workspace = backend
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                Polarisation::TransverseElectric,
                &ExteriorWavevectors::new(
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.left_exterior(),
                        &coordinates,
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.right_exterior(),
                        &coordinates,
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                ),
                RunMode::ResponseOnly,
            )
            .unwrap();

        let quantities = IsotropicLayerQuantities::real_axis(
            stack.layers()[0].material(),
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let expected = Transfer2Entries::from_layer(&quantities, stack.layers()[0].thickness_cm());

        assert_eq!(workspace.entries(), &expected);
    }

    #[test]
    fn two_layer_accumulation_preserves_stack_order() {
        let backend = Transfer2::new();
        let stack = two_layer_stack();
        let coordinates = coordinates();

        let workspace = backend
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                Polarisation::TransverseElectric,
                &ExteriorWavevectors::new(
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.left_exterior(),
                        &coordinates,
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.right_exterior(),
                        &coordinates,
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                ),
                RunMode::ResponseOnly,
            )
            .unwrap();

        let first_quantities = IsotropicLayerQuantities::real_axis(
            stack.layers()[0].material(),
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let second_quantities = IsotropicLayerQuantities::real_axis(
            stack.layers()[1].material(),
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let first =
            Transfer2Entries::from_layer(&first_quantities, stack.layers()[0].thickness_cm());

        let second =
            Transfer2Entries::from_layer(&second_quantities, stack.layers()[1].thickness_cm());

        assert_eq!(workspace.entries(), &first.multiply(&second),);
    }

    #[test]
    fn response_only_does_not_retain_layers() {
        let workspace = Transfer2::new()
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates(),
                &two_layer_stack(),
                Polarisation::TransverseElectric,
                &ExteriorWavevectors::new(
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        two_layer_stack().left_exterior(),
                        &coordinates(),
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        two_layer_stack().right_exterior(),
                        &coordinates(),
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                ),
                RunMode::ResponseOnly,
            )
            .unwrap();

        assert!(!workspace.retains_layers());
    }

    #[test]
    fn internal_fields_retains_every_layer() {
        let workspace = Transfer2::new()
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates(),
                &two_layer_stack(),
                Polarisation::TransverseElectric,
                &ExteriorWavevectors::new(
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        two_layer_stack().left_exterior(),
                        &coordinates(),
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        two_layer_stack().right_exterior(),
                        &coordinates(),
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                ),
                RunMode::InternalFields,
            )
            .unwrap();

        assert!(workspace.retains_layers());

        assert_eq!(workspace.retained().unwrap().len(), 2,);
    }

    #[test]
    fn te_and_tm_produce_different_matrices() {
        let backend = Transfer2::new();
        let stack = single_layer_stack();

        let coordinates = coordinates();
        let left_exterior = stack.left_exterior();
        let right_exterior = stack.right_exterior();

        let exterior = ExteriorWavevectors::new(
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                left_exterior,
                &coordinates,
                Polarisation::TransverseMagnetic,
            )
            .kappa()
            .clone(),
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                right_exterior,
                &coordinates,
                Polarisation::TransverseMagnetic,
            )
            .kappa()
            .clone(),
        );

        let te = backend
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                Polarisation::TransverseElectric,
                &exterior,
                RunMode::ResponseOnly,
            )
            .unwrap();

        let tm = backend
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                Polarisation::TransverseMagnetic,
                &exterior,
                RunMode::ResponseOnly,
            )
            .unwrap();

        assert_ne!(te.entries(), tm.entries());
    }

    #[test]
    fn solve_and_retain_have_identical_external_solution() {
        let backend = Transfer2::new();

        let stack = two_layer_stack();
        let coordinates = coordinates();
        let left_exterior = stack.left_exterior();
        let right_exterior = stack.right_exterior();
        let polarisation = Polarisation::TransverseElectric;

        let exterior = ExteriorWavevectors::new(
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                left_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                right_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
        );

        let response = backend
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                polarisation,
                &exterior,
                RunMode::ResponseOnly,
            )
            .unwrap();

        let retained = backend
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                polarisation,
                &exterior,
                RunMode::InternalFields,
            )
            .unwrap();

        assert_eq!(response.entries(), retained.entries(),);
    }

    #[test]
    fn dispersive_material_is_evaluated_at_coordinates() {
        let material = linear(2.0, 0.3, 1.0, 0.1);

        let stack = CanonicalStack::new(
            material.clone(),
            material.clone(),
            vec![CanonicalLayer::new(material, zero_jet_from_value(c(0.1)))],
        );

        let polarisation = Polarisation::TransverseElectric;

        let first_coordinates =
            CanonicalCoordinates::new(zero_jet_from_value(c(2.0)), zero_jet_from_value(c(0.1)));

        let first = Transfer2::new()
            .accumulate::<_, crate::domain::RealAxis, _>(
                &first_coordinates,
                &stack,
                polarisation,
                &ExteriorWavevectors::new(
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.left_exterior(),
                        &first_coordinates,
                        polarisation,
                    )
                    .kappa()
                    .clone(),
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.right_exterior(),
                        &first_coordinates,
                        polarisation,
                    )
                    .kappa()
                    .clone(),
                ),
                RunMode::ResponseOnly,
            )
            .unwrap();

        let second_coordinates =
            CanonicalCoordinates::new(zero_jet_from_value(c(3.0)), zero_jet_from_value(c(0.1)));

        let second = Transfer2::new()
            .accumulate::<_, crate::domain::RealAxis, _>(
                &second_coordinates,
                &stack,
                polarisation,
                &ExteriorWavevectors::new(
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.left_exterior(),
                        &second_coordinates,
                        polarisation,
                    )
                    .kappa()
                    .clone(),
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.right_exterior(),
                        &second_coordinates,
                        polarisation,
                    )
                    .kappa()
                    .clone(),
                ),
                RunMode::ResponseOnly,
            )
            .unwrap();

        assert_ne!(first.entries(), second.entries());
    }
}

#[cfg(test)]
mod stability_tests {
    use ndarray::{Array, Ix0, Ix1, array};

    use super::*;
    use crate::{
        Constant, RealAxis,
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::transfer2::error::Transfer2Entry,
        input::canonical::{CanonicalCoordinates, CanonicalLayer, CanonicalStack},
        test_support::{
            C, c,
            jet::{zero_jet_from_array, zero_jet_from_value},
        },
    };

    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn coordinates() -> CanonicalCoordinates<A> {
        CanonicalCoordinates::new(zero_jet_from_value(c(1.0)), zero_jet_from_value(c(0.0)))
    }

    fn stable_stack() -> CanonicalStack<Constant<f64>, A> {
        CanonicalStack::new(
            Constant::new(1.0, 1.0),
            Constant::new(1.0, 1.0),
            vec![CanonicalLayer::new(
                Constant::new(4.0, 1.0),
                zero_jet_from_value(c(0.1)),
            )],
        )
    }

    fn directly_overflowing_stack() -> CanonicalStack<Constant<f64>, A> {
        CanonicalStack::new(
            Constant::new(1.0, 1.0),
            Constant::new(1.0, 1.0),
            vec![CanonicalLayer::new(
                Constant::new(-1.0e6, 1.0),
                zero_jet_from_value(c(1.0)),
            )],
        )
    }

    #[test]
    fn default_policy_is_final() {
        let backend = Transfer2::new();

        assert_eq!(backend.stability_check(), TransferStabilityCheck::Final,);
    }

    #[test]
    fn stable_stack_passes_all_check_policies() {
        let stack = stable_stack();
        let coordinates = coordinates();
        let left_exterior = stack.left_exterior();
        let right_exterior = stack.right_exterior();
        let polarisation = Polarisation::TransverseElectric;

        let exterior = ExteriorWavevectors::new(
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                left_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                right_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
        );

        for policy in [
            TransferStabilityCheck::Final,
            TransferStabilityCheck::PerLayer,
            TransferStabilityCheck::Disabled,
        ] {
            let result = Transfer2::with_stability_check(policy)
                .accumulate::<_, crate::domain::RealAxis, _>(
                    &coordinates,
                    &stack,
                    polarisation,
                    &exterior,
                    RunMode::ResponseOnly,
                );

            assert!(
                result.is_ok(),
                "stable stack failed under {policy:?}: \
                 {result:?}",
            );
        }
    }

    #[test]
    fn per_layer_check_reports_non_finite_layer_matrix() {
        let stack = directly_overflowing_stack();
        let coordinates = coordinates();
        let left_exterior = stack.left_exterior();
        let right_exterior = stack.right_exterior();
        let polarisation = Polarisation::TransverseElectric;

        let exterior = ExteriorWavevectors::new(
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                left_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                right_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
        );

        let error = Transfer2::with_stability_check(TransferStabilityCheck::PerLayer)
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                polarisation,
                &exterior,
                RunMode::ResponseOnly,
            )
            .expect_err("overflowing layer should be rejected");

        assert!(matches!(
            error,
            Transfer2Error::NonFiniteLayerMatrix { layer: 0, .. }
        ));
    }

    #[test]
    fn final_check_reports_non_finite_final_matrix() {
        let stack = directly_overflowing_stack();
        let coordinates = coordinates();
        let left_exterior = stack.left_exterior();
        let right_exterior = stack.right_exterior();
        let polarisation = Polarisation::TransverseElectric;

        let exterior = ExteriorWavevectors::new(
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                left_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                right_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
        );

        let error = Transfer2::with_stability_check(TransferStabilityCheck::Final)
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                polarisation,
                &exterior,
                RunMode::ResponseOnly,
            )
            .expect_err("non-finite final matrix should be rejected");

        assert!(matches!(error, Transfer2Error::NonFiniteFinalMatrix { .. }));
    }

    #[test]
    fn disabled_check_returns_non_finite_workspace() {
        let stack = directly_overflowing_stack();
        let coordinates = coordinates();
        let left_exterior = stack.left_exterior();
        let right_exterior = stack.right_exterior();
        let polarisation = Polarisation::TransverseElectric;

        let exterior = ExteriorWavevectors::new(
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                left_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                right_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
        );

        let workspace = Transfer2::with_stability_check(TransferStabilityCheck::Disabled)
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                polarisation,
                &exterior,
                RunMode::ResponseOnly,
            )
            .expect("disabled checks should not reject the matrix");

        assert!(workspace.entries().first_non_finite().is_some(),);
    }

    #[test]
    fn final_and_per_layer_checks_return_different_diagnostics() {
        let stack = directly_overflowing_stack();
        let coordinates = coordinates();
        let left_exterior = stack.left_exterior();
        let right_exterior = stack.right_exterior();
        let polarisation = Polarisation::TransverseElectric;

        let exterior = ExteriorWavevectors::new(
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                left_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
            IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                right_exterior,
                &coordinates,
                polarisation,
            )
            .kappa()
            .clone(),
        );

        let per_layer = Transfer2::with_stability_check(TransferStabilityCheck::PerLayer)
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                polarisation,
                &exterior,
                RunMode::ResponseOnly,
            )
            .unwrap_err();

        let final_check = Transfer2::with_stability_check(TransferStabilityCheck::Final)
            .accumulate::<_, crate::domain::RealAxis, _>(
                &coordinates,
                &stack,
                polarisation,
                &exterior,
                RunMode::ResponseOnly,
            )
            .unwrap_err();

        assert!(matches!(
            per_layer,
            Transfer2Error::NonFiniteLayerMatrix { layer: 0, .. }
        ));

        assert!(matches!(
            final_check,
            Transfer2Error::NonFiniteFinalMatrix { .. }
        ));
    }

    fn finite_entries() -> Transfer2Entries<Jet0<Array<C, Ix1>, RealParameter>> {
        Transfer2Entries::new(
            zero_jet_from_array(array![c(1.0), c(2.0), c(3.0)]),
            zero_jet_from_array(array![c(4.0), c(5.0), c(6.0)]),
            zero_jet_from_array(array![c(7.0), c(8.0), c(9.0)]),
            zero_jet_from_array(array![c(10.0), c(11.0), c(12.0)]),
        )
    }

    #[test]
    fn layer_check_accepts_finite_matrix() {
        let result = check_layer_matrix(&finite_entries(), 2);

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn accumulation_check_accepts_finite_matrix() {
        let result = check_accumulation(&finite_entries(), 2);

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn layer_check_reports_layer_index_entry_and_sample_index() {
        let entries = Transfer2Entries::new(
            zero_jet_from_array(array![c(1.0), c(2.0), c(3.0)]),
            zero_jet_from_array(array![c(4.0), C::new(f64::INFINITY, 0.0), c(6.0)]),
            zero_jet_from_array(array![c(7.0), c(8.0), c(9.0)]),
            zero_jet_from_array(array![c(10.0), c(11.0), c(12.0)]),
        );

        let error =
            check_layer_matrix(&entries, 4).expect_err("non-finite layer entry should be rejected");

        assert_eq!(
            error,
            Transfer2Error::NonFiniteLayerMatrix {
                layer: 4,
                entry: Transfer2Entry::M12,
                index: vec![1],
            },
        );
    }

    #[test]
    fn accumulation_check_reports_layer_index_entry_and_sample_index() {
        let entries = Transfer2Entries::new(
            zero_jet_from_array(array![c(1.0), c(2.0), c(3.0)]),
            zero_jet_from_array(array![c(4.0), c(5.0), c(6.0)]),
            zero_jet_from_array(array![c(7.0), c(8.0), c(9.0)]),
            zero_jet_from_array(array![c(10.0), c(11.0), C::new(0.0, f64::NEG_INFINITY)]),
        );

        let error = check_accumulation(&entries, 7)
            .expect_err("non-finite accumulated entry should be rejected");

        assert_eq!(
            error,
            Transfer2Error::NonFiniteAccumulation {
                layer: 7,
                entry: Transfer2Entry::M22,
                index: vec![2],
            },
        );
    }

    #[test]
    fn layer_check_reports_nan() {
        let entries = Transfer2Entries::new(
            zero_jet_from_array(array![c(1.0), C::new(f64::NAN, 0.0), c(3.0)]),
            zero_jet_from_array(array![c(4.0), c(5.0), c(6.0)]),
            zero_jet_from_array(array![c(7.0), c(8.0), c(9.0)]),
            zero_jet_from_array(array![c(10.0), c(11.0), c(12.0)]),
        );

        let error = check_layer_matrix(&entries, 0).unwrap_err();

        assert_eq!(
            error,
            Transfer2Error::NonFiniteLayerMatrix {
                layer: 0,
                entry: Transfer2Entry::M11,
                index: vec![1],
            },
        );
    }

    #[test]
    fn accumulation_check_reports_non_finite_imaginary_component() {
        let entries = Transfer2Entries::new(
            zero_jet_from_array(array![c(1.0), c(2.0), c(3.0)]),
            zero_jet_from_array(array![c(4.0), c(5.0), c(6.0)]),
            zero_jet_from_array(array![C::new(7.0, f64::INFINITY), c(8.0), c(9.0)]),
            zero_jet_from_array(array![c(10.0), c(11.0), c(12.0)]),
        );

        let error = check_accumulation(&entries, 3).unwrap_err();

        assert_eq!(
            error,
            Transfer2Error::NonFiniteAccumulation {
                layer: 3,
                entry: Transfer2Entry::M21,
                index: vec![0],
            },
        );
    }

    #[test]
    fn checks_report_first_entry_in_matrix_order() {
        let entries = Transfer2Entries::new(
            zero_jet_from_array(array![c(1.0), C::new(f64::NAN, 0.0), c(3.0)]),
            zero_jet_from_array(array![C::new(f64::INFINITY, 0.0), c(5.0), c(6.0)]),
            zero_jet_from_array(array![c(7.0), c(8.0), c(9.0)]),
            zero_jet_from_array(array![c(10.0), c(11.0), c(12.0)]),
        );

        let error = check_layer_matrix(&entries, 1).unwrap_err();

        assert_eq!(
            error,
            Transfer2Error::NonFiniteLayerMatrix {
                layer: 1,
                entry: Transfer2Entry::M11,
                index: vec![1],
            },
        );
    }

    #[test]
    fn checks_report_first_sample_within_entry() {
        let entries = Transfer2Entries::new(
            zero_jet_from_array(array![c(1.0), c(2.0), c(3.0)]),
            zero_jet_from_array(array![
                C::new(f64::NAN, 0.0),
                c(5.0),
                C::new(f64::INFINITY, 0.0)
            ]),
            zero_jet_from_array(array![c(7.0), c(8.0), c(9.0)]),
            zero_jet_from_array(array![c(10.0), c(11.0), c(12.0)]),
        );

        let error = check_accumulation(&entries, 5).unwrap_err();

        assert_eq!(
            error,
            Transfer2Error::NonFiniteAccumulation {
                layer: 5,
                entry: Transfer2Entry::M12,
                index: vec![0],
            },
        );
    }
}
