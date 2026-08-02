//! Compilation of caller-facing plane-wave coordinates.
//!
//! This module converts caller-supplied spectral and in-plane coordinates into
//! the canonical coordinate representation consumed by the numerical backend.
//!
//! Coordinate compilation proceeds in four stages:
//!
//! 1. validate the caller-facing values;
//! 2. convert the real-valued samples into the backend complex scalar type;
//! 3. seed any assigned independent coordinate variables into the selected jet
//!    representation;
//! 4. canonicalise the seeded values into vacuum and parallel angular
//!    wavenumbers in inverse centimetres.
//!
//! Seeding is deliberately performed before canonicalisation. Derivatives are
//! therefore taken with respect to the caller-facing coordinates, while the
//! jet algebra propagates the required Jacobian through unit conversions and
//! nonlinear coordinate transformations.
//!
//! The resulting canonical coordinates are accompanied by a
//! [`CoordinateContext`], which retains the caller-facing coordinate metadata
//! required to interpret results and convert derivatives back into the input
//! parameterisation.

mod error;
mod in_plane;
mod jet;
mod seed;
mod spectral;

pub(crate) use error::CoordinateCompileError;
pub(crate) use jet::CanonicalCoordinateJet;
use seed::seed_coordinate;

#[cfg(test)]
pub(crate) use spectral::SpectralInputError;

use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};

use crate::{
    ComplexScalar, Stack,
    input::{
        CanonicalCoordinates, CompileJet, CoordinateReference, Coordinates, InPlaneCoordinate,
        SpectralCoordinate,
        compile::{ProjectionConstraint, context::CoordinateContext, seed::SeedJet},
    },
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    parameter::{DerivativeMapping, Parameter},
};
use in_plane::{canonicalise_in_plane, validate_in_plane};
use spectral::{canonicalise_spectral, validate_spectral};

/// A caller-facing coordinate that may be assigned an independent derivative
/// slot.
///
/// Coordinate variables are seeded before conversion to canonical backend
/// coordinates. A derivative slot therefore represents differentiation with
/// respect to the caller-facing parameterisation rather than directly with
/// respect to vacuum or parallel angular wavenumber.
///
/// The precise physical meaning and units of each variable are supplied by
/// [`Coordinates`](crate::input::Coordinates).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoordinateVariable {
    /// The caller-facing spectral coordinate.
    Spectral,

    /// The caller-facing in-plane coordinate.
    InPlane,
}

/// Canonical plane-wave coordinates together with their caller-facing context.
///
/// `canonical` contains the jet-valued vacuum and parallel angular
/// wavenumbers used by the backend. `context` retains the original coordinate
/// parameterisation and sampled values needed by higher-level APIs to describe
/// results and transform derivatives.
#[derive(Clone, Debug)]
pub(crate) struct CompiledCoordinates<J, R, D>
where
    D: Dimension,
{
    compiled: CompiledCoordinateProblem<J>,
    context: CoordinateContext<R, D>,
}

impl<J, R, D: Dimension> CompiledCoordinates<J, R, D> {
    /// Construct compiled coordinates from their canonical representation and
    /// caller-facing context.
    pub(crate) fn new(
        compiled: CompiledCoordinateProblem<J>,
        context: CoordinateContext<R, D>,
    ) -> Self {
        Self { compiled, context }
    }

    /// Return the caller-facing coordinate context.
    pub(crate) fn context(&self) -> &CoordinateContext<R, D> {
        &self.context
    }

    /// Return the compiled jet-valued coordinate problem
    pub(crate) fn compiled(&self) -> &CompiledCoordinateProblem<J> {
        &self.compiled
    }

    /// Decompose the compiled coordinates into their canonical representation
    /// and caller-facing context.
    pub(crate) fn into_parts(self) -> (CompiledCoordinateProblem<J>, CoordinateContext<R, D>) {
        (self.compiled, self.context)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CompiledCoordinateProblem<J> {
    canonical: CanonicalCoordinates<J>,
    projection_constraint: ProjectionConstraint,
}

impl<J> CompiledCoordinateProblem<J> {
    pub(crate) fn into_parts(self) -> (CanonicalCoordinates<J>, ProjectionConstraint) {
        (self.canonical, self.projection_constraint)
    }
}

pub(crate) fn compile_coordinates<M, J, E>(
    metadata: Coordinates,
    spectral_values: &Array<J::Scalar, J::Dimension>,
    in_plane_values: &Array<J::Scalar, J::Dimension>,
    reference: CoordinateReference,
    stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
    mapping: &DerivativeMapping,
) -> Result<CompiledCoordinateProblem<J>, CoordinateCompileError<J::Scalar>>
where
    J: CompileJet<M, E>,
    J::Scalar: ComplexScalar,
    <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Copy,
    J::Dimension: Dimension,
    E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
{
    let assignment = CoordinateAssignment::new(mapping);

    let spectral = compile_spectral(
        spectral_values,
        metadata.spectral(),
        assignment.spectral_slot(),
    )?;

    let (incident_index, projection_constraint) = match (metadata.in_plane(), reference) {
        (InPlaneCoordinate::IncidentAngle(_), CoordinateReference::Intrinsic) => {
            return Err(CoordinateCompileError::MissingIncidentSide);
        }

        (InPlaneCoordinate::IncidentAngle(_), CoordinateReference::IncidentSide(side)) => {
            let material = stack.incident_exterior(side);

            let incident_index =
                ConstitutiveLift::refractive_index(material, spectral.vacuum_angular_wavenumber());

            (Some(incident_index), ProjectionConstraint::Fixed(side))
        }

        _ => (None, ProjectionConstraint::Free),
    };

    let parallel_angular_wavenumber = compile_in_plane(
        in_plane_values,
        metadata.in_plane(),
        spectral.vacuum_angular_wavenumber(),
        incident_index.as_ref(),
        assignment.in_plane_slot(),
    )?;

    let canonical = CanonicalCoordinates::new(spectral.into_inner(), parallel_angular_wavenumber);

    Ok(CompiledCoordinateProblem {
        canonical,
        projection_constraint,
    })
}

/// Canonical jet-valued spectral coordinate.
///
/// The contained value is the vacuum angular wavenumber in inverse
/// centimetres. Any assigned derivative slot remains associated with the
/// original caller-facing spectral coordinate.
#[derive(Clone, Debug)]
pub(crate) struct CanonicalSpectral<J> {
    vacuum_angular_wavenumber: J,
}

impl<J> CanonicalSpectral<J> {
    /// Construct a canonical spectral coordinate.
    pub(crate) fn new(vacuum_angular_wavenumber: J) -> Self {
        Self {
            vacuum_angular_wavenumber,
        }
    }

    /// Return the canonical vacuum angular wavenumber.
    pub(crate) fn vacuum_angular_wavenumber(&self) -> &J {
        &self.vacuum_angular_wavenumber
    }

    /// Consume the wrapper and return the canonical vacuum angular wavenumber.
    pub(crate) fn into_inner(self) -> J {
        self.vacuum_angular_wavenumber
    }
}

/// Compile caller-facing spectral samples into canonical jet coordinates.
///
/// The input samples are validated, converted to the backend complex scalar
/// type, seeded into `slot` when one is assigned, and transformed into vacuum
/// angular wavenumber in inverse centimetres.
///
/// Seeding occurs before canonicalisation, so derivatives represented by the
/// resulting jet are taken with respect to the caller-facing spectral
/// coordinate.
///
/// When `slot` is `None`, the spectral coordinate is compiled as a constant.
///
/// # Errors
///
/// Returns [`CoordinateCompileError::Spectral`] when the sampled values are
/// invalid.
///
/// Returns [`CoordinateCompileError::Seed`] with
/// [`CoordinateVariable::Spectral`] when the selected jet representation does
/// not support the requested derivative slot.
fn compile_spectral<J>(
    values: &Array<J::Scalar, J::Dimension>,
    coordinate: SpectralCoordinate,
    slot: Option<usize>,
) -> Result<CanonicalSpectral<J>, CoordinateCompileError<J::Scalar>>
where
    J: SeedJet + CanonicalCoordinateJet,
    J::Scalar: ComplexField + Copy,
    <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Copy,
    J::Dimension: Dimension,
{
    validate_spectral(values)?;

    // let sampled = super::complexify(values);

    let seeded =
        seed_coordinate(values.clone(), slot).map_err(|source| CoordinateCompileError::Seed {
            variable: CoordinateVariable::Spectral,
            source,
        })?;

    Ok(CanonicalSpectral {
        vacuum_angular_wavenumber: canonicalise_spectral(seeded, coordinate),
    })
}

/// Compile caller-facing in-plane samples into a canonical jet coordinate.
///
/// The input samples are validated, converted to the backend complex scalar
/// type, seeded into `slot` when one is assigned, and transformed into
/// parallel angular wavenumber in inverse centimetres.
///
/// `vacuum_angular_wavenumber` is the already-compiled canonical spectral
/// coordinate. Incident-angle inputs additionally require `incident_index`.
///
/// Seeding occurs before canonicalisation, so derivatives represented by the
/// result are taken with respect to the caller-facing in-plane coordinate.
///
/// When `slot` is `None`, the in-plane coordinate is compiled as a constant.
///
/// `incident_index` may itself carry spectral derivatives, allowing the
/// canonical parallel wavenumber to include incident-medium dispersion.
///
/// # Errors
///
/// Returns [`CoordinateCompileError::InPlane`] when the sampled values are
/// invalid or when an incident-angle coordinate is compiled without an
/// incident refractive index.
///
/// Returns [`CoordinateCompileError::Seed`] with
/// [`CoordinateVariable::InPlane`] when the selected jet representation does
/// not support the requested derivative slot.
fn compile_in_plane<J>(
    values: &Array<J::Scalar, J::Dimension>,
    coordinate: InPlaneCoordinate,
    vacuum_angular_wavenumber: &J,
    incident_index: Option<&J>,
    slot: Option<usize>,
) -> Result<J, CoordinateCompileError<J::Scalar>>
where
    J: SeedJet + CanonicalCoordinateJet,
    J::Scalar: ComplexField + Copy,
    <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Copy,
    J::Dimension: Dimension,
{
    validate_in_plane(values, coordinate)?;

    let seeded =
        seed_coordinate(values.clone(), slot).map_err(|source| CoordinateCompileError::Seed {
            variable: CoordinateVariable::InPlane,
            source,
        })?;

    Ok(canonicalise_in_plane(
        seeded,
        coordinate,
        vacuum_angular_wavenumber,
        incident_index,
    )?)
}

/// Coordinate-specific view over a parameter assignment.
///
/// This translates canonical coordinate variables into their assigned jet
/// slots without exposing coordinate compilation to layer parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoordinateAssignment<'a> {
    mapping: &'a DerivativeMapping,
}

impl<'a> CoordinateAssignment<'a> {
    pub(crate) const fn new(mapping: &'a DerivativeMapping) -> Self {
        Self { mapping }
    }

    pub(crate) fn spectral_slot(&self) -> Option<usize> {
        self.mapping.slot_for(Parameter::Spectral)
    }

    pub(crate) fn in_plane_slot(&self) -> Option<usize> {
        self.mapping.slot_for(Parameter::InPlane)
    }

    pub(crate) fn slot_for(&self, variable: CoordinateVariable) -> Option<usize> {
        match variable {
            CoordinateVariable::Spectral => self.spectral_slot(),

            CoordinateVariable::InPlane => self.in_plane_slot(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use ndarray::{Array, Ix0, arr0};
    use num_complex::Complex64;
    use tmm_units::{AngleUnit, InverseLengthUnit, LengthUnit};

    use super::*;

    use super::in_plane::{InPlaneCanonicalisationError, InPlaneInputError};
    use super::spectral::SpectralInputError;
    use crate::input::compile::seed::UnsupportedDerivativeSlot;

    fn c(real: f64) -> Complex64 {
        Complex64::new(real, 0.0)
    }

    #[derive(Clone, Debug, PartialEq)]
    struct TestJet {
        value: Array<Complex64, Ix0>,
        seeded_slot: Option<usize>,
    }

    impl crate::algebra::Jet for TestJet {
        type Scalar = Complex64;
        type Dimension = Ix0;
    }

    impl SeedJet for TestJet {
        const VARIABLE_SLOTS: usize = 2;

        fn constant(value: Array<Complex64, Ix0>) -> Self {
            Self {
                value,
                seeded_slot: None,
            }
        }

        fn variable(
            value: Array<Complex64, Ix0>,
            slot: usize,
        ) -> Result<Self, UnsupportedDerivativeSlot> {
            if slot > 1 {
                return Err(UnsupportedDerivativeSlot { slot, available: 2 });
            }

            Ok(Self {
                value,
                seeded_slot: Some(slot),
            })
        }
    }

    impl CanonicalCoordinateJet for TestJet {
        fn scale_real(mut self, factor: f64) -> Self {
            self.value.mapv_inplace(|value| value * factor);
            self
        }

        fn reciprocal(mut self) -> Self {
            self.value.mapv_inplace(|value| value.recip());
            self
        }

        fn sin(mut self) -> Self {
            self.value.mapv_inplace(|value| value.sin());
            self
        }

        fn multiply(mut self, rhs: Self) -> Self {
            self.value = &self.value * &rhs.value;
            self
        }
    }

    fn scalar_value(jet: &TestJet) -> Complex64 {
        jet.value[()]
    }

    mod spectral {
        use super::*;

        #[test]
        fn compiles_constant_when_no_slot_is_assigned() {
            let values = arr0(c(2.0));

            let compiled = compile_spectral::<TestJet>(
                &values,
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
                None,
            )
            .unwrap();

            let jet = compiled.vacuum_angular_wavenumber();

            assert_eq!(jet.seeded_slot, None);
            assert_eq!(scalar_value(jet), Complex64::new(2.0, 0.0),);
        }

        #[test]
        fn passes_assigned_slot_to_spectral_seeder() {
            let values = arr0(c(2.0));

            let compiled = compile_spectral::<TestJet>(
                &values,
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
                Some(1),
            )
            .unwrap();

            assert_eq!(compiled.vacuum_angular_wavenumber().seeded_slot, Some(1),);
        }

        #[test]
        fn attributes_seed_error_to_spectral_variable() {
            let values = arr0(c(2.0));

            let error = compile_spectral::<TestJet>(
                &values,
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
                Some(2),
            )
            .unwrap_err();

            assert!(matches!(
                error,
                CoordinateCompileError::Seed {
                    variable: CoordinateVariable::Spectral,
                    ..
                }
            ));
        }

        #[test]
        fn validates_before_attempting_to_seed() {
            let values = arr0(c(-1.0));

            let error = compile_spectral::<TestJet>(
                &values,
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
                Some(100),
            )
            .unwrap_err();

            assert!(matches!(
                error,
                CoordinateCompileError::Spectral(SpectralInputError::NonPositive { index: 0, .. })
            ));
        }

        #[test]
        fn seeds_before_canonicalising_wavelength() {
            let values = arr0(c(2.0));

            let compiled = compile_spectral::<TestJet>(
                &values,
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
                Some(0),
            )
            .unwrap();

            let jet = compiled.vacuum_angular_wavenumber();

            assert_eq!(jet.seeded_slot, Some(0));
            assert!((scalar_value(jet).re - PI).abs() <= 1.0e-12);
        }
    }

    mod in_plane {
        use super::*;

        fn vacuum_angular_wavenumber() -> TestJet {
            TestJet {
                value: arr0(Complex64::new(10.0, 0.0)),
                seeded_slot: None,
            }
        }

        #[test]
        fn compiles_constant_when_no_slot_is_assigned() {
            let values = arr0(c(1.5));
            let spectral = vacuum_angular_wavenumber();

            let compiled = compile_in_plane::<TestJet>(
                &values,
                InPlaneCoordinate::EffectiveIndex,
                &spectral,
                None,
                None,
            )
            .unwrap();

            assert_eq!(compiled.seeded_slot, None);
            assert_eq!(scalar_value(&compiled), Complex64::new(15.0, 0.0),);
        }

        #[test]
        fn passes_assigned_slot_to_in_plane_seeder() {
            let values = arr0(c(1.5));
            let spectral = vacuum_angular_wavenumber();

            let compiled = compile_in_plane::<TestJet>(
                &values,
                InPlaneCoordinate::EffectiveIndex,
                &spectral,
                None,
                Some(1),
            )
            .unwrap();

            assert_eq!(compiled.seeded_slot, Some(1));
        }

        #[test]
        fn attributes_seed_error_to_in_plane_variable() {
            let values = arr0(c(1.5));
            let spectral = vacuum_angular_wavenumber();

            let error = compile_in_plane::<TestJet>(
                &values,
                InPlaneCoordinate::EffectiveIndex,
                &spectral,
                None,
                Some(2),
            )
            .unwrap_err();

            assert!(matches!(
                error,
                CoordinateCompileError::Seed {
                    variable: CoordinateVariable::InPlane,
                    ..
                }
            ));
        }

        #[test]
        fn validates_before_attempting_to_seed() {
            let values = arr0(c(f64::NAN));
            let spectral = vacuum_angular_wavenumber();

            let error = compile_in_plane::<TestJet>(
                &values,
                InPlaneCoordinate::EffectiveIndex,
                &spectral,
                None,
                Some(100),
            )
            .unwrap_err();

            assert!(matches!(
                error,
                CoordinateCompileError::InPlane(
                    in_plane::InPlaneInputError::NonFinite {
                        index: 0,
                        value,
                    }
                ) if value.is_nan()
            ));
        }

        #[test]
        fn propagates_missing_incident_index_error() {
            let values = arr0(c(30.0));
            let spectral = vacuum_angular_wavenumber();

            let error = compile_in_plane::<TestJet>(
                &values,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &spectral,
                None,
                None,
            )
            .unwrap_err();

            assert!(matches!(
                error,
                CoordinateCompileError::InPlaneCanonicalisation(
                    InPlaneCanonicalisationError::MissingIncidentIndex
                )
            ));
        }
    }
}

#[cfg(test)]
mod full_coordinate_compilation_tests {
    use super::in_plane::InPlaneInputError;
    use super::spectral::SpectralInputError;
    use super::*;

    use crate::{
        ComplexPlane, Constant, IncidentSide, RealAxis,
        input::compile::{DerivativeMapping, ProjectionConstraint},
        stack::{Layer, Thickness},
    };

    use ndarray::{Array, Ix1, array};
    use num_complex::Complex64;
    use tmm_units::{AngleUnit, InverseLengthUnit};

    type C = Complex64;
    type D = Ix1;

    const TOLERANCE: f64 = 1.0e-12;

    // Replace this with the crate's zero-derivative jet.
    type TestJet = crate::algebra::Jet0<Array<C, D>>;

    fn intrinsic_metadata() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
        )
    }

    fn effective_index_metadata() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::EffectiveIndex,
        )
    }

    fn angle_metadata() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
        )
    }

    fn assert_complex_eq(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_array_eq<D: Dimension>(actual: &Array<C, D>, expected: &Array<C, D>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_eq(actual, expected);
        }
    }

    fn test_stack() -> Stack<Constant<f64>, f64> {
        Stack::new(
            Constant::dielectric(1.0),
            vec![
                Layer::new(Constant::dielectric(4.0), Thickness::nanometres(500.0)),
                Layer::new(Constant::dielectric(2.0), Thickness::micrometres(2.0)),
            ],
            Constant::dielectric(1.0),
        )
    }

    fn test_stack_with_constant_exterior_index(index: f64) -> Stack<Constant<f64>, f64> {
        Stack::new(
            Constant::dielectric(index * index),
            vec![
                Layer::new(Constant::dielectric(4.0), Thickness::nanometres(500.0)),
                Layer::new(Constant::dielectric(2.0), Thickness::micrometres(2.0)),
            ],
            Constant::dielectric(index * index),
        )
    }

    fn test_stack_with_exterior_indices(left: f64, right: f64) -> Stack<Constant<f64>, f64> {
        Stack::new(
            Constant::dielectric(left * left),
            vec![
                Layer::new(Constant::dielectric(4.0), Thickness::nanometres(500.0)),
                Layer::new(Constant::dielectric(2.0), Thickness::micrometres(2.0)),
            ],
            Constant::dielectric(right * right),
        )
    }

    fn test_asymmetric_stack() -> Stack<Constant<f64>, f64> {
        Stack::new(
            Constant::dielectric(1.0),
            vec![
                Layer::new(Constant::dielectric(4.0), Thickness::nanometres(500.0)),
                Layer::new(Constant::dielectric(2.0), Thickness::micrometres(2.0)),
            ],
            Constant::dielectric(6.0),
        )
    }

    #[test]
    fn intrinsic_parallel_wavenumber_compiles_without_reference() {
        let spectral = array![C::new(2.0, 0.0), C::new(3.0, 0.0),];

        let in_plane = array![C::new(0.5, 0.2), C::new(0.75, -0.1),];

        let stack = test_stack();

        let assignment = DerivativeMapping::none();

        let compiled = compile_coordinates::<_, TestJet, RealAxis>(
            intrinsic_metadata(),
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            &stack,
            &assignment,
        )
        .unwrap();

        let (coordinates, constraint) = compiled.into_parts();

        assert_eq!(constraint, ProjectionConstraint::Free);

        assert_complex_array_eq(coordinates.vacuum_angular_wavenumber().value(), &spectral);

        assert_complex_array_eq(coordinates.parallel_angular_wavenumber().value(), &in_plane);
    }

    #[test]
    fn incident_side_does_not_constrain_intrinsic_coordinates() {
        let spectral = array![C::new(2.0, 0.0)];
        let in_plane = array![C::new(0.5, 0.0)];

        let stack = test_asymmetric_stack();

        let assignment = DerivativeMapping::none();

        let compiled = compile_coordinates::<_, TestJet, RealAxis>(
            intrinsic_metadata(),
            &spectral,
            &in_plane,
            CoordinateReference::IncidentSide(IncidentSide::Right),
            &stack,
            &assignment,
        )
        .unwrap();

        let (coordinates, constraint) = compiled.into_parts();

        assert_eq!(constraint, ProjectionConstraint::Free);

        assert_complex_array_eq(coordinates.parallel_angular_wavenumber().value(), &in_plane);
    }

    #[test]
    fn effective_index_is_intrinsic() {
        let spectral = array![C::new(2.0, 0.5), C::new(3.0, -0.25),];

        let effective_index = array![C::new(1.5, 0.1), C::new(1.25, -0.2),];

        let stack = test_stack();

        let assignment = DerivativeMapping::none();

        let compiled = compile_coordinates::<_, TestJet, ComplexPlane>(
            effective_index_metadata(),
            &spectral,
            &effective_index,
            CoordinateReference::Intrinsic,
            &stack,
            &assignment,
        )
        .unwrap();

        let (coordinates, constraint) = compiled.into_parts();

        let expected = &spectral * &effective_index;

        assert_eq!(constraint, ProjectionConstraint::Free);

        assert_complex_array_eq(coordinates.parallel_angular_wavenumber().value(), &expected);
    }

    #[test]
    fn incident_angle_requires_incident_side() {
        let spectral = array![C::new(2.0, 0.0)];
        let angle = array![C::new(0.25, 0.0)];

        let stack = test_stack();

        let assignment = DerivativeMapping::none();

        let error = compile_coordinates::<_, TestJet, RealAxis>(
            angle_metadata(),
            &spectral,
            &angle,
            CoordinateReference::Intrinsic,
            &stack,
            &assignment,
        )
        .unwrap_err();

        assert!(matches!(error, CoordinateCompileError::MissingIncidentSide));
    }

    #[test]
    fn incident_angle_fixes_projection_side() {
        let spectral = array![C::new(2.0, 0.0)];
        let angle = array![C::new(0.25, 0.0)];

        let stack = test_stack_with_constant_exterior_index(1.5);

        let assignment = DerivativeMapping::none();

        let compiled = compile_coordinates::<_, TestJet, RealAxis>(
            angle_metadata(),
            &spectral,
            &angle,
            CoordinateReference::IncidentSide(IncidentSide::Left),
            &stack,
            &assignment,
        )
        .unwrap();

        let (_, constraint) = compiled.into_parts();

        assert_eq!(constraint, ProjectionConstraint::Fixed(IncidentSide::Left),);
    }

    #[test]
    fn incident_angle_uses_index_from_selected_side() {
        let spectral = array![C::new(2.0, 0.0)];
        let angle = array![C::new(0.25, 0.0)];

        let stack = test_stack_with_exterior_indices(1.5, 2.0);

        let assignment = DerivativeMapping::none();

        let left = compile_coordinates::<_, TestJet, RealAxis>(
            angle_metadata(),
            &spectral,
            &angle,
            CoordinateReference::IncidentSide(IncidentSide::Left),
            &stack,
            &assignment,
        )
        .unwrap();

        let right = compile_coordinates::<_, TestJet, RealAxis>(
            angle_metadata(),
            &spectral,
            &angle,
            CoordinateReference::IncidentSide(IncidentSide::Right),
            &stack,
            &assignment,
        )
        .unwrap();

        let (left_coordinates, _) = left.into_parts();
        let (right_coordinates, _) = right.into_parts();

        let expected_left = spectral[0] * C::new(1.5, 0.0) * angle[0].sin();

        let expected_right = spectral[0] * C::new(2.0, 0.0) * angle[0].sin();

        dbg!(
            &expected_left,
            &expected_right,
            &left_coordinates,
            &right_coordinates
        );

        assert_complex_eq(
            left_coordinates.parallel_angular_wavenumber().value()[0],
            expected_left,
        );

        assert_complex_eq(
            right_coordinates.parallel_angular_wavenumber().value()[0],
            expected_right,
        );

        assert_ne!(
            left_coordinates.parallel_angular_wavenumber().value()[0],
            right_coordinates.parallel_angular_wavenumber().value()[0],
        );
    }

    #[test]
    fn lower_level_coordinate_compiler_accepts_complex_angle() {
        let spectral = array![C::new(2.0, 0.1)];
        let angle = array![C::new(0.25, 0.5)];

        let stack = test_stack_with_constant_exterior_index(1.5);

        let assignment = DerivativeMapping::none();

        let compiled = compile_coordinates::<_, TestJet, ComplexPlane>(
            angle_metadata(),
            &spectral,
            &angle,
            CoordinateReference::IncidentSide(IncidentSide::Left),
            &stack,
            &assignment,
        )
        .unwrap();

        let (coordinates, constraint) = compiled.into_parts();

        let expected = spectral[0] * C::new(1.5, 0.0) * angle[0].sin();

        assert_complex_eq(
            coordinates.parallel_angular_wavenumber().value()[0],
            expected,
        );

        assert_eq!(constraint, ProjectionConstraint::Fixed(IncidentSide::Left),);
    }

    #[test]
    fn propagates_spectral_validation_error() {
        let spectral = array![C::new(0.0, 1.0)];
        let in_plane = array![C::new(0.5, 0.0)];

        let stack = test_stack();

        let assignment = DerivativeMapping::none();

        let error = compile_coordinates::<_, TestJet, ComplexPlane>(
            intrinsic_metadata(),
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            &stack,
            &assignment,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CoordinateCompileError::Spectral(SpectralInputError::NonPositive { index: 0, .. })
        ));
    }

    #[test]
    fn propagates_in_plane_validation_error() {
        let spectral = array![C::new(2.0, 0.0)];
        let in_plane = array![C::new(0.5, f64::INFINITY)];

        let stack = test_stack();

        let assignment = DerivativeMapping::none();

        let error = compile_coordinates::<_, TestJet, RealAxis>(
            intrinsic_metadata(),
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            &stack,
            &assignment,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CoordinateCompileError::InPlane(InPlaneInputError::NonFinite { index: 0, .. })
        ));
    }
}
