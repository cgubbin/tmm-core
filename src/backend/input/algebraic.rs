use super::{PlanarInput, Polarisation};
use crate::algebra::{ArrayJet, ArrayJetBivariate, ArrayJetFirst};

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlanarSeed {
    VacuumWavenumber,
    ParallelWavenumber,
    Structural,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AlgebraicPlanarInput<I> {
    vacuum_wavenumber: I,
    parallel_wavenumber: I,
    polarisation: Polarisation,
}

impl<I> AlgebraicPlanarInput<I> {
    pub(crate) fn new(
        vacuum_wavenumber: I,
        parallel_wavenumber: I,
        polarisation: Polarisation,
    ) -> Self {
        Self {
            vacuum_wavenumber,
            parallel_wavenumber,
            polarisation,
        }
    }

    pub(crate) fn vacuum_wavenumber(&self) -> &I {
        &self.vacuum_wavenumber
    }

    pub(crate) fn parallel_wavenumber(&self) -> &I {
        &self.parallel_wavenumber
    }

    pub(crate) fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    pub(crate) fn into_parts(self) -> (I, I, Polarisation) {
        (
            self.vacuum_wavenumber,
            self.parallel_wavenumber,
            self.polarisation,
        )
    }
}

impl<C, D> AlgebraicPlanarInput<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexField,
    D: Dimension,
{
    pub(crate) fn values(planar: &PlanarInput<C, D>) -> Self {
        Self::new(
            planar.vacuum_wavenumber().clone(),
            planar.parallel_wavenumber().clone(),
            planar.polarisation(),
        )
    }
}

impl<C, D, P> AlgebraicPlanarInput<ArrayJetFirst<C, D, P>>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    pub(crate) fn first(planar: &PlanarInput<C, D>, seed: PlanarSeed) -> Self {
        let vacuum_wavenumber = match seed {
            PlanarSeed::VacuumWavenumber => {
                ArrayJetFirst::variable(planar.vacuum_wavenumber().clone())
            }
            PlanarSeed::ParallelWavenumber | PlanarSeed::Structural => {
                ArrayJetFirst::constant(planar.vacuum_wavenumber().clone())
            }
        };

        let parallel_wavenumber = match seed {
            PlanarSeed::ParallelWavenumber => {
                ArrayJetFirst::variable(planar.parallel_wavenumber().clone())
            }
            PlanarSeed::VacuumWavenumber | PlanarSeed::Structural => {
                ArrayJetFirst::constant(planar.parallel_wavenumber().clone())
            }
        };

        Self::new(
            vacuum_wavenumber,
            parallel_wavenumber,
            planar.polarisation(),
        )
    }
}

impl<C, D, P> AlgebraicPlanarInput<ArrayJet<C, D, P>>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    pub(crate) fn second(planar: &PlanarInput<C, D>, seed: PlanarSeed) -> Self {
        let vacuum_wavenumber = match seed {
            PlanarSeed::VacuumWavenumber => ArrayJet::variable(planar.vacuum_wavenumber().clone()),
            PlanarSeed::ParallelWavenumber | PlanarSeed::Structural => {
                ArrayJet::constant(planar.vacuum_wavenumber().clone())
            }
        };

        let parallel_wavenumber = match seed {
            PlanarSeed::ParallelWavenumber => {
                ArrayJet::variable(planar.parallel_wavenumber().clone())
            }
            PlanarSeed::VacuumWavenumber | PlanarSeed::Structural => {
                ArrayJet::constant(planar.parallel_wavenumber().clone())
            }
        };

        Self::new(
            vacuum_wavenumber,
            parallel_wavenumber,
            planar.polarisation(),
        )
    }
}

impl<C, D, P> AlgebraicPlanarInput<ArrayJetBivariate<C, D, P>>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    pub(crate) fn full_spectral(planar: &PlanarInput<C, D>) -> Self {
        Self::new(
            ArrayJetBivariate::variable_x(planar.vacuum_wavenumber().clone()),
            ArrayJetBivariate::variable_y(planar.parallel_wavenumber().clone()),
            planar.polarisation(),
        )
    }
}

#[cfg(test)]
mod test {
    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;

    use crate::{
        algebra::{ArrayJet, ArrayJetBivariate, ArrayJetFirst, RealParameter},
        backend::input::algebraic::PlanarSeed,
    };

    use super::{AlgebraicPlanarInput, PlanarInput, Polarisation};

    type C = Complex64;
    type D = Ix1;
    type P = RealParameter;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn planar_input() -> PlanarInput<C, D> {
        PlanarInput::new(
            arr1(&[c(1000.0, 1.0), c(1100.0, 2.0), c(1200.0, 3.0)]),
            arr1(&[c(100.0, -1.0), c(200.0, -2.0), c(300.0, -3.0)]),
            Polarisation::TransverseMagnetic,
        )
    }

    fn zeros() -> Array1<C> {
        arr1(&[c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)])
    }

    fn ones() -> Array1<C> {
        arr1(&[c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)])
    }

    fn assert_metadata_preserved<I>(input: &AlgebraicPlanarInput<I>) {
        assert_eq!(input.polarisation(), Polarisation::TransverseMagnetic,);
    }

    #[test]
    fn value_lift_preserves_coordinates_and_polarisation() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<Array1<C>> = AlgebraicPlanarInput::values(&planar);

        assert_eq!(lifted.vacuum_wavenumber(), planar.vacuum_wavenumber(),);

        assert_eq!(lifted.parallel_wavenumber(), planar.parallel_wavenumber(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn first_vacuum_lift_seeds_only_vacuum_wavenumber() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJetFirst<C, D, P>> =
            AlgebraicPlanarInput::first(&planar, PlanarSeed::VacuumWavenumber);

        assert_eq!(
            lifted.vacuum_wavenumber().value(),
            planar.vacuum_wavenumber(),
        );

        assert_eq!(lifted.vacuum_wavenumber().first(), &ones(),);

        assert_eq!(
            lifted.parallel_wavenumber().value(),
            planar.parallel_wavenumber(),
        );

        assert_eq!(lifted.parallel_wavenumber().first(), &zeros(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn first_parallel_lift_seeds_only_parallel_wavenumber() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJetFirst<C, D, P>> =
            AlgebraicPlanarInput::first(&planar, PlanarSeed::ParallelWavenumber);

        assert_eq!(
            lifted.vacuum_wavenumber().value(),
            planar.vacuum_wavenumber(),
        );

        assert_eq!(lifted.vacuum_wavenumber().first(), &zeros(),);

        assert_eq!(
            lifted.parallel_wavenumber().value(),
            planar.parallel_wavenumber(),
        );

        assert_eq!(lifted.parallel_wavenumber().first(), &ones(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn first_structural_lift_keeps_both_coordinates_constant() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJetFirst<C, D, P>> =
            AlgebraicPlanarInput::first(&planar, PlanarSeed::Structural);

        assert_eq!(
            lifted.vacuum_wavenumber().value(),
            planar.vacuum_wavenumber(),
        );

        assert_eq!(lifted.vacuum_wavenumber().first(), &zeros(),);

        assert_eq!(
            lifted.parallel_wavenumber().value(),
            planar.parallel_wavenumber(),
        );

        assert_eq!(lifted.parallel_wavenumber().first(), &zeros(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn second_vacuum_lift_seeds_first_derivative_only() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJet<C, D, P>> =
            AlgebraicPlanarInput::second(&planar, PlanarSeed::VacuumWavenumber);

        assert_eq!(
            lifted.vacuum_wavenumber().value(),
            planar.vacuum_wavenumber(),
        );

        assert_eq!(lifted.vacuum_wavenumber().first(), &ones(),);

        assert_eq!(lifted.vacuum_wavenumber().second(), &zeros(),);

        assert_eq!(
            lifted.parallel_wavenumber().value(),
            planar.parallel_wavenumber(),
        );

        assert_eq!(lifted.parallel_wavenumber().first(), &zeros(),);

        assert_eq!(lifted.parallel_wavenumber().second(), &zeros(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn second_parallel_lift_seeds_first_derivative_only() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJet<C, D, P>> =
            AlgebraicPlanarInput::second(&planar, PlanarSeed::ParallelWavenumber);

        assert_eq!(
            lifted.vacuum_wavenumber().value(),
            planar.vacuum_wavenumber(),
        );

        assert_eq!(lifted.vacuum_wavenumber().first(), &zeros(),);

        assert_eq!(lifted.vacuum_wavenumber().second(), &zeros(),);

        assert_eq!(
            lifted.parallel_wavenumber().value(),
            planar.parallel_wavenumber(),
        );

        assert_eq!(lifted.parallel_wavenumber().first(), &ones(),);

        assert_eq!(lifted.parallel_wavenumber().second(), &zeros(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn second_structural_lift_keeps_both_coordinates_constant() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJet<C, D, P>> =
            AlgebraicPlanarInput::second(&planar, PlanarSeed::Structural);

        assert_eq!(
            lifted.vacuum_wavenumber().value(),
            planar.vacuum_wavenumber(),
        );

        assert_eq!(lifted.vacuum_wavenumber().first(), &zeros(),);

        assert_eq!(lifted.vacuum_wavenumber().second(), &zeros(),);

        assert_eq!(
            lifted.parallel_wavenumber().value(),
            planar.parallel_wavenumber(),
        );

        assert_eq!(lifted.parallel_wavenumber().first(), &zeros(),);

        assert_eq!(lifted.parallel_wavenumber().second(), &zeros(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn bivariate_lift_assigns_vacuum_to_first_variable() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJetBivariate<C, D, P>> =
            AlgebraicPlanarInput::full_spectral(&planar);

        let vacuum = lifted.vacuum_wavenumber();

        assert_eq!(vacuum.value(), planar.vacuum_wavenumber(),);

        assert_eq!(vacuum.x(), &ones(),);

        assert_eq!(vacuum.y(), &zeros(),);

        assert_eq!(vacuum.xx(), &zeros(),);

        assert_eq!(vacuum.xy(), &zeros(),);

        assert_eq!(vacuum.yy(), &zeros(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn bivariate_lift_assigns_parallel_to_second_variable() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJetBivariate<C, D, P>> =
            AlgebraicPlanarInput::full_spectral(&planar);

        let parallel = lifted.parallel_wavenumber();

        assert_eq!(parallel.value(), planar.parallel_wavenumber(),);

        assert_eq!(parallel.x(), &zeros(),);

        assert_eq!(parallel.y(), &ones(),);

        assert_eq!(parallel.xx(), &zeros(),);

        assert_eq!(parallel.xy(), &zeros(),);

        assert_eq!(parallel.yy(), &zeros(),);

        assert_metadata_preserved(&lifted);
    }

    #[test]
    fn algebraic_input_into_parts_preserves_lifted_values() {
        let planar = planar_input();

        let lifted: AlgebraicPlanarInput<ArrayJetFirst<C, D, P>> =
            AlgebraicPlanarInput::first(&planar, PlanarSeed::VacuumWavenumber);

        let (vacuum, parallel, polarisation) = lifted.into_parts();

        assert_eq!(vacuum.value(), planar.vacuum_wavenumber(),);

        assert_eq!(vacuum.first(), &ones(),);

        assert_eq!(parallel.value(), planar.parallel_wavenumber(),);

        assert_eq!(parallel.first(), &zeros(),);

        assert_eq!(polarisation, Polarisation::TransverseMagnetic,);
    }
}
