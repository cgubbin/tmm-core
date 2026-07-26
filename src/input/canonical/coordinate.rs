use crate::input::{IncidentSide, Polarisation};

/// Complete canonical description of a driven plane-wave problem.
///
/// Public coordinate parameterisations are compiled into this type before
/// entering the backend.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalPlaneWaveInput<J> {
    coordinates: CanonicalCoordinates<J>,
    polarisation: Polarisation,
    incident_side: IncidentSide,
}

impl<J> CanonicalPlaneWaveInput<J> {
    /// Construct a planar evaluation input.
    ///
    /// `vacuum_angular_wavenumber` and `parallel_angular_wavenumber` must use the same
    /// inverse-length unit. For sampled values, they must also have matching
    /// shapes.
    pub(crate) fn from_coordinates(
        vacuum_angular_wavenumber: J,
        parallel_angular_wavenumber: J,
        polarisation: Polarisation,
        incident_side: IncidentSide,
    ) -> Self {
        Self {
            coordinates: CanonicalCoordinates::new(
                vacuum_angular_wavenumber,
                parallel_angular_wavenumber,
            ),
            polarisation,
            incident_side,
        }
    }

    pub(crate) fn new(
        coordinates: CanonicalCoordinates<J>,
        polarisation: Polarisation,
        incident_side: IncidentSide,
    ) -> Self {
        Self {
            coordinates,
            polarisation,
            incident_side,
        }
    }

    /// Return the canonical coordinates.
    pub(crate) fn coordinates(&self) -> &CanonicalCoordinates<J> {
        &self.coordinates
    }

    /// Return the vacuum wavenumber `k₀`.
    pub(crate) fn vacuum_angular_wavenumber(&self) -> &J {
        self.coordinates.vacuum_angular_wavenumber()
    }

    /// Return the conserved parallel wavenumber `k∥`.
    pub(crate) fn parallel_angular_wavenumber(&self) -> &J {
        self.coordinates.parallel_angular_wavenumber()
    }

    /// Return the polarisation.
    pub(crate) fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Return the incident side.
    pub(crate) fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    pub(crate) fn into_solver_input(self) -> (CanonicalSolverInput<J>, IncidentSide) {
        let Self {
            coordinates,
            polarisation,
            incident_side,
        } = self;

        (
            CanonicalSolverInput::new(coordinates, polarisation),
            incident_side,
        )
    }

    /// Consume the input and return its coordinates and polarisation.
    pub(crate) fn into_components(self) -> (CanonicalCoordinates<J>, Polarisation, IncidentSide) {
        (self.coordinates, self.polarisation, self.incident_side)
    }

    /// Consume the input and return its flattened components.
    pub(crate) fn into_parts(self) -> (J, J, Polarisation, IncidentSide) {
        let (coordinates, polarisation, incident_side) = self.into_components();

        let (vacuum_angular_wavenumber, parallel_angular_wavenumber) = coordinates.into_parts();

        (
            vacuum_angular_wavenumber,
            parallel_angular_wavenumber,
            polarisation,
            incident_side,
        )
    }
}

/// Minimal canonical input required by the oriented core solve loop.
///
/// Incident-side handling has already been resolved before this type reaches
/// the core solver.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalSolverInput<J> {
    coordinates: CanonicalCoordinates<J>,
    polarisation: Polarisation,
}

impl<J> CanonicalSolverInput<J> {
    /// Construct a planar evaluation input.
    ///
    /// `vacuum_angular_wavenumber` and `parallel_angular_wavenumber` must use the same
    /// inverse-length unit. For sampled values, they must also have matching
    /// shapes.
    pub(crate) fn from_coordinates(
        vacuum_angular_wavenumber: J,
        parallel_angular_wavenumber: J,
        polarisation: Polarisation,
    ) -> Self {
        Self {
            coordinates: CanonicalCoordinates::new(
                vacuum_angular_wavenumber,
                parallel_angular_wavenumber,
            ),
            polarisation,
        }
    }

    pub(crate) fn new(coordinates: CanonicalCoordinates<J>, polarisation: Polarisation) -> Self {
        Self {
            coordinates,
            polarisation,
        }
    }

    /// Return the canonical coordinates.
    pub(crate) fn coordinates(&self) -> &CanonicalCoordinates<J> {
        &self.coordinates
    }

    /// Return the vacuum wavenumber `k₀`.
    pub(crate) fn vacuum_angular_wavenumber(&self) -> &J {
        self.coordinates.vacuum_angular_wavenumber()
    }

    /// Return the conserved parallel wavenumber `k∥`.
    pub(crate) fn parallel_angular_wavenumber(&self) -> &J {
        self.coordinates.parallel_angular_wavenumber()
    }
    /// Both coordinate values:
    ///
    /// - are expressed in inverse centimetres;
    /// - have compatible sampled shapes;
    /// - are interpreted elementwise;
    /// - contain one `(k₀, k∥)` pair per solved state.
    ///
    /// The sampled representation may be an array or a jet whose coefficients
    /// are arrays.
    /// Return the polarisation.
    pub(crate) fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Consume the input and return its coordinates and polarisation.
    pub(crate) fn into_components(self) -> (CanonicalCoordinates<J>, Polarisation) {
        (self.coordinates, self.polarisation)
    }

    /// Consume the input and return its flattened components.
    pub(crate) fn into_parts(self) -> (J, J, Polarisation) {
        let (coordinates, polarisation) = self.into_components();

        let (vacuum_angular_wavenumber, parallel_angular_wavenumber) = coordinates.into_parts();

        (
            vacuum_angular_wavenumber,
            parallel_angular_wavenumber,
            polarisation,
        )
    }
}

/// Canonical coordinates used by the planar backend.
///
/// Both coordinate values:
///
/// - are expressed in inverse centimetres;
/// - have compatible sampled shapes;
/// - are interpreted elementwise;
/// - contain one `(k₀, k∥)` pair per solved plane-wave state.
///
/// `J` is the complete sampled algebraic representation. It may, for example,
/// be an array, a zeroth-order jet, a directional jet, or a bivariate jet.
///
/// Shape and finiteness validation are performed while compiling the public
/// input. This type represents coordinates whose canonical invariants have
/// already been established.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalCoordinates<J> {
    vacuum_angular_wavenumber: J,
    parallel_angular_wavenumber: J,
}

impl<J> CanonicalCoordinates<J> {
    pub(crate) fn new(vacuum_angular_wavenumber: J, parallel_angular_wavenumber: J) -> Self {
        Self {
            vacuum_angular_wavenumber,
            parallel_angular_wavenumber,
        }
    }

    pub(crate) fn vacuum_angular_wavenumber(&self) -> &J {
        &self.vacuum_angular_wavenumber
    }

    pub(crate) fn parallel_angular_wavenumber(&self) -> &J {
        &self.parallel_angular_wavenumber
    }

    pub(crate) fn into_parts(self) -> (J, J) {
        (
            self.vacuum_angular_wavenumber,
            self.parallel_angular_wavenumber,
        )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, arr1};
    use num_complex::Complex64;

    use super::{CanonicalCoordinates, CanonicalPlaneWaveInput, CanonicalSolverInput};
    use crate::{
        algebra::Jet0,
        input::{IncidentSide, Polarisation},
    };

    type C = Complex64;

    #[test]
    fn coordinates_expose_and_return_their_values() {
        let vacuum = arr1(&[C::new(1000.0, 0.0), C::new(1100.0, 0.0)]);

        let parallel = arr1(&[C::new(100.0, 0.0), C::new(200.0, 0.0)]);

        let coordinates = CanonicalCoordinates::new(vacuum.clone(), parallel.clone());

        assert_eq!(coordinates.vacuum_angular_wavenumber(), &vacuum);
        assert_eq!(coordinates.parallel_angular_wavenumber(), &parallel);

        let (returned_vacuum, returned_parallel) = coordinates.into_parts();

        assert_eq!(returned_vacuum, vacuum);
        assert_eq!(returned_parallel, parallel);
    }

    #[test]
    fn plane_wave_input_exposes_its_components() {
        let vacuum = arr1(&[C::new(1000.0, 0.0)]);
        let parallel = arr1(&[C::new(100.0, 0.0)]);

        let input = CanonicalPlaneWaveInput::from_coordinates(
            vacuum.clone(),
            parallel.clone(),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert_eq!(input.vacuum_angular_wavenumber(), &vacuum);
        assert_eq!(input.parallel_angular_wavenumber(), &parallel);
        assert_eq!(input.polarisation(), Polarisation::TransverseElectric,);
        assert_eq!(input.incident_side(), IncidentSide::Left);
    }

    #[test]
    fn solve_input_preserves_owned_components() {
        let vacuum: Array1<C> = arr1(&[C::new(1000.0, 1.0), C::new(1100.0, 2.0)]);

        let parallel: Array1<C> = arr1(&[C::new(100.0, -1.0), C::new(200.0, -2.0)]);

        let input = CanonicalSolverInput::from_coordinates(
            vacuum.clone(),
            parallel.clone(),
            Polarisation::TransverseMagnetic,
        );

        let (returned_vacuum, returned_parallel, returned_polarisation) = input.into_parts();

        assert_eq!(returned_vacuum, vacuum);
        assert_eq!(returned_parallel, parallel);
        assert_eq!(returned_polarisation, Polarisation::TransverseMagnetic,);
    }

    #[test]
    fn canonical_input_accepts_jet_storage() {
        let vacuum: Jet0<Array1<f64>> = Jet0::new(arr1(&[1000.0, 1100.0]));
        let parallel = Jet0::new(arr1(&[100.0, 200.0]));

        let input = CanonicalSolverInput::from_coordinates(
            vacuum.clone(),
            parallel.clone(),
            Polarisation::TransverseElectric,
        );

        assert_eq!(input.vacuum_angular_wavenumber(), &vacuum);
        assert_eq!(input.parallel_angular_wavenumber(), &parallel);
    }
}
