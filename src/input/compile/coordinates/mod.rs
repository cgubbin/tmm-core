mod error;
mod in_plane;
mod jet;
mod seed;
mod spectral;

pub(crate) use error::CoordinateCompileError;
pub(crate) use jet::CanonicalCoordinateJet;
use seed::seed_coordinate;

use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};
use std::collections::HashMap;

use crate::{
    IncidentSide, Polarisation,
    input::{
        CanonicalCoordinates, InPlaneCoordinate, PlaneWaveCoordinates, SpectralCoordinate,
        compile::{assignment::CoordinateAssignment, context::CoordinateContext, seed::SeedJet},
        plane_wave::PlaneWaveCoordinateValues,
    },
};
use in_plane::{canonicalise_in_plane, validate_in_plane};
use spectral::{canonicalise_spectral, validate_spectral};

/// A caller-facing independent coordinate variable.
///
/// These variables are seeded before conversion to canonical coordinates.
/// Their precise physical interpretation is provided by
/// [`PlaneWaveCoordinates`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoordinateVariable {
    /// The caller-facing spectral coordinate.
    Spectral,

    /// The caller-facing in-plane coordinate.
    InPlane,
}

pub(crate) struct CompiledCoordinates<J, R, D>
where
    D: Dimension,
{
    canonical: CanonicalCoordinates<J>,
    context: CoordinateContext<R, D>,
}

impl<J, R, D: Dimension> CompiledCoordinates<J, R, D> {
    pub(crate) fn new(
        canonical: CanonicalCoordinates<J>,
        context: CoordinateContext<R, D>,
    ) -> Self {
        Self { context, canonical }
    }

    pub(crate) fn context(&self) -> &CoordinateContext<R, D> {
        &self.context
    }

    pub(crate) fn canonical(&self) -> &CanonicalCoordinates<J> {
        &self.canonical
    }

    pub(crate) fn into_parts(self) -> (CanonicalCoordinates<J>, CoordinateContext<R, D>) {
        (self.canonical, self.context)
    }
}

pub(crate) struct SeededSpectral<J> {
    vacuum_angular_wavenumber: J,
}

impl<J> SeededSpectral<J> {
    pub(crate) fn new(vacuum_angular_wavenumber: J) -> Self {
        Self {
            vacuum_angular_wavenumber,
        }
    }

    pub(crate) fn vacuum_angular_wavenumber(&self) -> &J {
        &self.vacuum_angular_wavenumber
    }

    pub(crate) fn into_inner(self) -> J {
        self.vacuum_angular_wavenumber
    }
}

pub(crate) fn compile_spectral<C, D, J>(
    values: &Array<C::RealField, D>,
    coordinate: SpectralCoordinate,
    slot: Option<usize>,
) -> Result<SeededSpectral<J>, CoordinateCompileError<C::RealField>>
where
    C: ComplexField,
    C::RealField: Float + FloatConst + FromPrimitive + Copy,
    D: Dimension,
    J: SeedJet<Array<C, D>> + CanonicalCoordinateJet<C, D>,
{
    validate_spectral(values)?;

    let sampled = super::complexify(values);

    let seeded = seed_coordinate(sampled, slot).map_err(|source| CoordinateCompileError::Seed {
        variable: CoordinateVariable::Spectral,
        source,
    })?;

    Ok(SeededSpectral {
        vacuum_angular_wavenumber: canonicalise_spectral::<C, D, J>(seeded, coordinate),
    })
}

pub(crate) fn compile_in_plane<C, D, J>(
    values: &Array<C::RealField, D>,
    coordinate: InPlaneCoordinate,
    vacuum_angular_wavenumber: &J,
    incident_index: Option<&J>,
    slot: Option<usize>,
) -> Result<J, CoordinateCompileError<C::RealField>>
where
    C: ComplexField,
    C::RealField: Float + FloatConst + FromPrimitive + Copy,
    D: Dimension,
    J: SeedJet<Array<C, D>> + CanonicalCoordinateJet<C, D>,
{
    validate_in_plane(values, coordinate)?;

    let sampled = super::complexify(values);

    let seeded =
        seed_coordinate(sampled.clone(), slot).map_err(|source| CoordinateCompileError::Seed {
            variable: CoordinateVariable::InPlane,
            source,
        })?;

    Ok(canonicalise_in_plane::<C, D, J>(
        seeded,
        coordinate,
        vacuum_angular_wavenumber,
        incident_index,
    )?)
}
