use crate::differential::DirectionalCoordinate;

/// Crystallise values without derivative data.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct ValueMode;

/// Crystallise one first directional derivative.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DirectionalFirstMode {
    coordinate: DirectionalCoordinate,
}

impl DirectionalFirstMode {
    pub(crate) const fn new(coordinate: DirectionalCoordinate) -> Self {
        Self { coordinate }
    }

    pub(crate) const fn coordinate(self) -> DirectionalCoordinate {
        self.coordinate
    }
}

/// Crystallise first and second derivatives in one direction.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DirectionalSecondMode {
    coordinate: DirectionalCoordinate,
}

impl DirectionalSecondMode {
    pub(crate) const fn new(coordinate: DirectionalCoordinate) -> Self {
        Self { coordinate }
    }

    pub(crate) const fn coordinate(self) -> DirectionalCoordinate {
        self.coordinate
    }
}

/// Crystallise the canonical spectral gradient and Hessian.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct SpectralFirstMode;

/// Crystallise the canonical spectral gradient and Hessian.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct SpectralSecondMode;
