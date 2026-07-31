use crate::backend::PlaneWaveEntries;

pub trait ProjectPlaneWaveModeDeterminant: PlaneWaveEntries {
    type Determinant;

    fn project_determinant(&self, exterior: &Self::ExteriorContext) -> Self::Determinant;
}

pub struct PlaneWaveDeterminant<J> {
    value: J,
}

impl<J> PlaneWaveDeterminant<J> {
    pub fn new(value: J) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &J {
        &self.value
    }

    pub fn map<J2>(self, transform: impl Fn(J) -> J2) -> PlaneWaveDeterminant<J2> {
        PlaneWaveDeterminant {
            value: transform(self.value),
        }
    }

    pub fn into_inner(self) -> J {
        self.value
    }
}
