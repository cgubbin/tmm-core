#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Polarisation {
    TransverseElectric,
    TransverseMagnetic,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DerivativeVariable {
    Frequency,
    FrequencySquared,
    PropagationConstant,
    PropagationConstantSquared,
    Thickness(usize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DerivativeRequest {
    None,
    First(DerivativeVariable),
    Second(DerivativeVariable),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transfer2Input<I> {
    pub wavenumber: I,
    pub propagation_constant_squared: I,
    pub polarisation: Polarisation,
    pub derivatives: DerivativeRequest,
}

impl<I> Transfer2Input<I> {
    pub fn new(wavenumber: I, propagation_constant_squared: I, polarisation: Polarisation) -> Self {
        Self {
            wavenumber,
            propagation_constant_squared,
            polarisation,
            derivatives: DerivativeRequest::None,
        }
    }

    pub fn with_derivatives(mut self, derivatives: DerivativeRequest) -> Self {
        self.derivatives = derivatives;
        self
    }
}
