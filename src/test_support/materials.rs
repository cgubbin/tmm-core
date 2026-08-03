use crate::{
    ComplexScalar, Constant, DerivativeOrder, DifferentiableMaterial,
    DifferentiableMeromorphicMaterial, MeromorphicMaterial, Sampled, material::Material,
};

/// Linear dispersive material used for testing.
///
/// ε(k₀) = ε₀ + slope · k₀
/// μ(k₀) = μ₀ + slope · k₀
#[derive(Clone, Debug)]
pub struct LinearDispersion {
    pub(super) epsilon0: f64,
    pub(super) epsilon_slope: f64,
    pub(super) mu0: f64,
    pub(super) mu_slope: f64,
}

impl LinearDispersion {
    pub fn new(epsilon0: f64, epsilon_slope: f64, mu0: f64, mu_slope: f64) -> Self {
        Self {
            epsilon0,
            epsilon_slope,
            mu0,
            mu_slope,
        }
    }
}

pub fn vacuum() -> Constant<f64> {
    Constant::new(1.0, 1.0)
}

pub fn constant(epsilon: f64, mu: f64) -> Constant<f64> {
    Constant::new(epsilon, mu)
}

pub fn linear(epsilon0: f64, epsilon_slope: f64, mu0: f64, mu_slope: f64) -> LinearDispersion {
    LinearDispersion::new(epsilon0, epsilon_slope, mu0, mu_slope)
}

impl Material for LinearDispersion {
    type Real = f64;

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        wavenumber.map(|x| {
            C::from_real(self.epsilon0) + C::from_real(x) * C::from_real(self.epsilon_slope)
        })
    }

    fn relative_permeability<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        wavenumber.map(|x| C::from_real(self.mu0) + C::from_real(x) * C::from_real(self.mu_slope))
    }
}

impl DifferentiableMaterial for LinearDispersion {
    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        match order {
            DerivativeOrder::First => wavenumber.map(|_| C::from_real(self.epsilon_slope)),
            DerivativeOrder::Second => wavenumber.map(|_| C::zero()),
            DerivativeOrder::Third => wavenumber.map(|_| C::zero()),
        }
    }

    fn relative_permeability_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        match order {
            DerivativeOrder::First => wavenumber.map(|_| C::from_real(self.mu_slope)),
            DerivativeOrder::Second => wavenumber.map(|_| C::zero()),
            DerivativeOrder::Third => wavenumber.map(|_| C::zero()),
        }
    }
}

impl MeromorphicMaterial for LinearDispersion {
    fn relative_permittivity_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber
            .map(|x| C::from_real(self.epsilon0) + x * C::from_real(self.epsilon_slope))
    }

    fn relative_permeability_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|x| C::from_real(self.mu0) + x * C::from_real(self.mu_slope))
    }
}

impl DifferentiableMeromorphicMaterial for LinearDispersion {
    fn relative_permittivity_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        match order {
            DerivativeOrder::First => vacuum_wavenumber.map(|_| C::from_real(self.epsilon_slope)),
            DerivativeOrder::Second => vacuum_wavenumber.map(|_| C::zero()),
            DerivativeOrder::Third => vacuum_wavenumber.map(|_| C::zero()),
        }
    }

    fn relative_permeability_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C>,
    {
        match order {
            DerivativeOrder::First => vacuum_wavenumber.map(|_| C::from_real(self.mu_slope)),
            DerivativeOrder::Second => vacuum_wavenumber.map(|_| C::zero()),
            DerivativeOrder::Third => vacuum_wavenumber.map(|_| C::zero()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QuadraticDispersion {
    pub(super) epsilon0: f64,
    pub(super) epsilon_slope: f64,
    pub(super) epsilon_curvature: f64,
    pub(super) mu0: f64,
    pub(super) mu_slope: f64,
    pub(super) mu_curvature: f64,
}

impl QuadraticDispersion {
    pub fn new(
        epsilon0: f64,
        epsilon_slope: f64,
        epsilon_curvature: f64,
        mu0: f64,
        mu_slope: f64,
        mu_curvature: f64,
    ) -> Self {
        Self {
            epsilon0,
            epsilon_slope,
            epsilon_curvature,
            mu0,
            mu_slope,
            mu_curvature,
        }
    }
}

pub fn quadratic(
    epsilon0: f64,
    epsilon_slope: f64,
    epsilon_curvature: f64,
    mu0: f64,
    mu_slope: f64,
    mu_curvature: f64,
) -> QuadraticDispersion {
    QuadraticDispersion::new(
        epsilon0,
        epsilon_slope,
        epsilon_curvature,
        mu0,
        mu_slope,
        mu_curvature,
    )
}

impl Material for QuadraticDispersion {
    type Real = f64;

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        wavenumber.map(|x| {
            C::from_real(self.epsilon0)
                + C::from_real(x) * C::from_real(self.epsilon_slope)
                + C::from_real(x) * C::from_real(x) * C::from_real(self.epsilon_curvature)
        })
    }

    fn relative_permeability<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        wavenumber.map(|x| {
            C::from_real(self.mu0)
                + C::from_real(x) * C::from_real(self.mu_slope)
                + C::from_real(x) * C::from_real(x) * C::from_real(self.mu_curvature)
        })
    }
}

impl DifferentiableMaterial for QuadraticDispersion {
    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = f64>,
    {
        match order {
            DerivativeOrder::First => wavenumber.map(|x| {
                C::from_real(self.epsilon_slope)
                    + C::from_real(2.0 * x) * C::from_real(self.epsilon_curvature)
            }),
            DerivativeOrder::Second => {
                wavenumber.map(|_| C::from_real(2.0) * C::from_real(self.epsilon_curvature))
            }
            DerivativeOrder::Third => wavenumber.map(|_| C::zero()),
        }
    }

    fn relative_permeability_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C::RealField>,
    {
        match order {
            DerivativeOrder::First => vacuum_wavenumber.map(|x| {
                C::from_real(self.mu_slope)
                    + C::from_real(2.0 * x) * C::from_real(self.mu_curvature)
            }),
            DerivativeOrder::Second => {
                vacuum_wavenumber.map(|_| C::from_real(2.0) * C::from_real(self.mu_curvature))
            }
            DerivativeOrder::Third => vacuum_wavenumber.map(|_| C::zero()),
        }
    }
}

impl MeromorphicMaterial for QuadraticDispersion {
    fn relative_permittivity_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|x| {
            C::from_real(self.epsilon0)
                + x * C::from_real(self.epsilon_slope)
                + x * x * C::from_real(self.epsilon_curvature)
        })
    }

    fn relative_permeability_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|x| {
            C::from_real(self.mu0)
                + x * C::from_real(self.mu_slope)
                + x * x * C::from_real(self.mu_curvature)
        })
    }
}

impl DifferentiableMeromorphicMaterial for QuadraticDispersion {
    fn relative_permittivity_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        match order {
            DerivativeOrder::First => vacuum_wavenumber.map(|x| {
                C::from_real(self.epsilon_slope)
                    + C::from_real(2.0) * x * C::from_real(self.epsilon_curvature)
            }),
            DerivativeOrder::Second => {
                vacuum_wavenumber.map(|_| C::from_real(2.0) * C::from_real(self.epsilon_curvature))
            }
            DerivativeOrder::Third => vacuum_wavenumber.map(|_| C::zero()),
        }
    }

    fn relative_permeability_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        match order {
            DerivativeOrder::First => vacuum_wavenumber.map(|x| {
                C::from_real(self.mu_slope)
                    + C::from_real(2.0) * x * C::from_real(self.mu_curvature)
            }),
            DerivativeOrder::Second => {
                vacuum_wavenumber.map(|_| C::from_real(2.0) * C::from_real(self.mu_curvature))
            }
            DerivativeOrder::Third => vacuum_wavenumber.map(|_| C::zero()),
        }
    }
}
