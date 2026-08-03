use crate::{
    IncidentSide,
    algebra::{RealScalarAlgebra, ScalarAlgebra},
    material::lifting::ConstitutiveDerivativeEvaluator,
    observable::{BoundaryProjectionError, Layers, layer::energy::IsotropicEnergyCoefficients},
};

use super::EnergyDefinition;

/// Constitutive values required for Brillouin layer energy.
///
/// The spectral derivative fields are intrinsic material derivatives with
/// respect to the canonical vacuum angular wavenumber `k0`. They retain the
/// outer jet representation independently.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicBrillouinEnergyData<A> {
    epsilon_spectral_first: A,
    mu_spectral_first: A,
}

impl<A> IsotropicBrillouinEnergyData<A> {
    pub(crate) const fn new(epsilon_spectral_first: A, mu_spectral_first: A) -> Self {
        Self {
            epsilon_spectral_first,
            mu_spectral_first,
        }
    }

    pub(crate) fn epsilon_spectral_first(&self) -> &A {
        &self.epsilon_spectral_first
    }

    pub(crate) fn mu_spectral_first(&self) -> &A {
        &self.mu_spectral_first
    }

    pub(crate) fn into_parts(self) -> (A, A) {
        (self.epsilon_spectral_first, self.mu_spectral_first)
    }
}

/// Evaluated constitutive data required by one layer-energy projection.
///
/// These are analysis-specific values evaluated lazily from the canonical
/// stack. They are not retained by the propagation backend.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerEnergyData<A> {
    epsilon: A,
    mu: A,
    definition: EnergyDefinition,
}

impl<A> IsotropicLayerEnergyData<A> {
    pub(crate) const fn nondispersive(epsilon: A, mu: A) -> Self {
        Self {
            epsilon,
            mu,
            definition: EnergyDefinition::Nondispersive,
        }
    }

    pub(crate) fn epsilon(&self) -> &A {
        &self.epsilon
    }

    pub(crate) fn mu(&self) -> &A {
        &self.mu
    }

    pub(crate) const fn definition(&self) -> EnergyDefinition {
        self.definition
    }

    pub(crate) fn into_parts(self) -> (A, A, EnergyDefinition) {
        (self.epsilon, self.mu, self.definition)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LayerEnergyError {
    #[error("finite-layer boundary data are unavailable: {0}")]
    Boundary(#[from] BoundaryProjectionError),

    #[error(
        "Brillouin energy requires the first intrinsic spectral derivative \
         of {quantity:?} in finite layer {layer}"
    )]
    MissingSpectralDerivative {
        layer: usize,
        quantity: ConstitutiveQuantity,
    },

    #[error(
        "material in finite layer {layer} does not support \
         {definition:?} energy analysis"
    )]
    UnsupportedDefinition {
        layer: usize,
        definition: EnergyDefinition,
    },

    #[error("incident power-flux normalization is invalid for {side:?} incidence")]
    InvalidIncidentFlux { side: IncidentSide },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConstitutiveQuantity {
    Epsilon,
    Mu,
}

pub(crate) fn nondispersive_energy_coefficients<A>(
    vacuum_angular_wavenumber: &A,
    data: &IsotropicLayerEnergyData<A>,
    incident_flux_magnitude: &A::RealJet,
) -> IsotropicEnergyCoefficients<A::RealJet>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
{
    let vacuum_squared = vacuum_angular_wavenumber.magnitude_squared();

    /*
     * The common factor determines whether this represents energy per unit
     * incident power, group delay per unit length, or another canonical
     * normalization.
     */
    let common = vacuum_squared.divide(incident_flux_magnitude);

    let electric = common.multiply(&data.epsilon().real());

    let magnetic = common.multiply(&data.mu().real());

    IsotropicEnergyCoefficients::new(electric, magnetic)
}

use std::fmt::Debug;

use ndarray::Dimension;

use crate::{ComplexScalar, material::ConstitutiveSpectralFirstLift};

/// Evaluate the additional intrinsic constitutive derivatives required for
/// Brillouin energy.
///
/// The material iterator must be in physical finite-layer order.
pub(crate) fn evaluate_brillouin_layer_energy_data<'a, E, M, A>(
    materials: impl IntoIterator<Item = &'a M>,
    vacuum_angular_wavenumber: &A,
) -> Layers<IsotropicBrillouinEnergyData<A>>
where
    M: 'a,
    A: ScalarAlgebra + ConstitutiveSpectralFirstLift<E, M>,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
    E: ConstitutiveDerivativeEvaluator<A::Scalar, A::Dimension, M>,
{
    let layers = materials
        .into_iter()
        .map(|material| {
            let epsilon_spectral_first =
                A::relative_permittivity_spectral_first(material, vacuum_angular_wavenumber);

            let mu_spectral_first =
                A::relative_permeability_spectral_first(material, vacuum_angular_wavenumber);

            IsotropicBrillouinEnergyData::new(epsilon_spectral_first, mu_spectral_first)
        })
        .collect();

    Layers::new(layers)
}

#[cfg(test)]
mod evaluation_tests {
    use ndarray::{Array, Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        DifferentiableMaterial, Material, Sampled,
        algebra::{ArrayJet0, Jet0, RealParameter},
        domain::RealAxis,
        material::{DerivativeOrder, EvaluateDifferentiableMaterial, EvaluateMaterial},
    };

    type A = ArrayJet0<Complex64, Ix0, RealParameter>;

    #[derive(Clone, Debug)]
    struct PolynomialMaterial {
        epsilon_offset: f64,
        mu_offset: f64,
    }

    impl Material for PolynomialMaterial {
        type Real = f64;

        fn relative_permittivity<I, C>(&self, x: I) -> I::Mapped<C>
        where
            I: Sampled<Elem = Self::Real>,
            C: ComplexScalar<RealField = f64>,
        {
            x.map(|x| C::from_real(self.epsilon_offset + 2.0 * x + 3.0 * x * x + 5.0 * x * x * x))
        }

        fn relative_permeability<I, C>(&self, k0: I) -> I::Mapped<C>
        where
            I: Sampled<Elem = Self::Real>,
            C: ComplexScalar<RealField = f64>,
        {
            k0.map(|x| C::from_real(self.mu_offset + 7.0 * x + 11.0 * x * x + 13.0 * x * x * x))
        }
    }

    impl DifferentiableMaterial for PolynomialMaterial {
        fn relative_permittivity_derivative<I, C>(
            &self,
            k0: I,
            order: DerivativeOrder,
        ) -> I::Mapped<C>
        where
            I: Sampled<Elem = Self::Real>,
            C: ComplexScalar<RealField = f64>,
        {
            match order {
                DerivativeOrder::First => k0.map(|x| C::from_real(2.0 + 6.0 * x + 15.0 * x * x)),

                DerivativeOrder::Second => k0.map(|x| C::from_real(6.0 + 30.0 * x)),

                DerivativeOrder::Third => k0.map(|_| C::from_real(30.0)),
            }
        }

        fn relative_permeability_derivative<I, C>(
            &self,
            k0: I,
            order: DerivativeOrder,
        ) -> I::Mapped<C>
        where
            I: Sampled<Elem = Self::Real>,
            C: ComplexScalar<RealField = f64>,
        {
            match order {
                DerivativeOrder::First => k0.map(|x| C::from_real(7.0 + 22.0 * x + 39.0 * x * x)),

                DerivativeOrder::Second => k0.map(|x| C::from_real(22.0 + 78.0 * x)),

                DerivativeOrder::Third => k0.map(|_| C::from_real(78.0)),
            }
        }
    }

    #[test]
    fn evaluates_one_brillouin_record_per_material() {
        let materials = [
            PolynomialMaterial {
                epsilon_offset: 1.0,
                mu_offset: 2.0,
            },
            PolynomialMaterial {
                epsilon_offset: 3.0,
                mu_offset: 4.0,
            },
        ];

        let data = evaluate_brillouin_layer_energy_data::<RealAxis, _, A>(
            materials.iter(),
            &Jet0::new(arr0(Complex64::new(2.0, 0.0))),
        );

        assert_eq!(data.len(), 2);
    }

    #[test]
    fn evaluates_intrinsic_first_derivatives() {
        let material = PolynomialMaterial {
            epsilon_offset: 1.0,
            mu_offset: 2.0,
        };

        let data = evaluate_brillouin_layer_energy_data::<RealAxis, _, A>(
            [&material],
            &Jet0::new(arr0(Complex64::new(2.0, 0.0))),
        );

        let layer = data.get(0).unwrap();

        /*
         * epsilon' = 2 + 6k0 + 15k0²
         *          = 2 + 12 + 60 = 74
         *
         * mu'      = 7 + 22k0 + 39k0²
         *          = 7 + 44 + 156 = 207
         */
        assert_eq!(
            layer.epsilon_spectral_first().value()[()],
            Complex64::new(74.0, 0.0),
        );

        assert_eq!(
            layer.mu_spectral_first().value()[()],
            Complex64::new(207.0, 0.0),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::IsotropicBrillouinEnergyData;

    #[test]
    fn stores_both_intrinsic_derivatives() {
        let data = IsotropicBrillouinEnergyData::new(1, 2);

        assert_eq!(data.epsilon_spectral_first(), &1,);

        assert_eq!(data.mu_spectral_first(), &2,);
    }

    #[test]
    fn into_parts_preserves_order() {
        let data = IsotropicBrillouinEnergyData::new(1, 2);

        assert_eq!(data.into_parts(), (1, 2),);
    }
}

#[cfg(test)]
mod brillouin_tests {
    use ndarray::{Array, Dimension, Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        DifferentiableMaterial, Material, Sampled,
        algebra::{ArrayJet0, Jet0, RealParameter},
        domain::RealAxis,
        material::{DerivativeOrder, EvaluateDifferentiableMaterial, EvaluateMaterial},
    };

    type C = Complex64;
    type J = ArrayJet0<C, Ix0, RealParameter>;

    #[derive(Clone, Copy, Debug)]
    struct LinearMarkerMaterial {
        epsilon_slope: f64,
        mu_slope: f64,
    }

    impl Material for LinearMarkerMaterial {
        type Real = f64;

        fn relative_permittivity<I, X>(&self, vacuum_wavenumber: I) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            vacuum_wavenumber.map(|k| X::from_real(1.0 + self.epsilon_slope * k))
        }

        fn relative_permeability<I, X>(&self, vacuum_wavenumber: I) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            vacuum_wavenumber.map(|k| X::from_real(1.0 + self.mu_slope * k))
        }
    }

    impl DifferentiableMaterial for LinearMarkerMaterial {
        fn relative_permittivity_derivative<I, X>(
            &self,
            vacuum_wavenumber: I,
            order: DerivativeOrder,
        ) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            match order {
                DerivativeOrder::First => {
                    vacuum_wavenumber.map(|_| X::from_real(self.epsilon_slope))
                }

                DerivativeOrder::Second | DerivativeOrder::Third => {
                    vacuum_wavenumber.map(|_| X::zero())
                }
            }
        }

        fn relative_permeability_derivative<I, X>(
            &self,
            vacuum_wavenumber: I,
            order: DerivativeOrder,
        ) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            match order {
                DerivativeOrder::First => vacuum_wavenumber.map(|_| X::from_real(self.mu_slope)),

                DerivativeOrder::Second | DerivativeOrder::Third => {
                    vacuum_wavenumber.map(|_| X::zero())
                }
            }
        }
    }

    #[test]
    fn brillouin_data_stores_both_intrinsic_derivatives() {
        let data = IsotropicBrillouinEnergyData::new(1, 2);

        assert_eq!(data.epsilon_spectral_first(), &1,);

        assert_eq!(data.mu_spectral_first(), &2,);

        assert_eq!(data.into_parts(), (1, 2));
    }

    #[test]
    fn evaluation_returns_one_record_per_finite_layer_material() {
        let materials = [
            LinearMarkerMaterial {
                epsilon_slope: 2.0,
                mu_slope: 3.0,
            },
            LinearMarkerMaterial {
                epsilon_slope: 5.0,
                mu_slope: 7.0,
            },
        ];

        let spectral = Jet0::new(arr0(C::new(11.0, 0.0)));

        let result =
            evaluate_brillouin_layer_energy_data::<RealAxis, _, J>(materials.iter(), &spectral);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn evaluation_preserves_physical_material_order() {
        let materials = [
            LinearMarkerMaterial {
                epsilon_slope: 2.0,
                mu_slope: 3.0,
            },
            LinearMarkerMaterial {
                epsilon_slope: 5.0,
                mu_slope: 7.0,
            },
        ];

        let spectral = Jet0::new(arr0(C::new(11.0, 0.0)));

        let result =
            evaluate_brillouin_layer_energy_data::<RealAxis, _, J>(materials.iter(), &spectral);

        let first = result.get(0).unwrap();
        let second = result.get(1).unwrap();

        assert_eq!(first.epsilon_spectral_first().value()[()], C::new(2.0, 0.0),);

        assert_eq!(first.mu_spectral_first().value()[()], C::new(3.0, 0.0),);

        assert_eq!(
            second.epsilon_spectral_first().value()[()],
            C::new(5.0, 0.0),
        );

        assert_eq!(second.mu_spectral_first().value()[()], C::new(7.0, 0.0),);
    }

    #[test]
    fn empty_material_sequence_produces_empty_layer_sequence() {
        let materials: [LinearMarkerMaterial; 0] = [];

        let spectral = Jet0::new(arr0(C::new(11.0, 0.0)));

        let result =
            evaluate_brillouin_layer_energy_data::<RealAxis, _, J>(materials.iter(), &spectral);

        assert!(result.is_empty());
    }
}
