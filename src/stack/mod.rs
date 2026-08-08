mod builder;
mod layer;
mod thickness;
mod validation;

pub use builder::StackBuilder;
pub use layer::Layer;
pub use thickness::Thickness;
pub use validation::{ValidationConfig, ValidationError};

use either::Either;
use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};

use crate::{
    ComplexScalar, DifferentiableMaterial, DifferentiableMeromorphicMaterial, EvaluateMaterial,
    IncidentSide, MeromorphicMaterial,
    material::{
        AnalyticalMaterialHandle, DifferentiableMaterialHandle, Material, MaterialHandle,
        MeromorphicMaterialHandle, sample::Sampled,
    },
};

/// Heterogeneous stack supporting real-axis constitutive evaluation.
pub type MaterialStack<C> = Stack<MaterialHandle<C>, <C as ComplexField>::RealField>;

/// Heterogeneous stack supporting real-axis derivatives.
pub type DifferentiableMaterialStack<C> =
    Stack<DifferentiableMaterialHandle<C>, <C as ComplexField>::RealField>;

/// Heterogeneous stack supporting complex continuation.
pub type MeromorphicMaterialStack<C> =
    Stack<MeromorphicMaterialHandle<C>, <C as ComplexField>::RealField>;

/// Heterogeneous stack supporting complex continuation and derivatives.
pub type AnalyticalMaterialStack<C> =
    Stack<AnalyticalMaterialHandle<C>, <C as ComplexField>::RealField>;

#[derive(Clone, Debug, PartialEq)]
pub struct Stack<M, F> {
    left_exterior: M,
    right_exterior: M,
    layers_left_to_right: Vec<Layer<M, F>>,
}

pub(crate) enum PropagationDirection {
    LeftToRight,
    RightToLeft,
}

impl IncidentSide {
    fn propagation_direction(self) -> PropagationDirection {
        match self {
            Self::Left => PropagationDirection::LeftToRight,

            Self::Right => PropagationDirection::RightToLeft,
        }
    }
}

impl<M, F> Stack<M, F> {
    pub(crate) fn new(
        left_exterior: M,
        layers_left_to_right: Vec<Layer<M, F>>,
        right_exterior: M,
    ) -> Self {
        Self {
            left_exterior,
            right_exterior,
            layers_left_to_right,
        }
    }

    pub fn builder(left_exterior: M, right_exterior: M) -> StackBuilder<M, F>
    where
        F: Float,
    {
        StackBuilder::new(left_exterior, right_exterior)
    }
}

impl<C> Stack<MaterialHandle<C>, C::RealField>
where
    C: ComplexScalar + Copy + 'static,
    C::RealField: Copy + 'static,
{
    pub fn from_materials<L, U>(
        left_exterior: L,
        right_exterior: U,
    ) -> StackBuilder<MaterialHandle<C>, C::RealField>
    where
        L: Material<Real = C::RealField> + Send + Sync + 'static,
        U: Material<Real = C::RealField> + Send + Sync + 'static,
        C::RealField: Float,
    {
        StackBuilder::from_materials(left_exterior, right_exterior)
    }
}

impl<C> Stack<AnalyticalMaterialHandle<C>, C::RealField>
where
    C: ComplexScalar + Copy + 'static,
    C::RealField: Copy + 'static,
{
    pub fn from_analytical_materials<L, U>(
        left_exterior: L,
        right_exterior: U,
    ) -> StackBuilder<AnalyticalMaterialHandle<C>, C::RealField>
    where
        L: DifferentiableMeromorphicMaterial<Real = C::RealField> + Send + Sync + 'static,
        U: DifferentiableMeromorphicMaterial<Real = C::RealField> + Send + Sync + 'static,
        C::RealField: Float,
    {
        StackBuilder::from_analytical_materials(left_exterior, right_exterior)
    }
}

impl<C> Stack<DifferentiableMaterialHandle<C>, C::RealField>
where
    C: ComplexScalar + Copy + 'static,
    C::RealField: Copy + 'static,
{
    pub fn from_differentiable_materials<L, U>(
        left_exterior: L,
        right_exterior: U,
    ) -> StackBuilder<DifferentiableMaterialHandle<C>, C::RealField>
    where
        L: DifferentiableMaterial<Real = C::RealField> + Send + Sync + 'static,
        U: DifferentiableMaterial<Real = C::RealField> + Send + Sync + 'static,
        C::RealField: Float,
    {
        StackBuilder::from_differentiable_materials(left_exterior, right_exterior)
    }
}

impl<C> Stack<MeromorphicMaterialHandle<C>, C::RealField>
where
    C: ComplexScalar + Copy + 'static,
    C::RealField: Copy + 'static,
{
    pub fn from_meromorphic_materials<L, U>(
        left_exterior: L,
        right_exterior: U,
    ) -> StackBuilder<MeromorphicMaterialHandle<C>, C::RealField>
    where
        L: MeromorphicMaterial<Real = C::RealField> + Send + Sync + 'static,
        U: MeromorphicMaterial<Real = C::RealField> + Send + Sync + 'static,
        C::RealField: Float,
    {
        StackBuilder::from_meromorphic_materials(left_exterior, right_exterior)
    }
}

impl<M, F> Stack<M, F> {
    pub fn left_exterior(&self) -> &M {
        &self.left_exterior
    }

    pub fn right_exterior(&self) -> &M {
        &self.right_exterior
    }

    pub fn layers_left_to_right(&self) -> &[Layer<M, F>] {
        &self.layers_left_to_right
    }

    pub fn len(&self) -> usize {
        self.layers_left_to_right.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Layer<M, F>> {
        self.layers_left_to_right.iter()
    }

    /// Finite layers in the requested geometric direction.
    pub(crate) fn layers_in_direction(
        &self,
        direction: PropagationDirection,
    ) -> impl DoubleEndedIterator<Item = &Layer<M, F>> {
        match direction {
            PropagationDirection::LeftToRight => Either::Left(self.layers_left_to_right.iter()),

            PropagationDirection::RightToLeft => {
                Either::Right(self.layers_left_to_right.iter().rev())
            }
        }
    }

    /// Exterior encountered first in the requested direction.
    pub(crate) fn entrance_exterior(&self, direction: PropagationDirection) -> &M {
        match direction {
            PropagationDirection::LeftToRight => self.left_exterior(),

            PropagationDirection::RightToLeft => self.right_exterior(),
        }
    }

    pub(crate) fn incident_exterior(&self, incident_side: IncidentSide) -> &M {
        match incident_side {
            IncidentSide::Left => self.left_exterior(),

            IncidentSide::Right => self.right_exterior(),
        }
    }

    /// Exterior encountered last in the requested direction.
    pub(crate) fn exit_exterior(&self, direction: PropagationDirection) -> &M {
        match direction {
            PropagationDirection::LeftToRight => self.right_exterior(),

            PropagationDirection::RightToLeft => self.left_exterior(),
        }
    }

    pub fn incident_relative_permittivity<I, C>(
        &self,
        vacuum_wavenumber: I,
        side: IncidentSide,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = M::Real> + Copy,
        I: Sampled<Elem = M::Real>,
        M: EvaluateMaterial<C>,
    {
        let direction = side.propagation_direction();
        let material = self.entrance_exterior(direction);

        material.evaluate_relative_permittivity(vacuum_wavenumber)
    }

    pub fn validate(&self, config: &ValidationConfig<F>) -> Result<(), ValidationError<F>>
    where
        F: Copy + Float + FromPrimitive + std::fmt::Debug,
    {
        let thicknesses = self
            .layers_left_to_right()
            .iter()
            .map(|each| each.thickness())
            .collect::<Vec<_>>();

        config.validate_thicknesses(&thicknesses[..])
    }

    pub(crate) fn into_parts(self) -> (M, Vec<Layer<M, F>>, M) {
        (
            self.left_exterior,
            self.layers_left_to_right,
            self.right_exterior,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        ComplexScalar, Constant, Material, Sampled, material::MaterialHandle, stack::StackBuilder,
    };

    #[derive(Clone, Debug)]
    struct TestMaterial {
        relative_permittivity: f64,
    }

    impl Material for TestMaterial {
        type Real = f64;

        fn relative_permeability<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
        where
            I: Sampled<Elem = Self::Real>,
            C: ComplexScalar<RealField = Self::Real> + Copy,
        {
            vacuum_wavenumber.map(|_| C::one())
        }

        fn relative_permittivity<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
        where
            I: Sampled<Elem = Self::Real>,
            C: ComplexScalar<RealField = Self::Real> + Copy,
        {
            let epsilon = C::from_real(self.relative_permittivity);

            vacuum_wavenumber.map(|_| epsilon)
        }
    }

    #[test]
    fn material_handle_supports_heterogeneous_core_materials() {
        type C = num_complex::Complex64;
        type Handle = MaterialHandle<C>;

        let stack = StackBuilder::<Handle, f64>::from_materials(
            Constant::dielectric(1.0),
            TestMaterial {
                relative_permittivity: 2.25,
            },
        )
        .material_layer(Constant::dielectric(4.0), Thickness::centimetres(100.0))
        .material_layer(
            TestMaterial {
                relative_permittivity: 6.25,
            },
            Thickness::centimetres(200.0),
        )
        .finalise();

        assert_eq!(stack.layers_left_to_right().len(), 2);
    }
}
