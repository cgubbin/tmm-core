mod builder;
mod layer;
mod thickness;
mod units;
mod validation;

pub use builder::StackBuilder;
pub use layer::Layer;
pub use thickness::Thickness;
pub use validation::ValidationConfig;

use either::Either;
use num_traits::Float;

use crate::{
    ComplexScalar, DifferentiableMaterial, DifferentiableMeromorphicMaterial, EvaluateMaterial,
    IncidentSide, MeromorphicMaterial,
    material::{
        AnalyticalMaterialHandle, DifferentiableMaterialHandle, Material, MaterialHandle,
        MeromorphicMaterialHandle, sample::Sampled,
    },
};

/// Heterogeneous stack supporting real-axis constitutive evaluation.
pub type MaterialStack<R, C, F = R> = Stack<MaterialHandle<R, C>, F>;

/// Heterogeneous stack supporting real-axis derivatives.
pub type DifferentiableMaterialStack<R, C, F = R> = Stack<DifferentiableMaterialHandle<R, C>, F>;

/// Heterogeneous stack supporting complex continuation.
pub type MeromorphicMaterialStack<R, C, F = R> = Stack<MeromorphicMaterialHandle<R, C>, F>;

/// Heterogeneous stack supporting complex continuation and derivatives.
pub type AnalyticalMaterialStack<R, C, F = R> = Stack<AnalyticalMaterialHandle<R, C>, F>;

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

impl<R, C, F> Stack<MaterialHandle<R, C>, F>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + Copy + 'static,
{
    pub fn from_materials<L, U>(
        left_exterior: L,
        right_exterior: U,
    ) -> StackBuilder<MaterialHandle<R, C>, F>
    where
        L: Material<Real = R> + Send + Sync + 'static,
        U: Material<Real = R> + Send + Sync + 'static,
        F: Float,
    {
        StackBuilder::from_materials(left_exterior, right_exterior)
    }
}

impl<R, C, F> Stack<AnalyticalMaterialHandle<R, C>, F>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + Copy + 'static,
{
    pub fn from_analytical_materials<L, U>(
        left_exterior: L,
        right_exterior: U,
    ) -> StackBuilder<AnalyticalMaterialHandle<R, C>, F>
    where
        L: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync + 'static,
        U: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync + 'static,
        F: Float,
    {
        StackBuilder::from_analytical_materials(left_exterior, right_exterior)
    }
}

impl<R, C, F> Stack<DifferentiableMaterialHandle<R, C>, F>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + Copy + 'static,
{
    pub fn from_differentiable_materials<L, U>(
        left_exterior: L,
        right_exterior: U,
    ) -> StackBuilder<DifferentiableMaterialHandle<R, C>, F>
    where
        L: DifferentiableMaterial<Real = R> + Send + Sync + 'static,
        U: DifferentiableMaterial<Real = R> + Send + Sync + 'static,
        F: Float,
    {
        StackBuilder::from_differentiable_materials(left_exterior, right_exterior)
    }
}

impl<R, C, F> Stack<MeromorphicMaterialHandle<R, C>, F>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + Copy + 'static,
{
    pub fn from_meromorphic_materials<L, U>(
        left_exterior: L,
        right_exterior: U,
    ) -> StackBuilder<MeromorphicMaterialHandle<R, C>, F>
    where
        L: MeromorphicMaterial<Real = R> + Send + Sync + 'static,
        U: MeromorphicMaterial<Real = R> + Send + Sync + 'static,
        F: Float,
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
}

#[cfg(test)]
mod test {
    use crate::{
        ComplexScalar, Constant, Material, Sampled, Thickness, material::MaterialHandle,
        stack::StackBuilder,
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
        type Handle = MaterialHandle<f64, C>;

        let stack = StackBuilder::<Handle, f64>::from_materials(
            Constant::dielectric(1.0),
            TestMaterial {
                relative_permittivity: 2.25,
            },
        )
        .material_layer(
            Constant::dielectric(4.0),
            Thickness::from_cm(100.0).unwrap(),
        )
        .material_layer(
            TestMaterial {
                relative_permittivity: 6.25,
            },
            Thickness::from_cm(200.0).unwrap(),
        )
        .build()
        .unwrap();

        assert_eq!(stack.layers_left_to_right().len(), 2);
    }
}
