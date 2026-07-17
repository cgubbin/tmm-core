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
    ComplexScalar, DifferentiableMaterial, DifferentiableMeromorphicMaterial, IncidentSide,
    MeromorphicMaterial,
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
    pub fn from_analytical_materials<L, U>(
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
        M: Material,
    {
        let direction = side.propagation_direction();
        let material = self.entrance_exterior(direction);

        material.relative_permittivity(vacuum_wavenumber)
    }
}
