mod builder;
mod layer;

pub use builder::StackBuilder;
pub use layer::Layer;

use either::Either;
use nalgebra::ComplexField;
use num_traits::Float;

use crate::{
    ComplexScalar, DifferentiableMaterial, DifferentiableMeromorphicMaterial, EvaluateMaterial,
    IncidentSide, MeromorphicMaterial,
    material::{
        AnalyticalMaterialHandle, DifferentiableMaterialHandle, Material, MaterialHandle,
        MeromorphicMaterialHandle, Sampled,
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

/// A planar stack bounded by two semi-infinite exterior media.
///
/// Finite layers are stored in geometric left-to-right order. The exterior
/// media are not included in `len()` or layer iteration.
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

    pub fn builder(left_exterior: M, right_exterior: M) -> StackBuilder<M, F> {
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
    /// Return the left semi-infinite exterior material.
    pub fn left_exterior(&self) -> &M {
        &self.left_exterior
    }

    /// Return the right semi-infinite exterior material.
    pub fn right_exterior(&self) -> &M {
        &self.right_exterior
    }

    /// Return the finite layers in geometric left-to-right order.
    pub fn layers_left_to_right(&self) -> &[Layer<M, F>] {
        &self.layers_left_to_right
    }

    /// Return the number of finite layers.
    ///
    /// The two exterior media are not included.
    pub fn len(&self) -> usize {
        self.layers_left_to_right.len()
    }

    /// Return `true` when the stack contains no finite layers.
    ///
    /// The exterior media do not affect this result.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over finite layers in geometric left-to-right order.
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

    /// Evaluate the relative permittivity of the incident exterior medium.
    ///
    /// `vacuum_angular_wavenumber` is the vacuum angular wavenumber `k₀`,
    /// expressed in inverse centimetres.
    ///
    /// The exterior medium is selected from `side`.
    pub(crate) fn incident_relative_permittivity<I, C>(
        &self,
        vacuum_angular_wavenumber: I,
        side: IncidentSide,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = M::Real> + Copy,
        I: Sampled<Elem = M::Real>,
        M: EvaluateMaterial<C>,
    {
        self.incident_exterior(side)
            .evaluate_relative_permittivity(vacuum_angular_wavenumber)
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
mod tests {
    use lamina_units::Length;
    use num_complex::Complex64;

    use super::*;
    use crate::{ComplexScalar, Constant, Material, Sampled, material::MaterialHandle};

    #[derive(Clone, Debug, PartialEq)]
    struct TestMaterial {
        id: usize,
        relative_permittivity: f64,
    }

    impl Material for TestMaterial {
        type Real = f64;

        fn relative_permeability<I, C>(&self, vacuum_angular_wavenumber: I) -> I::Mapped<C>
        where
            I: Sampled<Elem = Self::Real>,
            C: ComplexScalar<RealField = Self::Real>,
        {
            vacuum_angular_wavenumber.map(|_| C::one())
        }

        fn relative_permittivity<I, C>(&self, vacuum_angular_wavenumber: I) -> I::Mapped<C>
        where
            I: Sampled<Elem = Self::Real>,
            C: ComplexScalar<RealField = Self::Real>,
        {
            let epsilon = C::from_real(self.relative_permittivity);

            vacuum_angular_wavenumber.map(|_| epsilon)
        }
    }

    fn test_stack() -> Stack<TestMaterial, f64> {
        Stack::builder(
            TestMaterial {
                id: 0,
                relative_permittivity: 1.0,
            },
            TestMaterial {
                id: 3,
                relative_permittivity: 4.0,
            },
        )
        .layer(
            TestMaterial {
                id: 1,
                relative_permittivity: 2.0,
            },
            Length::nanometres(100.0),
        )
        .layer(
            TestMaterial {
                id: 2,
                relative_permittivity: 3.0,
            },
            Length::micrometres(2.0),
        )
        .finalise()
    }

    #[test]
    fn builder_preserves_exterior_media() {
        let stack = test_stack();

        assert_eq!(stack.left_exterior().id, 0);
        assert_eq!(stack.right_exterior().id, 3);
    }

    #[test]
    fn builder_preserves_layer_order() {
        let stack = test_stack();

        let ids = stack
            .layers_left_to_right()
            .iter()
            .map(|layer| layer.material().id)
            .collect::<Vec<_>>();

        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn len_and_is_empty_describe_only_finite_layers() {
        let stack = test_stack();

        assert_eq!(stack.len(), 2);
        assert!(!stack.is_empty());

        let empty = Stack::<TestMaterial, f64>::builder(
            TestMaterial {
                id: 0,
                relative_permittivity: 1.0,
            },
            TestMaterial {
                id: 1,
                relative_permittivity: 1.0,
            },
        )
        .finalise();

        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn iter_follows_left_to_right_order() {
        let stack = test_stack();

        let ids = stack
            .iter()
            .map(|layer| layer.material().id)
            .collect::<Vec<_>>();

        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn layers_in_direction_respects_propagation_direction() {
        let stack = test_stack();

        let left_to_right = stack
            .layers_in_direction(PropagationDirection::LeftToRight)
            .map(|layer| layer.material().id)
            .collect::<Vec<_>>();

        let right_to_left = stack
            .layers_in_direction(PropagationDirection::RightToLeft)
            .map(|layer| layer.material().id)
            .collect::<Vec<_>>();

        assert_eq!(left_to_right, vec![1, 2]);
        assert_eq!(right_to_left, vec![2, 1]);
    }

    #[test]
    fn entrance_and_exit_exteriors_follow_direction() {
        let stack = test_stack();

        assert_eq!(
            stack
                .entrance_exterior(PropagationDirection::LeftToRight,)
                .id,
            0,
        );

        assert_eq!(
            stack.exit_exterior(PropagationDirection::LeftToRight,).id,
            3,
        );

        assert_eq!(
            stack
                .entrance_exterior(PropagationDirection::RightToLeft,)
                .id,
            3,
        );

        assert_eq!(
            stack.exit_exterior(PropagationDirection::RightToLeft,).id,
            0,
        );
    }

    #[test]
    fn incident_exterior_selects_requested_side() {
        let stack = test_stack();

        assert_eq!(stack.incident_exterior(IncidentSide::Left).id, 0,);

        assert_eq!(stack.incident_exterior(IncidentSide::Right).id, 3,);
    }

    #[test]
    fn layer_preserves_material_and_thickness() {
        let layer = Layer::new(
            TestMaterial {
                id: 7,
                relative_permittivity: 2.5,
            },
            Length::nanometres(350.0),
        );

        assert_eq!(layer.material().id, 7);
        assert_eq!(layer.thickness(), Length::nanometres(350.0),);

        let (material, thickness) = layer.into_parts();

        assert_eq!(material.id, 7);
        assert_eq!(thickness, Length::nanometres(350.0),);
    }

    #[test]
    fn incident_relative_permittivity_uses_requested_exterior() {
        let stack = test_stack();

        let k0 = crate::material::Scalar::new(1000.0);

        let left: Complex64 = stack.incident_relative_permittivity(k0, IncidentSide::Left);

        let right: Complex64 = stack.incident_relative_permittivity(
            crate::material::Scalar::new(1000.0),
            IncidentSide::Right,
        );

        assert_eq!(left, Complex64::new(1.0, 0.0));
        assert_eq!(right, Complex64::new(4.0, 0.0));
    }

    #[test]
    fn material_handle_supports_heterogeneous_core_materials() {
        type C = Complex64;
        type Handle = MaterialHandle<C>;

        let stack = StackBuilder::<Handle, f64>::from_materials(
            Constant::dielectric(1.0),
            TestMaterial {
                id: 10,
                relative_permittivity: 2.25,
            },
        )
        .material_layer(Constant::dielectric(4.0), Length::centimetres(100.0))
        .material_layer(
            TestMaterial {
                id: 11,
                relative_permittivity: 6.25,
            },
            Length::centimetres(200.0),
        )
        .finalise();

        assert_eq!(stack.len(), 2);
    }

    #[test]
    fn into_parts_preserves_stack_order() {
        let stack = test_stack();

        let (left, layers, right) = stack.into_parts();

        assert_eq!(left.id, 0);
        assert_eq!(right.id, 3);

        assert_eq!(
            layers
                .iter()
                .map(|layer| layer.material().id)
                .collect::<Vec<_>>(),
            vec![1, 2],
        );
    }
}
