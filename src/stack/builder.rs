use crate::{
    ComplexScalar, DifferentiableMaterial, DifferentiableMeromorphicMaterial, MeromorphicMaterial,
    material::{
        AnalyticalMaterialHandle, DifferentiableMaterialHandle, Material, MaterialHandle,
        MeromorphicMaterialHandle,
    },
    stack::ValidationConfig,
};

use super::{
    Layer, Stack, Thickness,
    validation::{StackValidator, ValidationError},
};

use num_traits::Float;

pub struct StackBuilder<M, F> {
    left_exterior: M,
    right_exterior: M,
    layers_left_to_right: Vec<Layer<M, F>>,
    validation: ValidationConfig<F>,
}

impl<M, F: Float> StackBuilder<M, F> {
    pub fn new(left_exterior: M, right_exterior: M) -> Self {
        Self {
            left_exterior,
            right_exterior,
            layers_left_to_right: Vec::new(),
            validation: ValidationConfig::default(),
        }
    }

    pub fn layer(mut self, material: M, thickness: Thickness<F>) -> Self {
        self.layers_left_to_right
            .push(Layer::new(material, thickness));
        self
    }

    pub fn push_layer(&mut self, material: M, thickness: Thickness<F>) {
        self.layers_left_to_right
            .push(Layer::new(material, thickness));
    }

    pub fn validation(mut self, validation: ValidationConfig<F>) -> Self {
        self.validation = validation;
        self
    }
}

impl<M, F> StackBuilder<M, F>
where
    F: Float + std::fmt::Debug + std::fmt::Display,
{
    pub fn build(self) -> Result<Stack<M, F>, ValidationError<F>> {
        let thicknesses: Vec<_> = self
            .layers_left_to_right
            .iter()
            .map(|l| l.thickness())
            .collect();
        let validator = StackValidator::new(self.validation);
        validator.validate_thicknesses(&thicknesses)?;

        Ok(Stack {
            left_exterior: self.left_exterior,
            right_exterior: self.right_exterior,
            layers_left_to_right: self.layers_left_to_right,
        })
    }
}

impl<R, C, F> StackBuilder<MaterialHandle<R, C>, F>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + Copy + 'static,
    F: Float,
{
    pub fn from_materials<L, U>(left_exterior: L, right_exterior: U) -> Self
    where
        L: Material<Real = R> + Send + Sync + 'static,
        U: Material<Real = R> + Send + Sync + 'static,
    {
        Self::new(
            MaterialHandle::new(left_exterior),
            MaterialHandle::new(right_exterior),
        )
    }

    pub fn material_layer<M>(self, material: M, thickness: Thickness<F>) -> Self
    where
        M: Material<Real = R> + Send + Sync + 'static,
    {
        self.layer(MaterialHandle::new(material), thickness)
    }

    pub fn push_material_layer<M>(&mut self, material: M, thickness: Thickness<F>)
    where
        M: Material<Real = R> + Send + Sync + 'static,
    {
        self.push_layer(MaterialHandle::new(material), thickness)
    }
}

impl<R, C, F> StackBuilder<AnalyticalMaterialHandle<R, C>, F>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + Copy + 'static,
    F: Float,
{
    pub fn from_analytical_materials<L, U>(left_exterior: L, right_exterior: U) -> Self
    where
        L: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync + 'static,
        U: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync + 'static,
    {
        Self::new(
            AnalyticalMaterialHandle::new(left_exterior),
            AnalyticalMaterialHandle::new(right_exterior),
        )
    }

    pub fn analytical_layer<M>(self, material: M, thickness: Thickness<F>) -> Self
    where
        M: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync + 'static,
    {
        self.layer(AnalyticalMaterialHandle::new(material), thickness)
    }

    pub fn push_analytical_layer<M>(&mut self, material: M, thickness: Thickness<F>)
    where
        M: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync + 'static,
    {
        self.push_layer(AnalyticalMaterialHandle::new(material), thickness)
    }
}

impl<R, C, F> StackBuilder<DifferentiableMaterialHandle<R, C>, F>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + Copy + 'static,
    F: Float,
{
    /// Create a builder whose materials support real-axis derivatives.
    pub fn from_differentiable_materials<L, U>(left_exterior: L, right_exterior: U) -> Self
    where
        L: DifferentiableMaterial<Real = R> + Send + Sync + 'static,
        U: DifferentiableMaterial<Real = R> + Send + Sync + 'static,
    {
        Self::new(
            DifferentiableMaterialHandle::new(left_exterior),
            DifferentiableMaterialHandle::new(right_exterior),
        )
    }

    /// Add a finite layer whose material supports real-axis derivatives.
    pub fn differentiable_layer<M>(self, material: M, thickness: Thickness<F>) -> Self
    where
        M: DifferentiableMaterial<Real = R> + Send + Sync + 'static,
    {
        self.layer(DifferentiableMaterialHandle::new(material), thickness)
    }

    /// Add a finite differentiable layer in place.
    pub fn push_differentiable_layer<M>(&mut self, material: M, thickness: Thickness<F>)
    where
        M: DifferentiableMaterial<Real = R> + Send + Sync + 'static,
    {
        self.push_layer(DifferentiableMaterialHandle::new(material), thickness)
    }
}

impl<R, C, F> StackBuilder<MeromorphicMaterialHandle<R, C>, F>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + Copy + 'static,
    F: Float,
{
    /// Create a builder whose materials support complex-frequency
    /// continuation.
    pub fn from_meromorphic_materials<L, U>(left_exterior: L, right_exterior: U) -> Self
    where
        L: MeromorphicMaterial<Real = R> + Send + Sync + 'static,
        U: MeromorphicMaterial<Real = R> + Send + Sync + 'static,
    {
        Self::new(
            MeromorphicMaterialHandle::new(left_exterior),
            MeromorphicMaterialHandle::new(right_exterior),
        )
    }

    /// Add a finite layer whose constitutive response supports
    /// complex-frequency continuation.
    pub fn meromorphic_layer<M>(self, material: M, thickness: Thickness<F>) -> Self
    where
        M: MeromorphicMaterial<Real = R> + Send + Sync + 'static,
    {
        self.layer(MeromorphicMaterialHandle::new(material), thickness)
    }

    /// Add a finite meromorphic layer in place.
    pub fn push_meromorphic_layer<M>(&mut self, material: M, thickness: Thickness<F>)
    where
        M: MeromorphicMaterial<Real = R> + Send + Sync + 'static,
    {
        self.push_layer(MeromorphicMaterialHandle::new(material), thickness)
    }
}
