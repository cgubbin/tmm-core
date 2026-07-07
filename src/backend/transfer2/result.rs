use crate::ComplexScalar;
use ndarray::{ArrayBase, Dimension, OwnedRepr, ScalarOperand};
use num_traits::One;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix2<C, D>
where
    D: Dimension,
{
    m11: ArrayBase<OwnedRepr<C>, D>,
    m12: ArrayBase<OwnedRepr<C>, D>,
    m21: ArrayBase<OwnedRepr<C>, D>,
    m22: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> Matrix2<C, D>
where
    D: Dimension,
{
    pub fn new(
        m11: ArrayBase<OwnedRepr<C>, D>,
        m12: ArrayBase<OwnedRepr<C>, D>,
        m21: ArrayBase<OwnedRepr<C>, D>,
        m22: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        Self { m11, m12, m21, m22 }
    }

    pub fn m11(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m11
    }
    pub fn m12(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m12
    }
    pub fn m21(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m21
    }
    pub fn m22(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m22
    }
    pub fn determinant(&self) -> ArrayBase<OwnedRepr<C>, D>
    where
        C: ComplexScalar,
    {
        self.m11.clone() * self.m22.view() - self.m12.clone() * self.m21.view()
    }

    pub fn zeros_like(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self
    where
        C: ComplexScalar,
    {
        let zero = shape_source.mapv(|_| C::zero());
        Self::new(zero.clone(), zero.clone(), zero.clone(), zero)
    }

    pub fn identity_like(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self
    where
        C: ComplexScalar,
    {
        let one = shape_source.mapv(|_| C::one());
        let zero = shape_source.mapv(|_| C::zero());
        Self::new(one.clone(), zero.clone(), zero, one)
    }

    pub fn add(&self, rhs: &Self) -> Self
    where
        C: ComplexScalar,
    {
        Self::new(
            self.m11.clone() + rhs.m11.view(),
            self.m12.clone() + rhs.m12.view(),
            self.m21.clone() + rhs.m21.view(),
            self.m22.clone() + rhs.m22.view(),
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DerivativeVariable {
    Frequency,
    FrequencySquared,
    PropagationConstant,
    PropagationConstantSquared,
    Thickness(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransferDerivatives<C, D>
where
    D: Dimension,
{
    first: Vec<(DerivativeVariable, Matrix2<C, D>)>,
    second: Vec<(DerivativeVariable, Matrix2<C, D>)>,
}

impl<C, D> TransferDerivatives<C, D>
where
    D: Dimension,
{
    pub fn new() -> Self {
        Self {
            first: Vec::new(),
            second: Vec::new(),
        }
    }

    pub fn push_first(&mut self, variable: DerivativeVariable, matrix: Matrix2<C, D>) {
        self.first.push((variable, matrix));
    }

    pub fn push_second(&mut self, variable: DerivativeVariable, matrix: Matrix2<C, D>) {
        self.second.push((variable, matrix));
    }

    pub fn first(&self, variable: DerivativeVariable) -> Option<&Matrix2<C, D>> {
        self.first
            .iter()
            .find_map(|(v, m)| (*v == variable).then_some(m))
    }

    pub fn second(&self, variable: DerivativeVariable) -> Option<&Matrix2<C, D>> {
        self.second
            .iter()
            .find_map(|(v, m)| (*v == variable).then_some(m))
    }
}

impl<C, D> Default for TransferDerivatives<C, D>
where
    D: Dimension,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransferResult<C, D>
where
    D: Dimension,
{
    matrix: Matrix2<C, D>,
    derivatives: TransferDerivatives<C, D>,
}

impl<C, D> TransferResult<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub fn new(matrix: Matrix2<C, D>) -> Self {
        Self {
            matrix,
            derivatives: TransferDerivatives::new(),
        }
    }

    pub fn with_derivatives(matrix: Matrix2<C, D>, derivatives: TransferDerivatives<C, D>) -> Self {
        Self {
            matrix,
            derivatives,
        }
    }

    pub fn matrix(&self) -> &Matrix2<C, D> {
        &self.matrix
    }

    pub fn derivatives(&self) -> &TransferDerivatives<C, D> {
        &self.derivatives
    }

    /// Complex reflection amplitude, assuming the standard convention:
    ///
    /// incident-side field = incident + reflected
    /// transmission-side field = transmitted only
    pub fn reflection_amplitude(&self) -> ArrayBase<OwnedRepr<C>, D> {
        -self.matrix.m21.clone() / self.matrix.m22.clone()
    }

    pub fn transmission_amplitude(&self) -> ArrayBase<OwnedRepr<C>, D> {
        let one = self.matrix.m22.mapv(|_| C::one());
        one / self.matrix.m22.clone()
    }

    pub fn reflectance(&self) -> ArrayBase<OwnedRepr<C::RealField>, D> {
        self.reflection_amplitude().mapv(|r| r.modulus_squared())
    }

    pub fn transmittance_unscaled(&self) -> ArrayBase<OwnedRepr<C::RealField>, D> {
        self.transmission_amplitude().mapv(|t| t.modulus_squared())
    }

    pub fn determinant(&self) -> ArrayBase<OwnedRepr<C>, D> {
        self.matrix.determinant()
    }

    pub fn determinant_derivative(
        &self,
        variable: DerivativeVariable,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>>
    where
        C: ScalarOperand,
    {
        let dm = self.derivatives.first(variable)?;

        Some(
            dm.m11.clone() * self.matrix.m22.view() + self.matrix.m11.clone() * dm.m22.view()
                - dm.m12.clone() * self.matrix.m21.view()
                - self.matrix.m12.clone() * dm.m21.view(),
        )
    }

    pub fn determinant_second_derivative(
        &self,
        variable: DerivativeVariable,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>>
    where
        C: ScalarOperand,
    {
        let dm = self.derivatives.first(variable)?;
        let ddm = self.derivatives.second(variable)?;

        let two = C::one() + C::one();

        Some(
            ddm.m11.clone() * self.matrix.m22.view()
                + dm.m11.mapv(|each| each * two) * dm.m22.view()
                + self.matrix.m11.clone() * ddm.m22.view()
                - ddm.m12.clone() * self.matrix.m21.view()
                - dm.m12.mapv(|each| each * two) * dm.m21.view()
                - self.matrix.m12.clone() * ddm.m21.view(),
        )
    }
}

impl<C, D> TransferResult<C, D>
where
    C: ComplexScalar,
    C::RealField: One,
    D: Dimension,
{
    pub fn reflection_amplitude_derivative(
        &self,
        variable: DerivativeVariable,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives.first(variable)?;

        let m21 = self.matrix.m21.clone();
        let m22 = self.matrix.m22.clone();

        let dm21 = dm.m21.clone();
        let dm22 = dm.m22.clone();

        Some(-(dm21 * m22.clone() - m21 * dm22) / m22.mapv(|x| x * x))
    }

    pub fn transmission_amplitude_derivative(
        &self,
        variable: DerivativeVariable,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives.first(variable)?;

        let m22 = self.matrix.m22.clone();
        let dm22 = dm.m22.clone();

        Some(-dm22 / m22.mapv(|x| x * x))
    }

    pub fn reflectance_derivative(
        &self,
        variable: DerivativeVariable,
    ) -> Option<ArrayBase<OwnedRepr<C::RealField>, D>> {
        let r = self.reflection_amplitude();
        let dr = self.reflection_amplitude_derivative(variable)?;

        Some((r.mapv(|x| x.conjugate()) * dr).mapv(|x| {
            let two = C::RealField::one() + C::RealField::one();
            two * x.real()
        }))
    }
}
