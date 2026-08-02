use crate::algebra::{
    ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet0, Jet1, Jet2,
    JetBivariate1, JetBivariate2, RealParameter, ScalarAlgebra,
};

use crate::field::VectorField;

use nalgebra::ComplexField;
use ndarray::Dimension;
use std::fmt::Debug;

pub trait CartesianScalarAlgebra: ScalarAlgebra {
    type Vector: CartesianVectorAlgebra<Coefficient = Self::Scalar, ScalarField = Self>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector;
}

pub trait CartesianVectorAlgebra: Clone + Sized {
    type Coefficient;
    type ScalarField;

    fn cross(&self, rhs: &Self) -> Self;

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self;

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self;
}

/// Cartesian vector operations involving complex conjugation.
///
/// Implemented only for plain fields and jets whose differentiation
/// parameters are real. These operations are not holomorphic and must not be
/// applied to complex-variable derivative jets.
pub trait RealCartesianVectorAlgebra: CartesianVectorAlgebra {
    type RealVector;
    type RealScalarField;

    fn conjugated(&self) -> Self;

    fn real(&self) -> Self::RealVector;

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField;

    fn magnitude_squared(&self) -> Self::RealScalarField {
        Self::scalar_real(self.hermitian_dot(self))
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField;
}

impl<T, D, P> CartesianScalarAlgebra for ArrayJet0<T, D, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Vector = Jet0<VectorField<T, D>, P>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        Jet0::new(VectorField::new_unchecked(
            x.into_inner(),
            y.into_inner(),
            z.into_inner(),
        ))
    }
}

impl<T, D, P> CartesianScalarAlgebra for ArrayJet1<T, D, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Vector = Jet1<VectorField<T, D>, P>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        let (x_value, x_first) = x.into_parts();
        let (y_value, y_first) = y.into_parts();
        let (z_value, z_first) = z.into_parts();

        Jet1::from_parts(
            VectorField::new_unchecked(x_value, y_value, z_value),
            VectorField::new_unchecked(x_first, y_first, z_first),
        )
    }
}

impl<T, D, P> CartesianScalarAlgebra for ArrayJet2<T, D, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Vector = Jet2<VectorField<T, D>, P>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        let (x_value, x_first, x_second) = x.into_parts();
        let (y_value, y_first, y_second) = y.into_parts();
        let (z_value, z_first, z_second) = z.into_parts();

        Jet2::from_parts(
            VectorField::new_unchecked(x_value, y_value, z_value),
            VectorField::new_unchecked(x_first, y_first, z_first),
            VectorField::new_unchecked(x_second, y_second, z_second),
        )
    }
}

impl<T, D, P> CartesianScalarAlgebra for ArrayJetBivariate1<T, D, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Vector = JetBivariate1<VectorField<T, D>, P>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        let (x_value, x_gradient) = x.into_parts();
        let (y_value, y_gradient) = y.into_parts();
        let (z_value, z_gradient) = z.into_parts();

        let (x_x, x_y) = x_gradient.into_parts();
        let (y_x, y_y) = y_gradient.into_parts();
        let (z_x, z_y) = z_gradient.into_parts();

        JetBivariate1::from_components(
            VectorField::new_unchecked(x_value, y_value, z_value),
            VectorField::new_unchecked(x_x, y_x, z_x),
            VectorField::new_unchecked(x_y, y_y, z_y),
        )
    }
}

impl<T, D, P> CartesianScalarAlgebra for ArrayJetBivariate2<T, D, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Vector = JetBivariate2<VectorField<T, D>, P>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        let (x_value, x_gradient, x_hessian) = x.into_parts();
        let (y_value, y_gradient, y_hessian) = y.into_parts();
        let (z_value, z_gradient, z_hessian) = z.into_parts();

        let (x_x, x_y) = x_gradient.into_parts();
        let (y_x, y_y) = y_gradient.into_parts();
        let (z_x, z_y) = z_gradient.into_parts();

        let (x_xx, x_xy, x_yy) = x_hessian.into_parts();
        let (y_xx, y_xy, y_yy) = y_hessian.into_parts();
        let (z_xx, z_xy, z_yy) = z_hessian.into_parts();

        JetBivariate2::from_components(
            VectorField::new_unchecked(x_value, y_value, z_value),
            VectorField::new_unchecked(x_x, y_x, z_x),
            VectorField::new_unchecked(x_y, y_y, z_y),
            VectorField::new_unchecked(x_xx, y_xx, z_xx),
            VectorField::new_unchecked(x_xy, y_xy, z_xy),
            VectorField::new_unchecked(x_yy, y_yy, z_yy),
        )
    }
}

impl<T, D, P> CartesianVectorAlgebra for Jet0<VectorField<T, D>, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Coefficient = T;
    type ScalarField = ArrayJet0<T, D, P>;

    fn cross(&self, rhs: &Self) -> Self {
        Jet0::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        Jet0::scale_by(self, factor)
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        Jet0::multiply_by_scalar(self, factor)
    }
}

impl<T, D> RealCartesianVectorAlgebra for Jet0<VectorField<T, D>, RealParameter>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = Jet0<VectorField<T::RealField, D>, RealParameter>;
    type RealScalarField = ArrayJet0<T::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        Jet0::conjugated(self)
    }

    fn real(&self) -> Self::RealVector {
        Jet0::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        Jet0::hermitian_dot_product(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        Jet0::real(&value)
    }
}

impl<T, D, P> CartesianVectorAlgebra for Jet1<VectorField<T, D>, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Coefficient = T;
    type ScalarField = ArrayJet1<T, D, P>;

    fn cross(&self, rhs: &Self) -> Self {
        Jet1::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        Jet1::scale_by(self, factor)
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        Jet1::multiply_by_scalar(self, factor)
    }
}

impl<T, D> RealCartesianVectorAlgebra for Jet1<VectorField<T, D>, RealParameter>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = Jet1<VectorField<T::RealField, D>, RealParameter>;
    type RealScalarField = ArrayJet1<T::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        Jet1::conjugated(self)
    }

    fn real(&self) -> Self::RealVector {
        Jet1::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        Jet1::hermitian_dot_product(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        Jet1::real(&value)
    }
}

impl<T, D, P> CartesianVectorAlgebra for Jet2<VectorField<T, D>, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Coefficient = T;
    type ScalarField = ArrayJet2<T, D, P>;

    fn cross(&self, rhs: &Self) -> Self {
        Jet2::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        Jet2::scale_by(self, factor)
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        Jet2::multiply_by_scalar(self, factor)
    }
}

impl<T, D> RealCartesianVectorAlgebra for Jet2<VectorField<T, D>, RealParameter>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = Jet2<VectorField<T::RealField, D>, RealParameter>;
    type RealScalarField = ArrayJet2<T::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        Jet2::conjugated(self)
    }

    fn real(&self) -> Self::RealVector {
        Jet2::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        Jet2::hermitian_dot_product(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        Jet2::real(&value)
    }
}

impl<T, D, P> CartesianVectorAlgebra for JetBivariate1<VectorField<T, D>, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Coefficient = T;
    type ScalarField = ArrayJetBivariate1<T, D, P>;

    fn cross(&self, rhs: &Self) -> Self {
        JetBivariate1::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        JetBivariate1::scale_by(self, factor)
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        JetBivariate1::multiply_by_scalar(self, factor)
    }
}

impl<T, D> RealCartesianVectorAlgebra for JetBivariate1<VectorField<T, D>, RealParameter>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = JetBivariate1<VectorField<T::RealField, D>, RealParameter>;
    type RealScalarField = ArrayJetBivariate1<T::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        JetBivariate1::conjugated(self)
    }

    fn real(&self) -> Self::RealVector {
        JetBivariate1::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        JetBivariate1::hermitian_dot_product(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        JetBivariate1::real(&value)
    }
}

impl<T, D, P> CartesianVectorAlgebra for JetBivariate2<VectorField<T, D>, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Coefficient = T;
    type ScalarField = ArrayJetBivariate2<T, D, P>;

    fn cross(&self, rhs: &Self) -> Self {
        JetBivariate2::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        JetBivariate2::scale_by(self, factor)
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        JetBivariate2::multiply_by_scalar(self, factor)
    }
}

impl<T, D> RealCartesianVectorAlgebra for JetBivariate2<VectorField<T, D>, RealParameter>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = JetBivariate2<VectorField<T::RealField, D>, RealParameter>;
    type RealScalarField = ArrayJetBivariate2<T::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        JetBivariate2::conjugated(self)
    }

    fn real(&self) -> Self::RealVector {
        JetBivariate2::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        JetBivariate2::hermitian_dot_product(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        JetBivariate2::real(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;

    use crate::algebra::{Jet1, Jet2, JetBivariate2, RealParameter};

    type C = Complex64;
    type D = Ix1;
    type Scalar = ArrayJet0<C, D, RealParameter>;

    type FirstScalar = ArrayJet1<C, D, RealParameter>;

    type SecondScalar = ArrayJet2<C, D, RealParameter>;

    type BivariateScalar = ArrayJetBivariate2<C, D, RealParameter>;

    type Vector = Jet0<VectorField<C, D>, RealParameter>;

    type FirstVector = Jet1<VectorField<C, D>, RealParameter>;

    type SecondVector = Jet2<VectorField<C, D>, RealParameter>;

    type BivariateVector = JetBivariate2<VectorField<C, D>, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn assert_complex_close(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_array_close(actual: &Array1<C>, expected: &Array1<C>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim(),);

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }

    fn assert_vector_close(actual: &VectorField<C, D>, expected: &VectorField<C, D>) {
        assert_complex_array_close(actual.x(), expected.x());

        assert_complex_array_close(actual.y(), expected.y());

        assert_complex_array_close(actual.z(), expected.z());
    }

    fn vector(x: C, y: C, z: C) -> VectorField<C, D> {
        VectorField::new_unchecked(arr1(&[x]), arr1(&[y]), arr1(&[z]))
    }

    fn scale_vector(
        vector: &VectorField<C, D>,
        scalar: &ndarray::Array<C, D>,
    ) -> VectorField<C, D> {
        vector.clone() * scalar
    }

    fn add_vectors(terms: &[VectorField<C, D>]) -> VectorField<C, D> {
        let mut result = VectorField::zeros_like(terms[0].x());

        for term in terms {
            result = result + term;
        }

        result
    }

    // ------------------------------------------------------------------
    // Trait relationships
    // ------------------------------------------------------------------

    fn assert_cartesian_scalar<S>()
    where
        S: CartesianScalarAlgebra,
    {
    }

    fn assert_cartesian_vector<V>()
    where
        V: CartesianVectorAlgebra<Coefficient = C>,
    {
    }

    fn assert_real_cartesian_vector<V>()
    where
        V: RealCartesianVectorAlgebra,
    {
    }

    #[test]
    fn all_scalar_representations_support_cartesian_assembly() {
        assert_cartesian_scalar::<Scalar>();
        assert_cartesian_scalar::<FirstScalar>();
        assert_cartesian_scalar::<SecondScalar>();
        assert_cartesian_scalar::<BivariateScalar>();
    }

    #[test]
    fn all_vector_representations_support_cartesian_algebra() {
        assert_cartesian_vector::<Vector>();
        assert_cartesian_vector::<FirstVector>();
        assert_cartesian_vector::<SecondVector>();
        assert_cartesian_vector::<BivariateVector>();
    }

    #[test]
    fn real_parameter_vectors_support_real_cartesian_algebra() {
        assert_real_cartesian_vector::<Vector>();
        assert_real_cartesian_vector::<FirstVector>();
        assert_real_cartesian_vector::<SecondVector>();
        assert_real_cartesian_vector::<BivariateVector>();
    }

    // ------------------------------------------------------------------
    // CartesianScalarAlgebra
    // ------------------------------------------------------------------

    #[test]
    fn arrays_are_assembled_into_cartesian_vector() {
        let x = arr1(&[c(1.0, 2.0), c(3.0, 4.0)]);

        let y = arr1(&[c(5.0, 6.0), c(7.0, 8.0)]);

        let z = arr1(&[c(9.0, 10.0), c(11.0, 12.0)]);

        let result = <Scalar as CartesianScalarAlgebra>::into_cartesian_vector(
            Jet0::new(x.clone()),
            Jet0::new(y.clone()),
            Jet0::new(z.clone()),
        );

        assert_eq!(result.x(), &x);
        assert_eq!(result.y(), &y);
        assert_eq!(result.z(), &z);
    }

    #[test]
    fn first_order_scalar_jets_are_transposed_into_vector_jet() {
        let x = Jet1::from_parts(arr1(&[c(1.0, 0.0)]), arr1(&[c(2.0, 0.0)]));

        let y = Jet1::from_parts(arr1(&[c(3.0, 0.0)]), arr1(&[c(4.0, 0.0)]));

        let z = Jet1::from_parts(arr1(&[c(5.0, 0.0)]), arr1(&[c(6.0, 0.0)]));

        let result = <FirstScalar as CartesianScalarAlgebra>::into_cartesian_vector(x, y, z);

        assert_vector_close(
            result.value(),
            &vector(c(1.0, 0.0), c(3.0, 0.0), c(5.0, 0.0)),
        );

        assert_vector_close(
            result.first(),
            &vector(c(2.0, 0.0), c(4.0, 0.0), c(6.0, 0.0)),
        );
    }

    #[test]
    fn second_order_scalar_jets_are_transposed_into_vector_jet() {
        let x = Jet2::from_parts(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0)]),
            arr1(&[c(3.0, 0.0)]),
        );

        let y = Jet2::from_parts(
            arr1(&[c(4.0, 0.0)]),
            arr1(&[c(5.0, 0.0)]),
            arr1(&[c(6.0, 0.0)]),
        );

        let z = Jet2::from_parts(
            arr1(&[c(7.0, 0.0)]),
            arr1(&[c(8.0, 0.0)]),
            arr1(&[c(9.0, 0.0)]),
        );

        let result = <SecondScalar as CartesianScalarAlgebra>::into_cartesian_vector(x, y, z);

        assert_vector_close(
            result.value(),
            &vector(c(1.0, 0.0), c(4.0, 0.0), c(7.0, 0.0)),
        );

        assert_vector_close(
            result.first(),
            &vector(c(2.0, 0.0), c(5.0, 0.0), c(8.0, 0.0)),
        );

        assert_vector_close(
            result.second(),
            &vector(c(3.0, 0.0), c(6.0, 0.0), c(9.0, 0.0)),
        );
    }

    #[test]
    fn bivariate_scalar_jets_are_transposed_into_vector_jet() {
        let x = JetBivariate2::from_components(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0)]),
            arr1(&[c(3.0, 0.0)]),
            arr1(&[c(4.0, 0.0)]),
            arr1(&[c(5.0, 0.0)]),
            arr1(&[c(6.0, 0.0)]),
        );

        let y = JetBivariate2::from_components(
            arr1(&[c(7.0, 0.0)]),
            arr1(&[c(8.0, 0.0)]),
            arr1(&[c(9.0, 0.0)]),
            arr1(&[c(10.0, 0.0)]),
            arr1(&[c(11.0, 0.0)]),
            arr1(&[c(12.0, 0.0)]),
        );

        let z = JetBivariate2::from_components(
            arr1(&[c(13.0, 0.0)]),
            arr1(&[c(14.0, 0.0)]),
            arr1(&[c(15.0, 0.0)]),
            arr1(&[c(16.0, 0.0)]),
            arr1(&[c(17.0, 0.0)]),
            arr1(&[c(18.0, 0.0)]),
        );

        let result = <BivariateScalar as CartesianScalarAlgebra>::into_cartesian_vector(x, y, z);

        assert_vector_close(
            result.value(),
            &vector(c(1.0, 0.0), c(7.0, 0.0), c(13.0, 0.0)),
        );

        assert_vector_close(
            result.axis0(),
            &vector(c(2.0, 0.0), c(8.0, 0.0), c(14.0, 0.0)),
        );

        assert_vector_close(
            result.axis1(),
            &vector(c(3.0, 0.0), c(9.0, 0.0), c(15.0, 0.0)),
        );

        assert_vector_close(
            result.axis0_axis0(),
            &vector(c(4.0, 0.0), c(10.0, 0.0), c(16.0, 0.0)),
        );

        assert_vector_close(
            result.axis0_axis1(),
            &vector(c(5.0, 0.0), c(11.0, 0.0), c(17.0, 0.0)),
        );

        assert_vector_close(
            result.axis1_axis1(),
            &vector(c(6.0, 0.0), c(12.0, 0.0), c(18.0, 0.0)),
        );
    }

    // ------------------------------------------------------------------
    // Constant scaling
    // ------------------------------------------------------------------

    #[test]
    fn zero_order_vector_constant_scaling_scales_every_jet_component() {
        let source = Jet0::new(vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)));

        let factor = c(2.0, 0.0);

        let result = <Vector as CartesianVectorAlgebra>::scale_by_constant(&source, factor);

        assert_vector_close(result.value(), &(source.value().clone() * factor));
    }

    #[test]
    fn first_order_vector_constant_scaling_scales_every_jet_component() {
        let source = Jet1::from_parts(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
        );

        let factor = c(2.0, 0.0);

        let result = <FirstVector as CartesianVectorAlgebra>::scale_by_constant(&source, factor);

        assert_vector_close(result.value(), &(source.value().clone() * factor));

        assert_vector_close(result.first(), &(source.first().clone() * factor));
    }

    #[test]
    fn second_order_vector_constant_scaling_scales_every_jet_component() {
        let source = Jet2::from_parts(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
            vector(c(7.0, 0.0), c(8.0, 0.0), c(9.0, 0.0)),
        );

        let factor = c(2.0, 0.0);

        let result = <SecondVector as CartesianVectorAlgebra>::scale_by_constant(&source, factor);

        assert_vector_close(result.value(), &(source.value().clone() * factor));

        assert_vector_close(result.first(), &(source.first().clone() * factor));

        assert_vector_close(result.second(), &(source.second().clone() * factor));
    }

    #[test]
    fn bivariate_vector_constant_scaling_scales_every_jet_component() {
        let source = JetBivariate2::from_components(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
            vector(c(7.0, 0.0), c(8.0, 0.0), c(9.0, 0.0)),
            vector(c(10.0, 0.0), c(11.0, 0.0), c(12.0, 0.0)),
            vector(c(13.0, 0.0), c(14.0, 0.0), c(15.0, 0.0)),
            vector(c(16.0, 0.0), c(17.0, 0.0), c(18.0, 0.0)),
        );

        let factor = c(2.0, 0.0);

        let result =
            <BivariateVector as CartesianVectorAlgebra>::scale_by_constant(&source, factor);

        assert_vector_close(result.value(), &(source.value().clone() * factor));

        assert_vector_close(result.axis0(), &(source.axis0().clone() * factor));

        assert_vector_close(result.axis1(), &(source.axis1().clone() * factor));

        assert_vector_close(
            result.axis0_axis0(),
            &(source.axis0_axis0().clone() * factor),
        );

        assert_vector_close(
            result.axis0_axis1(),
            &(source.axis0_axis1().clone() * factor),
        );

        assert_vector_close(
            result.axis1_axis1(),
            &(source.axis1_axis1().clone() * factor),
        );
    }

    // ------------------------------------------------------------------
    // Vector-scalar products
    // ------------------------------------------------------------------

    #[test]
    fn zero_order_vector_scalar_product_obeys_product_rule() {
        let vector = Jet0::new(vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)));

        let scalar = Jet0::new(arr1(&[c(7.0, 0.0)]));

        let result = <Vector as CartesianVectorAlgebra>::multiply_by_scalar(&vector, &scalar);

        let expected_value = scale_vector(vector.value(), scalar.value());

        assert_vector_close(result.value(), &expected_value);
    }

    #[test]
    fn first_order_vector_scalar_product_obeys_product_rule() {
        let vector = Jet1::from_parts(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
        );

        let scalar = Jet1::from_parts(arr1(&[c(7.0, 0.0)]), arr1(&[c(8.0, 0.0)]));

        let result = <FirstVector as CartesianVectorAlgebra>::multiply_by_scalar(&vector, &scalar);

        let expected_value = scale_vector(vector.value(), scalar.value());

        let expected_first = add_vectors(&[
            scale_vector(vector.first(), scalar.value()),
            scale_vector(vector.value(), scalar.first()),
        ]);

        assert_vector_close(result.value(), &expected_value);

        assert_vector_close(result.first(), &expected_first);
    }

    #[test]
    fn second_order_vector_scalar_product_obeys_product_rule() {
        let vector = Jet2::from_parts(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
            vector(c(7.0, 0.0), c(8.0, 0.0), c(9.0, 0.0)),
        );

        let scalar = Jet2::from_parts(
            arr1(&[c(10.0, 0.0)]),
            arr1(&[c(11.0, 0.0)]),
            arr1(&[c(12.0, 0.0)]),
        );

        let result = <SecondVector as CartesianVectorAlgebra>::multiply_by_scalar(&vector, &scalar);

        let expected_value = scale_vector(vector.value(), scalar.value());

        let expected_first = add_vectors(&[
            scale_vector(vector.first(), scalar.value()),
            scale_vector(vector.value(), scalar.first()),
        ]);

        let expected_second = add_vectors(&[
            scale_vector(vector.second(), scalar.value()),
            scale_vector(vector.first(), scalar.first()) * c(2.0, 0.0),
            scale_vector(vector.value(), scalar.second()),
        ]);

        assert_vector_close(result.value(), &expected_value);

        assert_vector_close(result.first(), &expected_first);

        assert_vector_close(result.second(), &expected_second);
    }

    #[test]
    fn bivariate_vector_scalar_product_obeys_all_product_rules() {
        let vector = JetBivariate2::from_components(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
            vector(c(7.0, 0.0), c(8.0, 0.0), c(9.0, 0.0)),
            vector(c(10.0, 0.0), c(11.0, 0.0), c(12.0, 0.0)),
            vector(c(13.0, 0.0), c(14.0, 0.0), c(15.0, 0.0)),
            vector(c(16.0, 0.0), c(17.0, 0.0), c(18.0, 0.0)),
        );

        let scalar = JetBivariate2::from_components(
            arr1(&[c(19.0, 0.0)]),
            arr1(&[c(20.0, 0.0)]),
            arr1(&[c(21.0, 0.0)]),
            arr1(&[c(22.0, 0.0)]),
            arr1(&[c(23.0, 0.0)]),
            arr1(&[c(24.0, 0.0)]),
        );

        let result =
            <BivariateVector as CartesianVectorAlgebra>::multiply_by_scalar(&vector, &scalar);

        let expected_value = scale_vector(vector.value(), scalar.value());

        let expected_x = add_vectors(&[
            scale_vector(vector.axis0(), scalar.value()),
            scale_vector(vector.value(), scalar.axis0()),
        ]);

        let expected_y = add_vectors(&[
            scale_vector(vector.axis1(), scalar.value()),
            scale_vector(vector.value(), scalar.axis1()),
        ]);

        let expected_xx = add_vectors(&[
            scale_vector(vector.axis0_axis0(), scalar.value()),
            scale_vector(vector.axis0(), scalar.axis0()) * c(2.0, 0.0),
            scale_vector(vector.value(), scalar.axis0_axis0()),
        ]);

        let expected_xy = add_vectors(&[
            scale_vector(vector.axis0_axis1(), scalar.value()),
            scale_vector(vector.axis0(), scalar.axis1()),
            scale_vector(vector.axis1(), scalar.axis0()),
            scale_vector(vector.value(), scalar.axis0_axis1()),
        ]);

        let expected_yy = add_vectors(&[
            scale_vector(vector.axis1_axis1(), scalar.value()),
            scale_vector(vector.axis1(), scalar.axis1()) * c(2.0, 0.0),
            scale_vector(vector.value(), scalar.axis1_axis1()),
        ]);

        assert_vector_close(result.value(), &expected_value);

        assert_vector_close(result.axis0(), &expected_x);

        assert_vector_close(result.axis1(), &expected_y);

        assert_vector_close(result.axis0_axis0(), &expected_xx);

        assert_vector_close(result.axis0_axis1(), &expected_xy);

        assert_vector_close(result.axis1_axis1(), &expected_yy);
    }

    // ------------------------------------------------------------------
    // Cross products
    // ------------------------------------------------------------------

    #[test]
    fn cross_delegates_for_plain_vectors() {
        let lhs = Vector::new(vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)));

        let rhs = Vector::new(vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)));

        assert_eq!(
            <Vector as CartesianVectorAlgebra>::cross(&lhs, &rhs),
            Jet0::cross(&lhs, &rhs,),
        );
    }

    #[test]
    fn cross_delegates_for_first_order_vector_jets() {
        let lhs = Jet1::from_parts(
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
        );

        let rhs = Jet1::from_parts(
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
        );

        assert_eq!(
            <FirstVector as CartesianVectorAlgebra>::cross(&lhs, &rhs),
            Jet1::cross(&lhs, &rhs,),
        );
    }

    #[test]
    fn cross_delegates_for_second_order_vector_jets() {
        let lhs = Jet2::from_parts(
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
        );

        let rhs = Jet2::from_parts(
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
        );

        assert_eq!(
            <SecondVector as CartesianVectorAlgebra>::cross(&lhs, &rhs),
            Jet2::cross(&lhs, &rhs,),
        );
    }

    #[test]
    fn cross_delegates_for_bivariate_vector_jets() {
        let lhs = JetBivariate2::from_components(
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
        );

        let rhs = JetBivariate2::from_components(
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
        );

        assert_eq!(
            <BivariateVector as CartesianVectorAlgebra>::cross(&lhs, &rhs),
            JetBivariate2::cross(&lhs, &rhs,),
        );
    }

    // ------------------------------------------------------------------
    // RealCartesianVectorAlgebra
    // ------------------------------------------------------------------

    #[test]
    fn plain_real_cartesian_operations_delegate_correctly() {
        let lhs = Vector::new(vector(c(1.0, 2.0), c(3.0, -4.0), c(-5.0, 6.0)));

        let rhs = Vector::new(vector(c(7.0, -8.0), c(9.0, 10.0), c(11.0, -12.0)));

        assert_eq!(
            <Vector as RealCartesianVectorAlgebra>::conjugated(&lhs),
            Jet0::conjugated(&lhs),
        );

        assert_eq!(
            <Vector as RealCartesianVectorAlgebra>::real(&lhs),
            Jet0::real(&lhs),
        );

        assert_eq!(
            <Vector as RealCartesianVectorAlgebra>::hermitian_dot(&lhs, &rhs,),
            Jet0::hermitian_dot(&lhs, &rhs,),
        );
    }

    #[test]
    fn first_order_real_cartesian_operations_delegate_correctly() {
        let lhs = Jet1::from_parts(
            vector(c(1.0, 2.0), c(3.0, -4.0), c(-5.0, 6.0)),
            vector(c(2.0, -1.0), c(4.0, 3.0), c(6.0, -5.0)),
        );

        let rhs = Jet1::from_parts(
            vector(c(7.0, -8.0), c(9.0, 10.0), c(11.0, -12.0)),
            vector(c(8.0, 7.0), c(10.0, -9.0), c(12.0, 11.0)),
        );

        assert_eq!(
            <FirstVector as RealCartesianVectorAlgebra>::conjugated(&lhs),
            Jet1::conjugated(&lhs),
        );

        assert_eq!(
            <FirstVector as RealCartesianVectorAlgebra>::real(&lhs),
            Jet1::real(&lhs),
        );

        assert_eq!(
            <FirstVector as RealCartesianVectorAlgebra>::hermitian_dot(&lhs, &rhs,),
            Jet1::hermitian_dot(&lhs, &rhs,),
        );
    }

    #[test]
    fn second_order_real_cartesian_operations_delegate_correctly() {
        let lhs = Jet2::from_parts(
            vector(c(1.0, 2.0), c(3.0, -4.0), c(-5.0, 6.0)),
            vector(c(2.0, -1.0), c(4.0, 3.0), c(6.0, -5.0)),
            vector(c(3.0, 0.5), c(5.0, -2.0), c(7.0, 4.0)),
        );

        let rhs = Jet2::from_parts(
            vector(c(7.0, -8.0), c(9.0, 10.0), c(11.0, -12.0)),
            vector(c(8.0, 7.0), c(10.0, -9.0), c(12.0, 11.0)),
            vector(c(9.0, -6.0), c(11.0, 8.0), c(13.0, -10.0)),
        );

        assert_eq!(
            <SecondVector as RealCartesianVectorAlgebra>::conjugated(&lhs),
            Jet2::conjugated(&lhs),
        );

        assert_eq!(
            <SecondVector as RealCartesianVectorAlgebra>::real(&lhs),
            Jet2::real(&lhs),
        );

        assert_eq!(
            <SecondVector as RealCartesianVectorAlgebra>::hermitian_dot(&lhs, &rhs,),
            Jet2::hermitian_dot(&lhs, &rhs,),
        );
    }

    #[test]
    fn bivariate_real_cartesian_operations_delegate_correctly() {
        let lhs = JetBivariate2::from_components(
            vector(c(1.0, 2.0), c(3.0, -4.0), c(-5.0, 6.0)),
            vector(c(2.0, -1.0), c(4.0, 3.0), c(6.0, -5.0)),
            vector(c(3.0, 0.5), c(5.0, -2.0), c(7.0, 4.0)),
            vector(c(4.0, -3.0), c(6.0, 5.0), c(8.0, -7.0)),
            vector(c(5.0, 4.0), c(7.0, -6.0), c(9.0, 8.0)),
            vector(c(6.0, -5.0), c(8.0, 7.0), c(10.0, -9.0)),
        );

        let rhs = JetBivariate2::from_components(
            vector(c(7.0, -8.0), c(9.0, 10.0), c(11.0, -12.0)),
            vector(c(8.0, 7.0), c(10.0, -9.0), c(12.0, 11.0)),
            vector(c(9.0, -6.0), c(11.0, 8.0), c(13.0, -10.0)),
            vector(c(10.0, 5.0), c(12.0, -7.0), c(14.0, 9.0)),
            vector(c(11.0, -4.0), c(13.0, 6.0), c(15.0, -8.0)),
            vector(c(12.0, 3.0), c(14.0, -5.0), c(16.0, 7.0)),
        );

        assert_eq!(
            <BivariateVector as RealCartesianVectorAlgebra>::conjugated(&lhs),
            JetBivariate2::conjugated(&lhs),
        );

        assert_eq!(
            <BivariateVector as RealCartesianVectorAlgebra>::real(&lhs),
            JetBivariate2::real(&lhs),
        );

        assert_eq!(
            <BivariateVector as RealCartesianVectorAlgebra>::hermitian_dot(&lhs, &rhs,),
            JetBivariate2::hermitian_dot(&lhs, &rhs,),
        );
    }
}
