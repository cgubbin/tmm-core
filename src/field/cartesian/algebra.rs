use crate::{
    algebra::{
        ArrayJet, ArrayJetBivariate, ArrayJetFirst, Jet, JetBivariate, JetFirst, RealParameter,
        ScalarAlgebra,
    },
    field::CartesianVector3,
};

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use std::fmt::Debug;

pub(crate) trait CartesianScalarAlgebra<T, D>: ScalarAlgebra<T, D>
where
    D: Dimension,
{
    type Vector: CartesianVectorAlgebra<Coefficient = T, ScalarField = Self>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector;
}

pub(crate) trait CartesianVectorAlgebra: Clone + Sized {
    type Coefficient;
    type ScalarField;

    fn cross(&self, rhs: &Self) -> Self;

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self;

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self;
}

pub(crate) trait RealCartesianVectorAlgebra: CartesianVectorAlgebra {
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

impl<T, D> CartesianScalarAlgebra<T, D> for ArrayBase<OwnedRepr<T>, D>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type Vector = CartesianVector3<T, D>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        CartesianVector3::new(x, y, z)
    }
}

impl<T, D, P> CartesianScalarAlgebra<T, D> for ArrayJetFirst<T, D, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Vector = JetFirst<CartesianVector3<T, D>, P>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        JetFirst::from_parts(
            CartesianVector3::new(x.value().clone(), y.value().clone(), z.value().clone()),
            CartesianVector3::new(x.first().clone(), y.first().clone(), z.first().clone()),
        )
    }
}

impl<T, D, P> CartesianScalarAlgebra<T, D> for ArrayJet<T, D, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Vector = Jet<CartesianVector3<T, D>, P>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        Jet::from_parts(
            CartesianVector3::new(x.value().clone(), y.value().clone(), z.value().clone()),
            CartesianVector3::new(x.first().clone(), y.first().clone(), z.first().clone()),
            CartesianVector3::new(x.second().clone(), y.second().clone(), z.second().clone()),
        )
    }
}

impl<T, D, P> CartesianScalarAlgebra<T, D> for ArrayJetBivariate<T, D, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Vector = JetBivariate<CartesianVector3<T, D>, P>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        JetBivariate::from_components(
            CartesianVector3::new(x.value().clone(), y.value().clone(), z.value().clone()),
            CartesianVector3::new(x.x().clone(), y.x().clone(), z.x().clone()),
            CartesianVector3::new(x.y().clone(), y.y().clone(), z.y().clone()),
            CartesianVector3::new(x.xx().clone(), y.xx().clone(), z.xx().clone()),
            CartesianVector3::new(x.xy().clone(), y.xy().clone(), z.xy().clone()),
            CartesianVector3::new(x.yy().clone(), y.yy().clone(), z.yy().clone()),
        )
    }
}

impl<T, D> CartesianVectorAlgebra for CartesianVector3<T, D>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type Coefficient = T;
    type ScalarField = ArrayBase<OwnedRepr<T>, D>;

    fn cross(&self, rhs: &Self) -> Self {
        CartesianVector3::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        self.clone() * factor
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        self.clone() * factor
    }
}

impl<T, D> RealCartesianVectorAlgebra for CartesianVector3<T, D>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = CartesianVector3<T::RealField, D>;
    type RealScalarField = ArrayBase<OwnedRepr<T::RealField>, D>;

    fn conjugated(&self) -> Self {
        CartesianVector3::conjugate(self)
    }

    fn real(&self) -> Self::RealVector {
        self.map(|value| value.real())
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        CartesianVector3::hermitian_dot(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        value.mapv(|value| value.real())
    }
}

impl<T, D, P> CartesianVectorAlgebra for JetFirst<CartesianVector3<T, D>, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Coefficient = T;
    type ScalarField = ArrayJetFirst<T, D, P>;

    fn cross(&self, rhs: &Self) -> Self {
        JetFirst::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        JetFirst::scale_by(self, factor)
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        JetFirst::multiply_by_scalar(self, factor)
    }
}

impl<T, D> RealCartesianVectorAlgebra for JetFirst<CartesianVector3<T, D>, RealParameter>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = JetFirst<CartesianVector3<T::RealField, D>, RealParameter>;
    type RealScalarField = ArrayJetFirst<T::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        JetFirst::conjugated(self)
    }

    fn real(&self) -> Self::RealVector {
        JetFirst::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        JetFirst::hermitian_dot_product(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        JetFirst::real(&value)
    }
}

impl<T, D, P> CartesianVectorAlgebra for Jet<CartesianVector3<T, D>, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Coefficient = T;
    type ScalarField = ArrayJet<T, D, P>;

    fn cross(&self, rhs: &Self) -> Self {
        Jet::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        Jet::scale_by(self, factor)
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        Jet::multiply_by_scalar(self, factor)
    }
}

impl<T, D> RealCartesianVectorAlgebra for Jet<CartesianVector3<T, D>, RealParameter>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = Jet<CartesianVector3<T::RealField, D>, RealParameter>;
    type RealScalarField = ArrayJet<T::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        Jet::conjugated(self)
    }

    fn real(&self) -> Self::RealVector {
        Jet::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        Jet::hermitian_dot_product(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        Jet::real(&value)
    }
}

impl<T, D, P> CartesianVectorAlgebra for JetBivariate<CartesianVector3<T, D>, P>
where
    T: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    type Coefficient = T;
    type ScalarField = ArrayJetBivariate<T, D, P>;

    fn cross(&self, rhs: &Self) -> Self {
        JetBivariate::cross(self, rhs)
    }

    fn scale_by_constant(&self, factor: Self::Coefficient) -> Self {
        JetBivariate::scale_by(self, factor)
    }

    fn multiply_by_scalar(&self, factor: &Self::ScalarField) -> Self {
        JetBivariate::multiply_by_scalar(self, factor)
    }
}

impl<T, D> RealCartesianVectorAlgebra for JetBivariate<CartesianVector3<T, D>, RealParameter>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    type RealVector = JetBivariate<CartesianVector3<T::RealField, D>, RealParameter>;
    type RealScalarField = ArrayJetBivariate<T::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        JetBivariate::conjugated(self)
    }

    fn real(&self) -> Self::RealVector {
        JetBivariate::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ScalarField {
        JetBivariate::hermitian_dot_product(self, rhs)
    }

    fn scalar_real(value: Self::ScalarField) -> Self::RealScalarField {
        JetBivariate::real(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;

    use crate::algebra::{Jet, JetBivariate, JetFirst, RealParameter};

    type C = Complex64;
    type D = Ix1;
    type Scalar = Array1<C>;

    type FirstScalar = ArrayJetFirst<C, D, RealParameter>;

    type SecondScalar = ArrayJet<C, D, RealParameter>;

    type BivariateScalar = ArrayJetBivariate<C, D, RealParameter>;

    type Vector = CartesianVector3<C, D>;

    type FirstVector = JetFirst<Vector, RealParameter>;

    type SecondVector = Jet<Vector, RealParameter>;

    type BivariateVector = JetBivariate<Vector, RealParameter>;

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

    fn assert_vector_close(actual: &Vector, expected: &Vector) {
        assert_complex_array_close(actual.x(), expected.x());

        assert_complex_array_close(actual.y(), expected.y());

        assert_complex_array_close(actual.z(), expected.z());
    }

    fn vector(x: C, y: C, z: C) -> Vector {
        CartesianVector3::new(arr1(&[x]), arr1(&[y]), arr1(&[z]))
    }

    fn scale_vector(vector: &Vector, scalar: &Scalar) -> Vector {
        vector.clone() * scalar
    }

    fn add_vectors(terms: &[Vector]) -> Vector {
        let mut result = CartesianVector3::zeros_like(terms[0].x());

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
        S: CartesianScalarAlgebra<C, D>,
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

        let result = <Scalar as CartesianScalarAlgebra<C, D>>::into_cartesian_vector(
            x.clone(),
            y.clone(),
            z.clone(),
        );

        assert_eq!(result.x(), &x);
        assert_eq!(result.y(), &y);
        assert_eq!(result.z(), &z);
    }

    #[test]
    fn first_order_scalar_jets_are_transposed_into_vector_jet() {
        let x = JetFirst::from_parts(arr1(&[c(1.0, 0.0)]), arr1(&[c(2.0, 0.0)]));

        let y = JetFirst::from_parts(arr1(&[c(3.0, 0.0)]), arr1(&[c(4.0, 0.0)]));

        let z = JetFirst::from_parts(arr1(&[c(5.0, 0.0)]), arr1(&[c(6.0, 0.0)]));

        let result = <FirstScalar as CartesianScalarAlgebra<C, D>>::into_cartesian_vector(x, y, z);

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
        let x = Jet::from_parts(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0)]),
            arr1(&[c(3.0, 0.0)]),
        );

        let y = Jet::from_parts(
            arr1(&[c(4.0, 0.0)]),
            arr1(&[c(5.0, 0.0)]),
            arr1(&[c(6.0, 0.0)]),
        );

        let z = Jet::from_parts(
            arr1(&[c(7.0, 0.0)]),
            arr1(&[c(8.0, 0.0)]),
            arr1(&[c(9.0, 0.0)]),
        );

        let result = <SecondScalar as CartesianScalarAlgebra<C, D>>::into_cartesian_vector(x, y, z);

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
        let x = JetBivariate::from_components(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0)]),
            arr1(&[c(3.0, 0.0)]),
            arr1(&[c(4.0, 0.0)]),
            arr1(&[c(5.0, 0.0)]),
            arr1(&[c(6.0, 0.0)]),
        );

        let y = JetBivariate::from_components(
            arr1(&[c(7.0, 0.0)]),
            arr1(&[c(8.0, 0.0)]),
            arr1(&[c(9.0, 0.0)]),
            arr1(&[c(10.0, 0.0)]),
            arr1(&[c(11.0, 0.0)]),
            arr1(&[c(12.0, 0.0)]),
        );

        let z = JetBivariate::from_components(
            arr1(&[c(13.0, 0.0)]),
            arr1(&[c(14.0, 0.0)]),
            arr1(&[c(15.0, 0.0)]),
            arr1(&[c(16.0, 0.0)]),
            arr1(&[c(17.0, 0.0)]),
            arr1(&[c(18.0, 0.0)]),
        );

        let result =
            <BivariateScalar as CartesianScalarAlgebra<C, D>>::into_cartesian_vector(x, y, z);

        assert_vector_close(
            result.value(),
            &vector(c(1.0, 0.0), c(7.0, 0.0), c(13.0, 0.0)),
        );

        assert_vector_close(result.x(), &vector(c(2.0, 0.0), c(8.0, 0.0), c(14.0, 0.0)));

        assert_vector_close(result.y(), &vector(c(3.0, 0.0), c(9.0, 0.0), c(15.0, 0.0)));

        assert_vector_close(
            result.xx(),
            &vector(c(4.0, 0.0), c(10.0, 0.0), c(16.0, 0.0)),
        );

        assert_vector_close(
            result.xy(),
            &vector(c(5.0, 0.0), c(11.0, 0.0), c(17.0, 0.0)),
        );

        assert_vector_close(
            result.yy(),
            &vector(c(6.0, 0.0), c(12.0, 0.0), c(18.0, 0.0)),
        );
    }

    // ------------------------------------------------------------------
    // Constant scaling
    // ------------------------------------------------------------------

    #[test]
    fn plain_vector_constant_scaling_delegates_correctly() {
        let source = vector(c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0));

        let factor = c(2.0, -1.0);

        let result = <Vector as CartesianVectorAlgebra>::scale_by_constant(&source, factor);

        assert_eq!(result, source * factor,);
    }

    #[test]
    fn first_order_vector_constant_scaling_scales_every_jet_component() {
        let source = JetFirst::from_parts(
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
        let source = Jet::from_parts(
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
        let source = JetBivariate::from_components(
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

        assert_vector_close(result.x(), &(source.x().clone() * factor));

        assert_vector_close(result.y(), &(source.y().clone() * factor));

        assert_vector_close(result.xx(), &(source.xx().clone() * factor));

        assert_vector_close(result.xy(), &(source.xy().clone() * factor));

        assert_vector_close(result.yy(), &(source.yy().clone() * factor));
    }

    // ------------------------------------------------------------------
    // Vector-scalar products
    // ------------------------------------------------------------------

    #[test]
    fn plain_vector_scalar_multiplication_is_pointwise() {
        let source = CartesianVector3::new(
            arr1(&[c(1.0, 0.0), c(2.0, 0.0)]),
            arr1(&[c(3.0, 0.0), c(4.0, 0.0)]),
            arr1(&[c(5.0, 0.0), c(6.0, 0.0)]),
        );

        let scalar = arr1(&[c(2.0, 0.0), c(3.0, 0.0)]);

        let result = <Vector as CartesianVectorAlgebra>::multiply_by_scalar(&source, &scalar);

        assert_eq!(result, source * &scalar,);
    }

    #[test]
    fn first_order_vector_scalar_product_obeys_product_rule() {
        let vector = JetFirst::from_parts(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
        );

        let scalar = JetFirst::from_parts(arr1(&[c(7.0, 0.0)]), arr1(&[c(8.0, 0.0)]));

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
        let vector = Jet::from_parts(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
            vector(c(7.0, 0.0), c(8.0, 0.0), c(9.0, 0.0)),
        );

        let scalar = Jet::from_parts(
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
        let vector = JetBivariate::from_components(
            vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)),
            vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)),
            vector(c(7.0, 0.0), c(8.0, 0.0), c(9.0, 0.0)),
            vector(c(10.0, 0.0), c(11.0, 0.0), c(12.0, 0.0)),
            vector(c(13.0, 0.0), c(14.0, 0.0), c(15.0, 0.0)),
            vector(c(16.0, 0.0), c(17.0, 0.0), c(18.0, 0.0)),
        );

        let scalar = JetBivariate::from_components(
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
            scale_vector(vector.x(), scalar.value()),
            scale_vector(vector.value(), scalar.x()),
        ]);

        let expected_y = add_vectors(&[
            scale_vector(vector.y(), scalar.value()),
            scale_vector(vector.value(), scalar.y()),
        ]);

        let expected_xx = add_vectors(&[
            scale_vector(vector.xx(), scalar.value()),
            scale_vector(vector.x(), scalar.x()) * c(2.0, 0.0),
            scale_vector(vector.value(), scalar.xx()),
        ]);

        let expected_xy = add_vectors(&[
            scale_vector(vector.xy(), scalar.value()),
            scale_vector(vector.x(), scalar.y()),
            scale_vector(vector.y(), scalar.x()),
            scale_vector(vector.value(), scalar.xy()),
        ]);

        let expected_yy = add_vectors(&[
            scale_vector(vector.yy(), scalar.value()),
            scale_vector(vector.y(), scalar.y()) * c(2.0, 0.0),
            scale_vector(vector.value(), scalar.yy()),
        ]);

        assert_vector_close(result.value(), &expected_value);

        assert_vector_close(result.x(), &expected_x);

        assert_vector_close(result.y(), &expected_y);

        assert_vector_close(result.xx(), &expected_xx);

        assert_vector_close(result.xy(), &expected_xy);

        assert_vector_close(result.yy(), &expected_yy);
    }

    // ------------------------------------------------------------------
    // Cross products
    // ------------------------------------------------------------------

    #[test]
    fn cross_delegates_for_plain_vectors() {
        let lhs = vector(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0));

        let rhs = vector(c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0));

        assert_eq!(
            <Vector as CartesianVectorAlgebra>::cross(&lhs, &rhs),
            CartesianVector3::cross(&lhs, &rhs,),
        );
    }

    #[test]
    fn cross_delegates_for_first_order_vector_jets() {
        let lhs = JetFirst::from_parts(
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
        );

        let rhs = JetFirst::from_parts(
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
        );

        assert_eq!(
            <FirstVector as CartesianVectorAlgebra>::cross(&lhs, &rhs),
            JetFirst::cross(&lhs, &rhs,),
        );
    }

    #[test]
    fn cross_delegates_for_second_order_vector_jets() {
        let lhs = Jet::from_parts(
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
        );

        let rhs = Jet::from_parts(
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
        );

        assert_eq!(
            <SecondVector as CartesianVectorAlgebra>::cross(&lhs, &rhs),
            Jet::cross(&lhs, &rhs,),
        );
    }

    #[test]
    fn cross_delegates_for_bivariate_vector_jets() {
        let lhs = JetBivariate::from_components(
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
        );

        let rhs = JetBivariate::from_components(
            vector(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            vector(c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)),
            vector(c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
        );

        assert_eq!(
            <BivariateVector as CartesianVectorAlgebra>::cross(&lhs, &rhs),
            JetBivariate::cross(&lhs, &rhs,),
        );
    }

    // ------------------------------------------------------------------
    // RealCartesianVectorAlgebra
    // ------------------------------------------------------------------

    #[test]
    fn plain_real_cartesian_operations_delegate_correctly() {
        let lhs = vector(c(1.0, 2.0), c(3.0, -4.0), c(-5.0, 6.0));

        let rhs = vector(c(7.0, -8.0), c(9.0, 10.0), c(11.0, -12.0));

        assert_eq!(
            <Vector as RealCartesianVectorAlgebra>::conjugated(&lhs),
            lhs.conjugate(),
        );

        assert_eq!(
            <Vector as RealCartesianVectorAlgebra>::real(&lhs),
            lhs.map(|value| value.re),
        );

        assert_eq!(
            <Vector as RealCartesianVectorAlgebra>::hermitian_dot(&lhs, &rhs,),
            lhs.hermitian_dot(&rhs),
        );
    }

    #[test]
    fn first_order_real_cartesian_operations_delegate_correctly() {
        let lhs = JetFirst::from_parts(
            vector(c(1.0, 2.0), c(3.0, -4.0), c(-5.0, 6.0)),
            vector(c(2.0, -1.0), c(4.0, 3.0), c(6.0, -5.0)),
        );

        let rhs = JetFirst::from_parts(
            vector(c(7.0, -8.0), c(9.0, 10.0), c(11.0, -12.0)),
            vector(c(8.0, 7.0), c(10.0, -9.0), c(12.0, 11.0)),
        );

        assert_eq!(
            <FirstVector as RealCartesianVectorAlgebra>::conjugated(&lhs),
            JetFirst::conjugated(&lhs),
        );

        assert_eq!(
            <FirstVector as RealCartesianVectorAlgebra>::real(&lhs),
            JetFirst::real(&lhs),
        );

        assert_eq!(
            <FirstVector as RealCartesianVectorAlgebra>::hermitian_dot(&lhs, &rhs,),
            JetFirst::hermitian_dot(&lhs, &rhs,),
        );
    }

    #[test]
    fn second_order_real_cartesian_operations_delegate_correctly() {
        let lhs = Jet::from_parts(
            vector(c(1.0, 2.0), c(3.0, -4.0), c(-5.0, 6.0)),
            vector(c(2.0, -1.0), c(4.0, 3.0), c(6.0, -5.0)),
            vector(c(3.0, 0.5), c(5.0, -2.0), c(7.0, 4.0)),
        );

        let rhs = Jet::from_parts(
            vector(c(7.0, -8.0), c(9.0, 10.0), c(11.0, -12.0)),
            vector(c(8.0, 7.0), c(10.0, -9.0), c(12.0, 11.0)),
            vector(c(9.0, -6.0), c(11.0, 8.0), c(13.0, -10.0)),
        );

        assert_eq!(
            <SecondVector as RealCartesianVectorAlgebra>::conjugated(&lhs),
            Jet::conjugated(&lhs),
        );

        assert_eq!(
            <SecondVector as RealCartesianVectorAlgebra>::real(&lhs),
            Jet::real(&lhs),
        );

        assert_eq!(
            <SecondVector as RealCartesianVectorAlgebra>::hermitian_dot(&lhs, &rhs,),
            Jet::hermitian_dot(&lhs, &rhs,),
        );
    }

    #[test]
    fn bivariate_real_cartesian_operations_delegate_correctly() {
        let lhs = JetBivariate::from_components(
            vector(c(1.0, 2.0), c(3.0, -4.0), c(-5.0, 6.0)),
            vector(c(2.0, -1.0), c(4.0, 3.0), c(6.0, -5.0)),
            vector(c(3.0, 0.5), c(5.0, -2.0), c(7.0, 4.0)),
            vector(c(4.0, -3.0), c(6.0, 5.0), c(8.0, -7.0)),
            vector(c(5.0, 4.0), c(7.0, -6.0), c(9.0, 8.0)),
            vector(c(6.0, -5.0), c(8.0, 7.0), c(10.0, -9.0)),
        );

        let rhs = JetBivariate::from_components(
            vector(c(7.0, -8.0), c(9.0, 10.0), c(11.0, -12.0)),
            vector(c(8.0, 7.0), c(10.0, -9.0), c(12.0, 11.0)),
            vector(c(9.0, -6.0), c(11.0, 8.0), c(13.0, -10.0)),
            vector(c(10.0, 5.0), c(12.0, -7.0), c(14.0, 9.0)),
            vector(c(11.0, -4.0), c(13.0, 6.0), c(15.0, -8.0)),
            vector(c(12.0, 3.0), c(14.0, -5.0), c(16.0, 7.0)),
        );

        assert_eq!(
            <BivariateVector as RealCartesianVectorAlgebra>::conjugated(&lhs),
            JetBivariate::conjugated(&lhs),
        );

        assert_eq!(
            <BivariateVector as RealCartesianVectorAlgebra>::real(&lhs),
            JetBivariate::real(&lhs),
        );

        assert_eq!(
            <BivariateVector as RealCartesianVectorAlgebra>::hermitian_dot(&lhs, &rhs,),
            JetBivariate::hermitian_dot(&lhs, &rhs,),
        );
    }
}
