use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    material::{DerivativeOrder, Material, Scalar, SpectralVariable},
    stack::Thickness,
};

use super::{Matrix2, Polarisation, isotropic_layer_quantities};

/// Compute the 2×2 transfer matrix for one isotropic layer.
pub fn isotropic_layer_matrix<M, C, D>(
    material: &M,
    thickness: Thickness<C::RealField>,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> Matrix2<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let d = C::from_real(thickness.as_cm());

    let kd = q.kappa.mapv(|k| k * d);
    let coskd = kd.mapv(|x| x.cos());
    let sinkd = kd.mapv(|x| x.sin());

    Matrix2::new(
        coskd.clone(),
        -sinkd.clone() * q.factor.view() / q.kappa.view(),
        sinkd * q.kappa.view() / q.factor.view(),
        coskd,
    )
}
/// First derivative of the layer matrix with respect to physical thickness.
pub fn isotropic_layer_thickness_derivative<M, C, D>(
    material: &M,
    thickness: Thickness<C::RealField>,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> Matrix2<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let d = C::from_real(thickness.as_cm());

    let kd = q.kappa.mapv(|k| k * d);
    let coskd = kd.mapv(|x| x.cos());
    let sinkd = kd.mapv(|x| x.sin());

    Matrix2::new(
        -q.kappa.clone() * sinkd.clone(),
        -q.factor.clone() * coskd.clone(),
        q.kappa.mapv(|k| k * k) * coskd / q.factor.view(),
        -q.kappa * sinkd,
    )
}

/// Second derivative of the layer matrix with respect to physical thickness.
pub fn isotropic_layer_thickness_second_derivative<M, C, D>(
    material: &M,
    thickness: Thickness<C::RealField>,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> Matrix2<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let d = C::from_real(thickness.as_cm());

    let kd = q.kappa.mapv(|k| k * d);
    let coskd = kd.mapv(|x| x.cos());
    let sinkd = kd.mapv(|x| x.sin());

    let k2 = q.kappa.mapv(|k| k * k);
    let k3 = q.kappa.mapv(|k| k * k * k);

    Matrix2::new(
        -k2.clone() * coskd.clone(),
        q.kappa.clone() * q.factor.clone() * sinkd.clone(),
        -k3 * sinkd / q.factor.view(),
        -k2 * coskd,
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0, arr1};
    use num_complex::Complex64;

    use super::*;
    use crate::material::Constant;

    use std::ops::Add;

    type C = Complex64;

    fn c(x: f64) -> C {
        C::new(x, 0.0)
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.shape(), expected.shape());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(
                actual.re,
                expected.re,
                max_relative = tolerance,
                epsilon = tolerance
            );
            assert_relative_eq!(
                actual.im,
                expected.im,
                max_relative = tolerance,
                epsilon = tolerance
            );
        }
    }

    fn assert_matrix_close<D>(actual: &Matrix2<C, D>, expected: &Matrix2<C, D>, tolerance: f64)
    where
        D: Dimension,
    {
        assert_array_close(actual.m11(), expected.m11(), tolerance);
        assert_array_close(actual.m12(), expected.m12(), tolerance);
        assert_array_close(actual.m21(), expected.m21(), tolerance);
        assert_array_close(actual.m22(), expected.m22(), tolerance);
    }

    #[test]
    fn zero_thickness_layer_is_identity() {
        let material = Constant::new(2.25);

        let matrix = isotropic_layer_matrix(
            &material,
            Thickness::zero(),
            &arr0(c(1000.0)),
            &arr0(c(0.0)),
            Polarisation::TransverseElectric,
        );

        let expected = Matrix2::identity_like(matrix.m11());

        assert_matrix_close(&matrix, &expected, 1e-12);
    }

    #[test]
    fn layer_matrix_has_unit_determinant() {
        let material = Constant::new(2.25);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let matrix = isotropic_layer_matrix(
                &material,
                Thickness::from_nm(100.0).unwrap(),
                &arr0(c(1000.0)),
                &arr0(c(100.0)),
                polarisation,
            );

            assert_relative_eq!(
                matrix.determinant()[()],
                c(1.0),
                max_relative = 1e-12,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn thickness_first_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let d0_nm = 100.0;
        let h_nm = 1e-3;

        let wavenumber = arr0(c(1000.0));
        let beta2 = arr0(c(100.0));

        let analytical = isotropic_layer_thickness_derivative(
            &material,
            Thickness::from_nm(d0_nm).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        let plus = isotropic_layer_matrix(
            &material,
            Thickness::from_nm(d0_nm + h_nm).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        let minus = isotropic_layer_matrix(
            &material,
            Thickness::from_nm(d0_nm - h_nm).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        let h_cm = Thickness::from_nm(h_nm).unwrap().as_cm();
        let expected = (&plus.add(&(&minus).scale(c(-1.0)))).scale(c(1.0 / (2.0 * h_cm)));

        assert_matrix_close(&analytical, &expected, 1e-6);
    }

    #[test]
    fn thickness_second_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let d0_nm = 100.0;
        let h_nm = 1e-2;

        let wavenumber = arr0(c(1000.0));
        let beta2 = arr0(c(100.0));

        let analytical = isotropic_layer_thickness_second_derivative(
            &material,
            Thickness::from_nm(d0_nm).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        let plus = isotropic_layer_matrix(
            &material,
            Thickness::from_nm(d0_nm + h_nm).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        let zero = isotropic_layer_matrix(
            &material,
            Thickness::from_nm(d0_nm).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        let minus = isotropic_layer_matrix(
            &material,
            Thickness::from_nm(d0_nm - h_nm).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        let h_cm = Thickness::from_nm(h_nm).unwrap().as_cm();
        let expected =
            (&plus.add(&(&zero).scale(c(-2.0))).add(&minus)).scale(c(1.0 / (h_cm * h_cm)));

        assert_matrix_close(&analytical, &expected, 1e-4);
    }

    #[test]
    fn ndarray_input_shape_is_preserved() {
        let material = Constant::new(2.25);

        let wavenumber = arr1(&[c(1000.0), c(1100.0), c(1200.0)]);
        let beta2 = arr1(&[c(0.0), c(10.0), c(20.0)]);

        let matrix = isotropic_layer_matrix(
            &material,
            Thickness::from_nm(100.0).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        assert_eq!(matrix.m11().shape(), &[3]);
        assert_eq!(matrix.m12().shape(), &[3]);
        assert_eq!(matrix.m21().shape(), &[3]);
        assert_eq!(matrix.m22().shape(), &[3]);
    }

    #[test]
    fn te_and_tm_have_same_determinant_for_nonmagnetic_isotropic_layer() {
        let material = Constant::new(2.25);
        let wavenumber = arr0(c(1000.0));
        let beta2 = arr0(c(0.0));

        let te = isotropic_layer_matrix(
            &material,
            Thickness::from_nm(100.0).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseElectric,
        );

        let tm = isotropic_layer_matrix(
            &material,
            Thickness::from_nm(100.0).unwrap(),
            &wavenumber,
            &beta2,
            Polarisation::TransverseMagnetic,
        );

        assert_relative_eq!(
            te.determinant()[()],
            tm.determinant()[()],
            max_relative = 1e-12,
            epsilon = 1e-12
        );
    }
}
