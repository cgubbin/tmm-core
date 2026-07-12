use ndarray::Dimension;

use crate::{
    ComplexScalar,
    backend::transfer2::{
        derivatives::{LayerFirstDerivatives, LayerSecondDerivatives},
        quantities::IsotropicLayerQuantities,
    },
    stack::Thickness,
};

use super::Matrix2;

impl<C, D> Matrix2<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    /// Compute the 2×2 transfer matrix for one isotropic layer.
    pub(crate) fn from_layer(
        q: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
    ) -> Self {
        let d = C::from_real(thickness.as_cm());

        let kd = q.kappa.mapv(|k| k * d);
        let coskd = kd.mapv(|x| x.cos());
        let sinkd = kd.mapv(|x| x.sin());

        Self::new(
            coskd.clone(),
            -sinkd.clone() * q.factor.view() / q.kappa.view(),
            sinkd * q.kappa.view() / q.factor.view(),
            coskd,
        )
    }
    /// First derivative of the layer matrix with respect to physical thickness.
    pub(crate) fn thickness_derivative(
        q: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
    ) -> Self {
        let d = C::from_real(thickness.as_cm());

        let kd = q.kappa.mapv(|k| k * d);
        let coskd = kd.mapv(|x| x.cos());
        let sinkd = kd.mapv(|x| x.sin());

        Self::new(
            -q.kappa.clone() * sinkd.clone(),
            -q.factor.clone() * coskd.clone(),
            q.kappa.mapv(|k| k * k) * coskd / q.factor.view(),
            -sinkd * &q.kappa,
        )
    }

    /// Second derivative of the layer matrix with respect to physical thickness.
    pub(crate) fn thickness_second_derivative(
        q: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
    ) -> Self {
        let d = C::from_real(thickness.as_cm());

        let kd = q.kappa.mapv(|k| k * d);
        let coskd = kd.mapv(|x| x.cos());
        let sinkd = kd.mapv(|x| x.sin());

        let k2 = q.kappa.mapv(|k| k * k);
        let k3 = q.kappa.mapv(|k| k * k * k);

        Self::new(
            -k2.clone() * coskd.clone(),
            q.kappa.clone() * q.factor.clone() * sinkd.clone(),
            -k3 * sinkd / q.factor.view(),
            -k2 * coskd,
        )
    }

    pub(crate) fn spectral_derivative(
        q: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
        derivatives: &LayerFirstDerivatives<C, D>,
    ) -> Self {
        let d = C::from_real(thickness.as_cm());

        let kd = q.kappa.mapv(|k| k * d);
        let coskd = kd.mapv(|x| x.cos());
        let sinkd = kd.mapv(|x| x.sin());

        let d_sin = coskd.clone() * derivatives.dkappa.clone() * d;
        let d_cos = -sinkd.clone() * derivatives.dkappa.clone() * d;

        let k2 = q.kappa.mapv(|k| k * k);
        let f2 = q.factor.mapv(|f| f * f);

        let d_factor_over_kappa = derivatives.dfactor.clone() / q.kappa.view()
            - q.factor.clone() * derivatives.dkappa.clone() / k2.view();

        let d_kappa_over_factor = derivatives.dkappa.clone() / q.factor.view()
            - q.kappa.clone() * derivatives.dfactor.clone() / f2.view();

        Self::new(
            d_cos.clone(),
            -(d_sin.clone() * q.factor.view() / q.kappa.view()
                + sinkd.clone() * d_factor_over_kappa),
            d_sin * q.kappa.view() / q.factor.view() + sinkd * d_kappa_over_factor,
            d_cos,
        )
    }

    pub(crate) fn spectral_second_derivative(
        q: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
        derivatives: &LayerSecondDerivatives<C, D>,
    ) -> Self {
        let d = C::from_real(thickness.as_cm());

        let kd = q.kappa.mapv(|k| k * d);
        let coskd = kd.mapv(|x| x.cos());
        let sinkd = kd.mapv(|x| x.sin());

        let dk = &derivatives.first.dkappa;
        let ddk = &derivatives.ddkappa;
        let df = &derivatives.first.dfactor;
        let ddf = &derivatives.ddfactor;

        let dtheta = dk.clone() * d;
        let ddtheta = ddk.clone() * d;

        let d2_cos = -coskd.clone() * dtheta.mapv(|x| x * x) - sinkd.clone() * ddtheta.clone();

        let d_sin = coskd.clone() * dtheta.clone();
        let d2_sin = -sinkd.clone() * dtheta.mapv(|x| x * x) + coskd.clone() * ddtheta;

        let k = &q.kappa;
        let f = &q.factor;

        let k2 = k.mapv(|x| x * x);
        let k3 = k.mapv(|x| x * x * x);

        let f2 = f.mapv(|x| x * x);
        let f3 = f.mapv(|x| x * x * x);

        // A = f/k
        let a = f.clone() / k.view();
        let da = df.clone() / k.view() - f.clone() * dk.clone() / k2.view();
        let dda = ddf.clone() / k.view()
            - f.clone() * ddk.clone() / k2.view()
            - (df.clone() * dk.clone()).mapv(|x| x + x) / k2.view()
            + (f.clone() * dk.mapv(|x| x * x)).mapv(|x| x + x) / k3.view();

        // B = k/f
        let b = k.clone() / f.view();
        let db = dk.clone() / f.view() - k.clone() * df.clone() / f2.view();
        let ddb = ddk.clone() / f.view()
            - k.clone() * ddf / f2.view()
            - (dk * df.clone()).mapv(|x| x + x) / f2.view()
            + (k * df.mapv(|x| x * x)).mapv(|x| x + x) / f3.view();

        let two = C::one() + C::one();

        // m12 = -sin(theta) A
        let m12 = -(d2_sin.clone() * a
            + (d_sin.clone() * da.clone()).mapv(|each| each * two)
            + sinkd.clone() * dda);

        // m21 = sin(theta) B
        let m21 = d2_sin * b + (d_sin * db).mapv(|each| each * two) + sinkd * ddb;

        Self::new(d2_cos.clone(), m12, m21, d2_cos)
    }
}

#[cfg(test)]
mod thickness_tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0, arr1};
    use num_complex::Complex64;

    use super::*;
    use crate::backend::PlanarInput;
    use crate::backend::{Polarisation, transfer2::quantities::isotropic_layer_quantities};
    use crate::material::Constant;

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

        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                arr0(c(1000.0)),
                arr0(c(0.0)),
                Polarisation::TransverseElectric,
            ),
        );

        let matrix = Matrix2::from_layer(&q, Thickness::zero());

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
            let q = isotropic_layer_quantities(
                &material,
                &PlanarInput::new(arr0(c(1000.0)), arr0(c(100.0)), polarisation),
            );
            let matrix = Matrix2::from_layer(&q, Thickness::from_nm(100.0).unwrap());

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

        let vacuum_wavenumber = arr0(c(1000.0));
        let parallel_wavenumber = arr0(c(100.0));

        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                vacuum_wavenumber,
                parallel_wavenumber,
                Polarisation::TransverseElectric,
            ),
        );

        let analytical = Matrix2::thickness_derivative(&q, Thickness::from_nm(d0_nm).unwrap());

        let plus = Matrix2::from_layer(&q, Thickness::from_nm(d0_nm + h_nm).unwrap());

        let minus = Matrix2::from_layer(&q, Thickness::from_nm(d0_nm - h_nm).unwrap());

        let h_cm = Thickness::from_nm(h_nm).unwrap().as_cm();
        let expected = (&plus.add(&(&minus).scale(c(-1.0)))).scale(c(1.0 / (2.0 * h_cm)));

        assert_matrix_close(&analytical, &expected, 1e-6);
    }

    #[test]
    fn thickness_second_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let d0_nm = 100.0;
        let h_nm = 1e-2;

        let vacuum_wavenumber = arr0(c(1000.0));
        let parallel_wavenumber = arr0(c(100.0));

        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                vacuum_wavenumber,
                parallel_wavenumber,
                Polarisation::TransverseElectric,
            ),
        );

        let analytical =
            Matrix2::thickness_second_derivative(&q, Thickness::from_nm(d0_nm).unwrap());

        let plus = Matrix2::from_layer(&q, Thickness::from_nm(d0_nm + h_nm).unwrap());

        let zero = Matrix2::from_layer(&q, Thickness::from_nm(d0_nm).unwrap());

        let minus = Matrix2::from_layer(&q, Thickness::from_nm(d0_nm - h_nm).unwrap());

        let h_cm = Thickness::from_nm(h_nm).unwrap().as_cm();
        let expected =
            (&plus.add(&(&zero).scale(c(-2.0))).add(&minus)).scale(c(1.0 / (h_cm * h_cm)));

        assert_matrix_close(&analytical, &expected, 1e-4);
    }

    #[test]
    fn ndarray_input_shape_is_preserved() {
        let material = Constant::new(2.25);

        let vacuum_wavenumber = arr1(&[c(1000.0), c(1100.0), c(1200.0)]);
        let parallel_wavenumber = arr1(&[c(0.0), c(10.0), c(20.0)]);

        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                vacuum_wavenumber,
                parallel_wavenumber,
                Polarisation::TransverseElectric,
            ),
        );

        let matrix = Matrix2::from_layer(&q, Thickness::from_nm(100.0).unwrap());

        assert_eq!(matrix.m11().shape(), &[3]);
        assert_eq!(matrix.m12().shape(), &[3]);
        assert_eq!(matrix.m21().shape(), &[3]);
        assert_eq!(matrix.m22().shape(), &[3]);
    }

    #[test]
    fn te_and_tm_have_same_determinant_for_nonmagnetic_isotropic_from_layer() {
        let material = Constant::new(2.25);
        let vacuum_wavenumber = arr0(c(1000.0));
        let parallel_wavenumber = arr0(c(0.0));

        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                vacuum_wavenumber,
                parallel_wavenumber,
                Polarisation::TransverseElectric,
            ),
        );

        let te = Matrix2::from_layer(&q, Thickness::from_nm(100.0).unwrap());

        let tm = Matrix2::from_layer(&q, Thickness::from_nm(100.0).unwrap());

        assert_relative_eq!(
            te.determinant()[()],
            tm.determinant()[()],
            max_relative = 1e-12,
            epsilon = 1e-12
        );
    }
}
