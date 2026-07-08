use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    material::{DerivativeOrder, Material, Scalar, SpectralVariable},
    stack::Thickness,
};

use super::{Matrix2, Polarisation, isotropic_layer_quantities};

#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicLayerDerivatives<C, D>
where
    D: Dimension,
{
    pub dkappa: ArrayBase<OwnedRepr<C>, D>,
    pub dfactor: ArrayBase<OwnedRepr<C>, D>,
}

pub fn frequency_squared_derivatives<M, C, D>(
    material: &M,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> IsotropicLayerDerivatives<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let epsilon = q.epsilon;
    let mu = q.mu;
    let kappa = q.kappa;

    let deps = wavenumber.mapv(|w| {
        material.relative_permittivity_derivative(
            Scalar(w),
            DerivativeOrder::First,
            SpectralVariable::FrequencySquared,
        )
    });

    let dmu = wavenumber.mapv(|w| {
        material.relative_permeability_derivative(
            Scalar(w),
            DerivativeOrder::First,
            SpectralVariable::FrequencySquared,
        )
    });

    let omega2 = wavenumber.mapv(|w| w * w);

    let dq = (deps.clone() * mu.view() + epsilon.clone() * dmu.view()) * omega2
        + epsilon.clone() * mu.view();

    let two = C::one() + C::one();
    let dkappa = dq / kappa.mapv(|k| two * k);

    let dfactor = match polarisation {
        Polarisation::TransverseElectric => dmu,
        Polarisation::TransverseMagnetic => deps,
    };

    IsotropicLayerDerivatives { dkappa, dfactor }
}

pub fn propagation_constant_squared_derivatives<M, C, D>(
    material: &M,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> IsotropicLayerDerivatives<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let two = C::one() + C::one();

    let dkappa = q.kappa.mapv(|k| -C::one() / (two * k));
    let dfactor = q.factor.mapv(|_| C::zero());

    IsotropicLayerDerivatives { dkappa, dfactor }
}

pub fn isotropic_layer_matrix_from_derivatives<M, C, D>(
    material: &M,
    thickness: Thickness<C::RealField>,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
    derivatives: IsotropicLayerDerivatives<C, D>,
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

    let d_sin = coskd.clone() * derivatives.dkappa.clone() * d;
    let d_cos = -sinkd.clone() * derivatives.dkappa.clone() * d;

    let k2 = q.kappa.mapv(|k| k * k);
    let f2 = q.factor.mapv(|f| f * f);

    let d_factor_over_kappa = derivatives.dfactor.clone() / q.kappa.view()
        - q.factor.clone() * derivatives.dkappa.clone() / k2.view();

    let d_kappa_over_factor = derivatives.dkappa.clone() / q.factor.view()
        - q.kappa.clone() * derivatives.dfactor.clone() / f2.view();

    Matrix2::new(
        d_cos.clone(),
        -(d_sin.clone() * q.factor.view() / q.kappa.view() + sinkd.clone() * d_factor_over_kappa),
        d_sin * q.kappa.view() / q.factor.view() + sinkd * d_kappa_over_factor,
        d_cos,
    )
}

pub fn frequency_squared_derivative<M, C, D>(
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
    let derivatives = frequency_squared_derivatives(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    isotropic_layer_matrix_from_derivatives(
        material,
        thickness,
        wavenumber,
        propagation_constant_squared,
        polarisation,
        derivatives,
    )
}

pub fn propagation_constant_squared_derivative<M, C, D>(
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
    let derivatives = propagation_constant_squared_derivatives(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    isotropic_layer_matrix_from_derivatives(
        material,
        thickness,
        wavenumber,
        propagation_constant_squared,
        polarisation,
        derivatives,
    )
}

#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicLayerSecondDerivatives<C, D>
where
    D: Dimension,
{
    pub dkappa: ArrayBase<OwnedRepr<C>, D>,
    pub ddkappa: ArrayBase<OwnedRepr<C>, D>,
    pub dfactor: ArrayBase<OwnedRepr<C>, D>,
    pub ddfactor: ArrayBase<OwnedRepr<C>, D>,
}

pub fn frequency_squared_second_derivatives<M, C, D>(
    material: &M,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> IsotropicLayerSecondDerivatives<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let omega2 = wavenumber.mapv(|w| w * w);

    let deps = wavenumber.mapv(|w| {
        material.relative_permittivity_derivative(
            Scalar(w),
            DerivativeOrder::First,
            SpectralVariable::FrequencySquared,
        )
    });

    let ddeps = wavenumber.mapv(|w| {
        material.relative_permittivity_derivative(
            Scalar(w),
            DerivativeOrder::Second,
            SpectralVariable::FrequencySquared,
        )
    });

    let dmu = wavenumber.mapv(|w| {
        material.relative_permeability_derivative(
            Scalar(w),
            DerivativeOrder::First,
            SpectralVariable::FrequencySquared,
        )
    });

    let ddmu = wavenumber.mapv(|w| {
        material.relative_permeability_derivative(
            Scalar(w),
            DerivativeOrder::Second,
            SpectralVariable::FrequencySquared,
        )
    });

    // q = ε μ ω² - β²
    //
    // dq/dω² =
    //     (ε' μ + ε μ') ω² + ε μ
    //
    // d²q/d(ω²)² =
    //     (ε'' μ + 2 ε' μ' + ε μ'') ω²
    //     + 2(ε' μ + ε μ')
    let a = deps.clone() * q.mu.view() + q.epsilon.clone() * dmu.view();

    let da = ddeps.clone() * q.mu.view()
        + deps.clone() * dmu.view()
        + deps.clone() * dmu.view()
        + q.epsilon.clone() * ddmu.view();

    let dq = a.clone() * omega2.view() + q.epsilon.clone() * q.mu.view();
    let ddq = da * omega2 + a.mapv(|x| x + x);

    let two = C::one() + C::one();

    let dkappa = dq.clone() / q.kappa.mapv(|k| two * k);

    // κ = sqrt(q)
    // κ'  = q' / (2κ)
    // κ'' = q'' / (2κ) - (q')² / (4κ³)
    let four = two + two;
    let ddkappa =
        ddq / q.kappa.mapv(|k| two * k) - dq.mapv(|x| x * x) / q.kappa.mapv(|k| four * k * k * k);

    let (dfactor, ddfactor) = match polarisation {
        Polarisation::TransverseElectric => (dmu, ddmu),
        Polarisation::TransverseMagnetic => (deps, ddeps),
    };

    IsotropicLayerSecondDerivatives {
        dkappa,
        ddkappa,
        dfactor,
        ddfactor,
    }
}

pub fn propagation_constant_squared_second_derivatives<M, C, D>(
    material: &M,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> IsotropicLayerSecondDerivatives<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let two = C::one() + C::one();
    let four = two + two;

    // q = ε μ ω² - β², with x = β²
    // dq/dx = -1, d²q/dx² = 0
    //
    // κ'  = -1/(2κ)
    // κ'' = -1/(4κ³)
    let dkappa = q.kappa.mapv(|k| -C::one() / (two * k));
    let ddkappa = q.kappa.mapv(|k| -C::one() / (four * k * k * k));

    let zero = q.factor.mapv(|_| C::zero());

    IsotropicLayerSecondDerivatives {
        dkappa,
        ddkappa,
        dfactor: zero.clone(),
        ddfactor: zero,
    }
}

pub fn isotropic_layer_matrix_from_second_derivatives<M, C, D>(
    material: &M,
    thickness: Thickness<C::RealField>,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
    derivatives: IsotropicLayerSecondDerivatives<C, D>,
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

    let dk = derivatives.dkappa;
    let ddk = derivatives.ddkappa;
    let df = derivatives.dfactor;
    let ddf = derivatives.ddfactor;

    let dtheta = dk.clone() * d;
    let ddtheta = ddk.clone() * d;

    let d2_cos = -coskd.clone() * dtheta.mapv(|x| x * x) - sinkd.clone() * ddtheta.clone();

    let d_sin = coskd.clone() * dtheta.clone();
    let d2_sin = -sinkd.clone() * dtheta.mapv(|x| x * x) + coskd.clone() * ddtheta;

    let k = q.kappa;
    let f = q.factor;

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
    let ddb = ddk / f.view()
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

    Matrix2::new(d2_cos.clone(), m12, m21, d2_cos)
}

pub fn frequency_squared_second_derivative<M, C, D>(
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
    let derivatives = frequency_squared_second_derivatives(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    isotropic_layer_matrix_from_second_derivatives(
        material,
        thickness,
        wavenumber,
        propagation_constant_squared,
        polarisation,
        derivatives,
    )
}

pub fn propagation_constant_squared_second_derivative<M, C, D>(
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
    let derivatives = propagation_constant_squared_second_derivatives(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    isotropic_layer_matrix_from_second_derivatives(
        material,
        thickness,
        wavenumber,
        propagation_constant_squared,
        polarisation,
        derivatives,
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0, arr1};
    use num_complex::Complex64;

    use super::*;
    use crate::{backend::transfer2::thickness::isotropic_layer_matrix, material::Constant};

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
    fn frequency_squared_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let omega2 = 1000.0_f64.powi(2);
        let h = 1e-2 * omega2;

        let beta2 = arr0(c(100.0));

        let analytical = frequency_squared_derivative(
            &material,
            thickness,
            &arr0(c(omega2.sqrt())),
            &beta2,
            Polarisation::TransverseElectric,
        );

        let plus = isotropic_layer_matrix(
            &material,
            thickness,
            &arr0(c((omega2 + h).sqrt())),
            &beta2,
            Polarisation::TransverseElectric,
        );

        let minus = isotropic_layer_matrix(
            &material,
            thickness,
            &arr0(c((omega2 - h).sqrt())),
            &beta2,
            Polarisation::TransverseElectric,
        );

        let expected = (&plus.add(&(&minus).scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-8);
    }

    #[test]
    fn propagation_constant_squared_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let wavenumber = arr0(c(1000.0));
        let beta2 = 100.0;
        let h = 1e-3;

        let analytical = propagation_constant_squared_derivative(
            &material,
            thickness,
            &wavenumber,
            &arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let plus = isotropic_layer_matrix(
            &material,
            thickness,
            &wavenumber,
            &arr0(c(beta2 + h)),
            Polarisation::TransverseElectric,
        );

        let minus = isotropic_layer_matrix(
            &material,
            thickness,
            &wavenumber,
            &arr0(c(beta2 - h)),
            Polarisation::TransverseElectric,
        );

        let expected = (&plus.add(&(&minus).scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-8);
    }

    #[test]
    fn frequency_squared_second_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let omega2 = 1000.0_f64.powi(2);
        let h = 1e-2 * omega2;
        let beta2 = arr0(c(100.0));

        let analytical = frequency_squared_second_derivative(
            &material,
            thickness,
            &arr0(c(omega2.sqrt())),
            &beta2,
            Polarisation::TransverseElectric,
        );

        let plus = isotropic_layer_matrix(
            &material,
            thickness,
            &arr0(c((omega2 + h).sqrt())),
            &beta2,
            Polarisation::TransverseElectric,
        );

        let zero = isotropic_layer_matrix(
            &material,
            thickness,
            &arr0(c(omega2.sqrt())),
            &beta2,
            Polarisation::TransverseElectric,
        );

        let minus = isotropic_layer_matrix(
            &material,
            thickness,
            &arr0(c((omega2 - h).sqrt())),
            &beta2,
            Polarisation::TransverseElectric,
        );

        let expected = (&plus.add(&(&zero).scale(c(-2.0))).add(&minus)).scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected, 1e-4);
    }

    #[test]
    fn propagation_constant_squared_second_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let wavenumber = arr0(c(1000.0));
        let beta2 = 100.0;
        let h = 1e-2;

        let analytical = propagation_constant_squared_second_derivative(
            &material,
            thickness,
            &wavenumber,
            &arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let plus = isotropic_layer_matrix(
            &material,
            thickness,
            &wavenumber,
            &arr0(c(beta2 + h)),
            Polarisation::TransverseElectric,
        );

        let zero = isotropic_layer_matrix(
            &material,
            thickness,
            &wavenumber,
            &arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let minus = isotropic_layer_matrix(
            &material,
            thickness,
            &wavenumber,
            &arr0(c(beta2 - h)),
            Polarisation::TransverseElectric,
        );

        let expected = (&plus.add(&(&zero).scale(c(-2.0))).add(&minus)).scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected, 1e-5);
    }
}
