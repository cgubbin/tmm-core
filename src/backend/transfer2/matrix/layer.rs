//! Isotropic finite-layer transfer matrices.
//!
//! This module constructs the transfer matrix of one homogeneous isotropic
//! layer and its derivatives.
//!
//! For a layer of physical thickness `d`, normal wavenumber `κ`, and
//! polarisation-dependent factor `f`, the matrix is:
//!
//! ```text
//!       [ cos(κd)   -sin(κd) f/κ ]
//! M  =  [                         ]
//!       [ sin(κd) κ/f   cos(κd)  ]
//! ```
//!
//! Thickness derivatives are taken with respect to thickness measured in the
//! backend's canonical length unit, currently centimetres.
//!
//! Spectral derivatives are taken with respect to the primitive coordinate
//! associated with the supplied isotropic derivative quantities.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::isotropic::{
        IsotropicLayerFirstDerivatives, IsotropicLayerQuantities, IsotropicLayerSecondDerivatives,
    },
    stack::Thickness,
};

use super::Matrix2;

/// Trigonometric quantities shared by a layer matrix and its derivatives.
struct LayerPhase<C, D>
where
    D: Dimension,
{
    sin: ArrayBase<OwnedRepr<C>, D>,
    cos: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> LayerPhase<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn new(kappa: &ArrayBase<OwnedRepr<C>, D>, thickness_cm: C) -> Self {
        let theta = kappa.mapv(|value| value * thickness_cm);

        Self {
            sin: theta.mapv(|value| value.sin()),
            cos: theta.mapv(|value| value.cos()),
        }
    }
}

impl<C, D> Matrix2<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    /// Construct the transfer matrix of one homogeneous isotropic layer.
    pub(crate) fn from_layer(
        quantities: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
    ) -> Self {
        let thickness_cm = C::from_real(thickness.as_cm());

        let phase = LayerPhase::new(quantities.kappa(), thickness_cm);

        Self::new(
            phase.cos.clone(),
            -phase.sin.clone() * quantities.factor().view() / quantities.kappa().view(),
            phase.sin * quantities.kappa().view() / quantities.factor().view(),
            phase.cos,
        )
    }

    /// Compute the first derivative with respect to physical thickness.
    ///
    /// Thickness is differentiated in the canonical centimetre coordinate.
    pub(crate) fn thickness_derivative(
        quantities: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
    ) -> Self {
        let thickness_cm = C::from_real(thickness.as_cm());

        let kappa = quantities.kappa();
        let factor = quantities.factor();

        let phase = LayerPhase::new(kappa, thickness_cm);

        let kappa_squared = kappa.mapv(|value| value * value);

        Self::new(
            -kappa.clone() * phase.sin.clone(),
            -factor.clone() * phase.cos.clone(),
            kappa_squared * phase.cos / factor.view(),
            -phase.sin * kappa.view(),
        )
    }

    /// Compute the second derivative with respect to physical thickness.
    ///
    /// Thickness is differentiated in the canonical centimetre coordinate.
    pub(crate) fn thickness_second_derivative(
        quantities: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
    ) -> Self {
        let thickness_cm = C::from_real(thickness.as_cm());

        let kappa = quantities.kappa();
        let factor = quantities.factor();

        let phase = LayerPhase::new(kappa, thickness_cm);

        let kappa_squared = kappa.mapv(|value| value * value);

        let kappa_cubed = kappa.mapv(|value| value * value * value);

        Self::new(
            -kappa_squared.clone() * phase.cos.clone(),
            kappa.clone() * factor.view() * phase.sin.clone(),
            -kappa_cubed * phase.sin / factor.view(),
            -kappa_squared * phase.cos,
        )
    }

    /// Compute the first layer-matrix derivative with respect to a primitive
    /// spectral coordinate.
    ///
    /// `derivatives` must describe derivatives of `quantities` with respect to
    /// the same coordinate and at the same evaluation point.
    pub(crate) fn spectral_derivative(
        quantities: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
        derivatives: &IsotropicLayerFirstDerivatives<C, D>,
    ) -> Self {
        let thickness_cm = C::from_real(thickness.as_cm());

        let kappa = quantities.kappa();
        let factor = quantities.factor();

        let dkappa = derivatives.dkappa();
        let dfactor = derivatives.dfactor();

        let phase = LayerPhase::new(kappa, thickness_cm);

        let dtheta = dkappa.clone() * thickness_cm;

        let d_sin = phase.cos.clone() * dtheta.view();

        let d_cos = -phase.sin.clone() * dtheta;

        let kappa_squared = kappa.mapv(|value| value * value);

        let factor_squared = factor.mapv(|value| value * value);

        // d(f/κ) = f′/κ - fκ′/κ²
        let d_factor_over_kappa =
            dfactor.clone() / kappa.view() - factor.clone() * dkappa.view() / kappa_squared.view();

        // d(κ/f) = κ′/f - κf′/f²
        let d_kappa_over_factor =
            dkappa.clone() / factor.view() - kappa.clone() * dfactor.view() / factor_squared.view();

        Self::new(
            d_cos.clone(),
            -(d_sin.clone() * factor.view() / kappa.view()
                + phase.sin.clone() * d_factor_over_kappa),
            d_sin * kappa.view() / factor.view() + phase.sin * d_kappa_over_factor,
            d_cos,
        )
    }

    /// Compute the second layer-matrix derivative with respect to a primitive
    /// spectral coordinate.
    ///
    /// `derivatives` must contain first and second derivatives of the isotropic
    /// quantities with respect to the same coordinate.
    pub(crate) fn spectral_second_derivative(
        quantities: &IsotropicLayerQuantities<C, D>,
        thickness: Thickness<C::RealField>,
        derivatives: &IsotropicLayerSecondDerivatives<C, D>,
    ) -> Self {
        let thickness_cm = C::from_real(thickness.as_cm());

        let kappa = quantities.kappa();
        let factor = quantities.factor();

        let dkappa = derivatives.first().dkappa();

        let ddkappa = derivatives.ddkappa();

        let dfactor = derivatives.first().dfactor();

        let ddfactor = derivatives.ddfactor();

        let phase = LayerPhase::new(kappa, thickness_cm);

        let dtheta = dkappa.clone() * thickness_cm;

        let ddtheta = ddkappa.clone() * thickness_cm;

        let dtheta_squared = dtheta.mapv(|value| value * value);

        let d_sin = phase.cos.clone() * dtheta.view();

        let dd_sin =
            -phase.sin.clone() * dtheta_squared.view() + phase.cos.clone() * ddtheta.view();

        let dd_cos = -phase.cos.clone() * dtheta_squared - phase.sin.clone() * ddtheta;

        let kappa_squared = kappa.mapv(|value| value * value);

        let kappa_cubed = kappa.mapv(|value| value * value * value);

        let factor_squared = factor.mapv(|value| value * value);

        let factor_cubed = factor.mapv(|value| value * value * value);

        // A = f/κ
        let factor_over_kappa = factor.clone() / kappa.view();

        let d_factor_over_kappa =
            dfactor.clone() / kappa.view() - factor.clone() * dkappa.view() / kappa_squared.view();

        let dd_factor_over_kappa = ddfactor.clone() / kappa.view()
            - factor.clone() * ddkappa.view() / kappa_squared.view()
            - (dfactor.clone() * dkappa.view()).mapv(|value| value + value) / kappa_squared.view()
            + (factor.clone() * dkappa.mapv(|value| value * value)).mapv(|value| value + value)
                / kappa_cubed.view();

        // B = κ/f
        let kappa_over_factor = kappa.clone() / factor.view();

        let d_kappa_over_factor =
            dkappa.clone() / factor.view() - kappa.clone() * dfactor.view() / factor_squared.view();

        let dd_kappa_over_factor = ddkappa.clone() / factor.view()
            - kappa.clone() * ddfactor.view() / factor_squared.view()
            - (dkappa.clone() * dfactor.view()).mapv(|value| value + value) / factor_squared.view()
            + (kappa.clone() * dfactor.mapv(|value| value * value)).mapv(|value| value + value)
                / factor_cubed.view();

        let two = C::one() + C::one();

        // m12 = -sin(θ) A
        let m12 = -(dd_sin.clone() * factor_over_kappa
            + (d_sin.clone() * d_factor_over_kappa).mapv(|value| two * value)
            + phase.sin.clone() * dd_factor_over_kappa);

        // m21 = sin(θ) B
        let m21 = dd_sin * kappa_over_factor
            + (d_sin * d_kappa_over_factor).mapv(|value| two * value)
            + phase.sin * dd_kappa_over_factor;

        Self::new(dd_cos.clone(), m12, m21, dd_cos)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Ix0, arr0, array};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        backend::{
            PlanarInput, Polarisation,
            isotropic::{
                IsotropicLayerFirstDerivatives, IsotropicLayerQuantities,
                IsotropicLayerSecondDerivatives,
            },
        },
        material::Constant,
    };

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn material(epsilon: f64, mu: f64) -> Constant<f64> {
        Constant::new(epsilon, mu)
    }

    fn thickness(value_cm: f64) -> Thickness<f64> {
        Thickness::from_cm(value_cm).unwrap()
    }

    fn make_input(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<Array0<C>> {
        PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            polarisation,
        )
    }

    fn assert_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    fn assert_matrix_close(actual: &Matrix2<C, Ix0>, expected: &Matrix2<C, Ix0>, tolerance: f64) {
        assert_close(actual.m11()[()], expected.m11()[()], tolerance);
        assert_close(actual.m12()[()], expected.m12()[()], tolerance);
        assert_close(actual.m21()[()], expected.m21()[()], tolerance);
        assert_close(actual.m22()[()], expected.m22()[()], tolerance);
    }

    fn finite_difference_first(
        plus: &Matrix2<C, Ix0>,
        minus: &Matrix2<C, Ix0>,
        step: f64,
    ) -> Matrix2<C, Ix0> {
        let scale = c(1.0 / (2.0 * step));

        &(plus - minus) * scale
    }

    fn finite_difference_second(
        plus: &Matrix2<C, Ix0>,
        zero: &Matrix2<C, Ix0>,
        minus: &Matrix2<C, Ix0>,
        step: f64,
    ) -> Matrix2<C, Ix0> {
        let twice_zero = zero * c(2.0);

        let numerator = &(plus - &twice_zero) + minus;

        &numerator * c(1.0 / (step * step))
    }

    #[test]
    fn zero_thickness_layer_is_identity() {
        let material = material(2.25, 1.0);

        let input = make_input(3.0, 0.7, Polarisation::TransverseElectric);

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let matrix = Matrix2::from_layer(&quantities, thickness(0.0));

        let identity = Matrix2::identity_like(input.vacuum_wavenumber());

        assert_matrix_close(&matrix, &identity, 1e-12);
    }

    #[test]
    fn layer_matrix_has_unit_determinant() {
        let material = material(2.25, 1.4);

        let input = make_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let matrix = Matrix2::from_layer(&quantities, thickness(0.2));

        assert_close(matrix.determinant()[()], c(1.0), 1e-12);
    }

    #[test]
    fn thickness_first_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let input = make_input(3.0, 0.7, Polarisation::TransverseElectric);

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let d = 0.2;
        let h = 1e-6;

        let analytic = Matrix2::thickness_derivative(&quantities, thickness(d));

        let plus = Matrix2::from_layer(&quantities, thickness(d + h));

        let minus = Matrix2::from_layer(&quantities, thickness(d - h));

        let expected = finite_difference_first(&plus, &minus, h);

        assert_matrix_close(&analytic, &expected, 1e-7);
    }

    #[test]
    fn thickness_second_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let input = make_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let d = 0.2;
        let h = 1e-4;

        let analytic = Matrix2::thickness_second_derivative(&quantities, thickness(d));

        let plus = Matrix2::from_layer(&quantities, thickness(d + h));

        let zero = Matrix2::from_layer(&quantities, thickness(d));

        let minus = Matrix2::from_layer(&quantities, thickness(d - h));

        let expected = finite_difference_second(&plus, &zero, &minus, h);

        assert_matrix_close(&analytic, &expected, 2e-6);
    }

    #[test]
    fn vacuum_wavenumber_squared_first_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let k0_squared: f64 = 9.0;
        let parallel = 0.7;
        let layer_thickness = thickness(0.2);
        let h = 1e-5;

        let input = make_input(
            k0_squared.sqrt(),
            parallel,
            Polarisation::TransverseElectric,
        );

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let derivatives = IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &quantities,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        let analytic = Matrix2::spectral_derivative(&quantities, layer_thickness, &derivatives);

        let plus_input = make_input(
            (k0_squared + h).sqrt(),
            parallel,
            Polarisation::TransverseElectric,
        );

        let minus_input = make_input(
            (k0_squared - h).sqrt(),
            parallel,
            Polarisation::TransverseElectric,
        );

        let plus_quantities = IsotropicLayerQuantities::real_axis(&material, &plus_input);

        let minus_quantities = IsotropicLayerQuantities::real_axis(&material, &minus_input);

        let plus = Matrix2::from_layer(&plus_quantities, layer_thickness);

        let minus = Matrix2::from_layer(&minus_quantities, layer_thickness);

        let expected = finite_difference_first(&plus, &minus, h);

        assert_matrix_close(&analytic, &expected, 2e-7);
    }

    #[test]
    fn vacuum_wavenumber_squared_second_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let k0_squared: f64 = 9.0;
        let parallel = 0.7;
        let layer_thickness = thickness(0.2);
        let h = 2e-3;

        let input = make_input(
            k0_squared.sqrt(),
            parallel,
            Polarisation::TransverseMagnetic,
        );

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let derivatives = IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &quantities,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        let analytic =
            Matrix2::spectral_second_derivative(&quantities, layer_thickness, &derivatives);

        let plus_input = make_input(
            (k0_squared + h).sqrt(),
            parallel,
            Polarisation::TransverseMagnetic,
        );

        let zero_input = make_input(
            k0_squared.sqrt(),
            parallel,
            Polarisation::TransverseMagnetic,
        );

        let minus_input = make_input(
            (k0_squared - h).sqrt(),
            parallel,
            Polarisation::TransverseMagnetic,
        );

        let plus_quantities = IsotropicLayerQuantities::real_axis(&material, &plus_input);

        let zero_quantities = IsotropicLayerQuantities::real_axis(&material, &zero_input);

        let minus_quantities = IsotropicLayerQuantities::real_axis(&material, &minus_input);

        let plus = Matrix2::from_layer(&plus_quantities, layer_thickness);

        let zero = Matrix2::from_layer(&zero_quantities, layer_thickness);

        let minus = Matrix2::from_layer(&minus_quantities, layer_thickness);

        let expected = finite_difference_second(&plus, &zero, &minus, h);

        assert_matrix_close(&analytic, &expected, 3e-6);
    }

    #[test]
    fn parallel_wavenumber_squared_first_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let vacuum = 3.0;
        let parallel_squared: f64 = 0.49;
        let layer_thickness = thickness(0.2);
        let h = 1e-5;

        let input = make_input(
            vacuum,
            parallel_squared.sqrt(),
            Polarisation::TransverseElectric,
        );

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let derivatives = IsotropicLayerFirstDerivatives::parallel_wavenumber_squared(&quantities);

        let analytic = Matrix2::spectral_derivative(&quantities, layer_thickness, &derivatives);

        let plus_input = make_input(
            vacuum,
            (parallel_squared + h).sqrt(),
            Polarisation::TransverseElectric,
        );

        let minus_input = make_input(
            vacuum,
            (parallel_squared - h).sqrt(),
            Polarisation::TransverseElectric,
        );

        let plus_quantities = IsotropicLayerQuantities::real_axis(&material, &plus_input);

        let minus_quantities = IsotropicLayerQuantities::real_axis(&material, &minus_input);

        let plus = Matrix2::from_layer(&plus_quantities, layer_thickness);

        let minus = Matrix2::from_layer(&minus_quantities, layer_thickness);

        let expected = finite_difference_first(&plus, &minus, h);

        assert_matrix_close(&analytic, &expected, 2e-7);
    }

    #[test]
    fn parallel_wavenumber_squared_second_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let vacuum = 3.0;
        let parallel_squared: f64 = 0.49;
        let layer_thickness = thickness(0.2);
        let h = 2e-3;

        let input = make_input(
            vacuum,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let derivatives = IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&quantities);

        let analytic =
            Matrix2::spectral_second_derivative(&quantities, layer_thickness, &derivatives);

        let plus_input = make_input(
            vacuum,
            (parallel_squared + h).sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let zero_input = make_input(
            vacuum,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let minus_input = make_input(
            vacuum,
            (parallel_squared - h).sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let plus_quantities = IsotropicLayerQuantities::real_axis(&material, &plus_input);

        let zero_quantities = IsotropicLayerQuantities::real_axis(&material, &zero_input);

        let minus_quantities = IsotropicLayerQuantities::real_axis(&material, &minus_input);

        let plus = Matrix2::from_layer(&plus_quantities, layer_thickness);

        let zero = Matrix2::from_layer(&zero_quantities, layer_thickness);

        let minus = Matrix2::from_layer(&minus_quantities, layer_thickness);

        let expected = finite_difference_second(&plus, &zero, &minus, h);

        assert_matrix_close(&analytic, &expected, 3e-6);
    }

    #[test]
    fn array_input_shape_is_preserved() {
        let material = material(2.25, 1.4);

        let input = PlanarInput::new(
            array![c(2.0), c(2.5), c(3.0)],
            array![c(0.3), c(0.4), c(0.5)],
            Polarisation::TransverseElectric,
        );

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let matrix = Matrix2::from_layer(&quantities, thickness(0.2));

        let expected = input.vacuum_wavenumber().raw_dim();

        assert_eq!(matrix.m11().raw_dim(), expected);
        assert_eq!(matrix.m12().raw_dim(), expected);
        assert_eq!(matrix.m21().raw_dim(), expected);
        assert_eq!(matrix.m22().raw_dim(), expected);
    }
}
