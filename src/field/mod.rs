mod cartesian;

pub(crate) use cartesian::{
    CartesianScalarAlgebra, CartesianVector3, CartesianVectorAlgebra, RealCartesianVectorAlgebra,
};

use ndarray::Dimension;
use num_traits::One;

use crate::algebra::{Jet, JetBivariate, JetFirst};

pub(crate) type CartesianField<C, D> = CartesianElectromagneticField<CartesianVector3<C, D>>;

pub(super) type CartesianFieldFirst<C, D, P> =
    CartesianElectromagneticField<JetFirst<CartesianVector3<C, D>, P>>;
pub(super) type CartesianFieldSecond<C, D, P> =
    CartesianElectromagneticField<Jet<CartesianVector3<C, D>, P>>;
pub(super) type CartesianFieldBivariate<C, D, P> =
    CartesianElectromagneticField<JetBivariate<CartesianVector3<C, D>, P>>;

/// Pointwise Cartesian electric and magnetic phasor fields.
///
/// The field uses the electromagnetic normalization chosen by the producing
/// backend. The electric and magnetic vectors share the same ndarray sampling
/// shape.
///
/// The complex Poynting vector uses:
///
/// ```text
/// S = 1/2 E × H*
/// ```
///
/// and the time-averaged Poynting vector is its real part.
#[derive(Clone, Debug, PartialEq)]
pub struct CartesianElectromagneticField<V> {
    electric: V,
    magnetic: V,
}

impl<V> CartesianElectromagneticField<V> {
    pub(crate) fn new(electric: V, magnetic: V) -> Self {
        Self { electric, magnetic }
    }

    pub fn electric(&self) -> &V {
        &self.electric
    }

    pub fn magnetic(&self) -> &V {
        &self.magnetic
    }

    pub fn into_parts(self) -> (V, V) {
        (self.electric, self.magnetic)
    }

    pub fn map_vectors<U>(self, f: impl Fn(V) -> U) -> CartesianElectromagneticField<U> {
        CartesianElectromagneticField {
            electric: f(self.electric),
            magnetic: f(self.magnetic),
        }
    }
}

impl<V> CartesianElectromagneticField<V> {
    /// Return the pointwise squared electric-field magnitude.
    pub fn electric_magnitude_squared(&self) -> V::RealScalarField
    where
        V: RealCartesianVectorAlgebra,
    {
        self.electric.magnitude_squared()
    }

    /// Return the pointwise squared magnetic-field magnitude.
    pub fn magnetic_magnitude_squared(&self) -> V::RealScalarField
    where
        V: RealCartesianVectorAlgebra,
    {
        self.magnetic.magnitude_squared()
    }

    /// Return the pointwise complex Poynting vector.
    ///
    /// This evaluates `1/2 E × H*`.
    pub fn complex_poynting_vector(&self) -> V
    where
        V: RealCartesianVectorAlgebra,
        V::Coefficient: One
            + Copy
            + std::ops::Add<Output = V::Coefficient>
            + std::ops::Div<Output = V::Coefficient>,
    {
        self.electric
            .cross(&self.magnetic.conjugated())
            .scale_by_constant(half())
    }

    /// Return the time average poynting vector
    pub fn time_averaged_poynting_vector(&self) -> V::RealVector
    where
        V: RealCartesianVectorAlgebra,
        V::Coefficient: One
            + Copy
            + std::ops::Add<Output = V::Coefficient>
            + std::ops::Div<Output = V::Coefficient>,
    {
        self.complex_poynting_vector().real()
    }

    /// Return the pointwise cycle-averaged electric energy density.
    ///
    /// This evaluates:
    ///
    /// ```text
    /// uₑ = 1/4 Re(E · D*)
    /// ```
    ///
    /// `displacement` is the electric displacement field `D` represented
    /// using the same sampling domain and derivative representation as `E`.
    ///
    /// This is the conventional nondispersive expression. Energy density in
    /// a dispersive medium generally requires frequency derivatives of the
    /// constitutive response and should be computed at a higher layer.
    pub fn electric_energy_density(&self, displacement: &V) -> V::RealScalarField
    where
        V: RealCartesianVectorAlgebra,
        V::Coefficient: One
            + Copy
            + std::ops::Add<Output = V::Coefficient>
            + std::ops::Div<Output = V::Coefficient>,
    {
        let scaled_electric = self.electric.scale_by_constant(quarter::<V::Coefficient>());

        V::scalar_real(scaled_electric.hermitian_dot(displacement))
    }

    /// Return the pointwise cycle-averaged magnetic energy density.
    ///
    /// This evaluates:
    ///
    /// ```text
    /// uₘ = 1/4 Re(H · B*)
    /// ```
    ///
    /// `magnetic_flux_density` is the magnetic flux-density field `B`
    /// represented using the same sampling domain and derivative
    /// representation as `H`.
    ///
    /// This is the conventional nondispersive expression. Energy density in
    /// a dispersive medium generally requires frequency derivatives of the
    /// constitutive response and should be computed at a higher layer.
    pub fn magnetic_energy_density(&self, magnetic_flux_density: &V) -> V::RealScalarField
    where
        V: RealCartesianVectorAlgebra,
        V::Coefficient: One
            + Copy
            + std::ops::Add<Output = V::Coefficient>
            + std::ops::Div<Output = V::Coefficient>,
    {
        let scaled_magnetic = self.magnetic.scale_by_constant(quarter::<V::Coefficient>());

        V::scalar_real(scaled_magnetic.hermitian_dot(magnetic_flux_density))
    }

    /// Return the pointwise cycle-averaged electromagnetic energy density.
    ///
    /// This evaluates:
    ///
    /// ```text
    /// u = 1/4 Re(E · D* + H · B*)
    /// ```
    ///
    /// This is the conventional nondispersive expression.
    pub fn energy_density(&self, displacement: &V, magnetic_flux_density: &V) -> V::RealScalarField
    where
        V: RealCartesianVectorAlgebra,
        V::Coefficient: One
            + Copy
            + std::ops::Add<Output = V::Coefficient>
            + std::ops::Div<Output = V::Coefficient>,
        V::RealScalarField: std::ops::Add<Output = V::RealScalarField>,
    {
        self.electric_energy_density(displacement)
            + self.magnetic_energy_density(magnetic_flux_density)
    }
}

impl<C, D, P> CartesianFieldFirst<C, D, P>
where
    D: Dimension,
{
    pub(super) fn split(self) -> (CartesianField<C, D>, CartesianField<C, D>) {
        let (electric, magnetic) = self.into_parts();

        let (electric_value, electric_first) = electric.into_parts();
        let (magnetic_value, magnetic_first) = magnetic.into_parts();

        (
            CartesianField::new(electric_value, magnetic_value),
            CartesianField::new(electric_first, magnetic_first),
        )
    }
}

impl<C, D, P> CartesianFieldSecond<C, D, P>
where
    D: Dimension,
{
    pub(super) fn split(
        self,
    ) -> (
        CartesianField<C, D>,
        CartesianField<C, D>,
        CartesianField<C, D>,
    ) {
        let (electric, magnetic) = self.into_parts();

        let (electric_value, electric_first, electric_second) = electric.into_parts();
        let (magnetic_value, magnetic_first, magnetic_second) = magnetic.into_parts();

        (
            CartesianField::new(electric_value, magnetic_value),
            CartesianField::new(electric_first, magnetic_first),
            CartesianField::new(electric_second, magnetic_second),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CartesianFieldBivariateParts<F> {
    pub value: F,
    pub x: F,
    pub y: F,
    pub xx: F,
    pub xy: F,
    pub yy: F,
}

impl<C, D, P> CartesianFieldBivariate<C, D, P>
where
    D: Dimension,
{
    pub(super) fn split(self) -> CartesianFieldBivariateParts<CartesianField<C, D>> {
        let (electric, magnetic) = self.into_parts();

        let (electric_value, electric_gradient, electric_hessian) = electric.into_parts();
        let (magnetic_value, magnetic_gradient, magnetic_hessian) = magnetic.into_parts();

        let (electric_x, electric_y) = electric_gradient.into_parts();
        let (magnetic_x, magnetic_y) = magnetic_gradient.into_parts();
        let (electric_xx, electric_xy, electric_yy) = electric_hessian.into_parts();
        let (magnetic_xx, magnetic_xy, magnetic_yy) = magnetic_hessian.into_parts();
        CartesianFieldBivariateParts {
            value: CartesianField::new(electric_value, magnetic_value),
            x: CartesianField::new(electric_x, magnetic_x),
            y: CartesianField::new(electric_y, magnetic_y),
            xx: CartesianField::new(electric_xx, magnetic_xx),
            xy: CartesianField::new(electric_xy, magnetic_xy),
            yy: CartesianField::new(electric_yy, magnetic_yy),
        }
    }
}

fn quarter<C>() -> C
where
    C: One + Copy + std::ops::Add<Output = C> + std::ops::Div<Output = C>,
{
    let one = C::one();
    let two = one + one;

    one / (two + two)
}

fn half<C>() -> C
where
    C: One + Copy + std::ops::Add<Output = C> + std::ops::Div<Output = C>,
{
    let one = C::one();

    one / (one + one)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;

    use crate::algebra::{Jet, JetBivariate, JetFirst, RealParameter};

    use super::*;

    type C = Complex64;
    type D = Ix1;
    type Vector = CartesianVector3<C, D>;
    type Field = CartesianElectromagneticField<Vector>;

    type FirstVector = JetFirst<Vector, RealParameter>;
    type FirstField = CartesianElectromagneticField<FirstVector>;

    type SecondVector = Jet<Vector, RealParameter>;
    type SecondField = CartesianElectromagneticField<SecondVector>;

    type BivariateVector = JetBivariate<Vector, RealParameter>;
    type BivariateField = CartesianElectromagneticField<BivariateVector>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn scalar_vector(value: f64) -> Vector {
        CartesianVector3::new(
            arr1(&[c(value, 0.0)]),
            arr1(&[c(value + 0.1, 0.0)]),
            arr1(&[c(value + 0.2, 0.0)]),
        )
    }

    fn vector(x: &[C], y: &[C], z: &[C]) -> Vector {
        CartesianVector3::new(
            Array1::from_vec(x.to_vec()),
            Array1::from_vec(y.to_vec()),
            Array1::from_vec(z.to_vec()),
        )
    }

    fn assert_real_close(actual: f64, expected: f64) {
        let error = (actual - expected).abs();

        assert!(
            error <= TOLERANCE,
            "expected {expected:e}, \
             got {actual:e}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_close(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, \
             got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_real_array_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim(),);

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_real_close(actual, expected);
        }
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

    fn assert_field_equals(actual: &Field, electric: &Vector, magnetic: &Vector) {
        assert_vector_close(actual.electric(), electric);

        assert_vector_close(actual.magnetic(), magnetic);
    }

    #[test]
    fn construction_preserves_electric_and_magnetic_fields() {
        let electric = scalar_vector(1.0);
        let magnetic = scalar_vector(2.0);

        let field = Field::new(electric.clone(), magnetic.clone());

        assert_eq!(field.electric(), &electric,);

        assert_eq!(field.magnetic(), &magnetic,);

        assert_eq!(field.into_parts(), (electric, magnetic),);
    }

    #[test]
    fn magnitude_methods_use_corresponding_vector() {
        let electric = vector(
            &[c(3.0, 4.0), c(1.0, 0.0)],
            &[c(0.0, 2.0), c(0.0, 2.0)],
            &[c(1.0, 0.0), c(2.0, 0.0)],
        );

        let magnetic = vector(
            &[c(1.0, 0.0), c(2.0, 0.0)],
            &[c(2.0, 0.0), c(3.0, 0.0)],
            &[c(3.0, 0.0), c(4.0, 0.0)],
        );

        let field = Field::new(electric, magnetic);

        assert_real_array_close(&field.electric_magnitude_squared(), &arr1(&[30.0, 9.0]));

        assert_real_array_close(&field.magnetic_magnitude_squared(), &arr1(&[14.0, 29.0]));
    }

    #[test]
    fn complex_poynting_is_half_e_cross_conjugate_h() {
        let zero = c(0.0, 0.0);

        let electric = vector(&[c(1.0, 1.0)], &[zero], &[zero]);

        let magnetic = vector(&[zero], &[c(2.0, 1.0)], &[zero]);

        let field = Field::new(electric, magnetic);

        let result = field.complex_poynting_vector();

        let expected_z = c(0.5, 0.0) * c(1.0, 1.0) * c(2.0, 1.0).conj();

        assert_complex_array_close(result.x(), &arr1(&[zero]));

        assert_complex_array_close(result.y(), &arr1(&[zero]));

        assert_complex_array_close(result.z(), &arr1(&[expected_z]));
    }

    #[test]
    fn time_averaged_poynting_is_real_part_of_complex_poynting() {
        let zero = c(0.0, 0.0);

        let field = Field::new(
            vector(&[c(1.0, 1.0)], &[zero], &[zero]),
            vector(&[zero], &[c(2.0, 1.0)], &[zero]),
        );

        let complex = field.complex_poynting_vector();

        let averaged = field.time_averaged_poynting_vector();

        assert_real_array_close(averaged.x(), &complex.x().mapv(|value| value.re));

        assert_real_array_close(averaged.y(), &complex.y().mapv(|value| value.re));

        assert_real_array_close(averaged.z(), &complex.z().mapv(|value| value.re));
    }

    #[test]
    fn energy_density_uses_constitutive_fields() {
        let zero = c(0.0, 0.0);

        let electric = vector(&[c(2.0, 0.0)], &[zero], &[zero]);

        let magnetic = vector(&[zero], &[c(4.0, 0.0)], &[zero]);

        let displacement = vector(&[c(3.0, 0.0)], &[zero], &[zero]);

        let flux_density = vector(&[zero], &[c(5.0, 0.0)], &[zero]);

        let field = Field::new(electric, magnetic);

        let electric_energy = field.electric_energy_density(&displacement);

        let magnetic_energy = field.magnetic_energy_density(&flux_density);

        let total_energy = field.energy_density(&displacement, &flux_density);

        // 1/4 * 2 * 3
        assert_real_array_close(&electric_energy, &arr1(&[1.5]));

        // 1/4 * 4 * 5
        assert_real_array_close(&magnetic_energy, &arr1(&[5.0]));

        assert_real_array_close(&total_energy, &arr1(&[6.5]));
    }

    #[test]
    fn energy_density_takes_real_part_of_complex_products() {
        let zero = c(0.0, 0.0);

        let electric = vector(&[c(1.0, 1.0)], &[zero], &[zero]);

        let displacement = vector(&[c(2.0, -1.0)], &[zero], &[zero]);

        let magnetic = vector(&[zero], &[c(1.0, -2.0)], &[zero]);

        let flux_density = vector(&[zero], &[c(3.0, 1.0)], &[zero]);

        let expected_electric = 0.25 * (c(1.0, 1.0) * c(2.0, -1.0).conj()).re;

        let expected_magnetic = 0.25 * (c(1.0, -2.0) * c(3.0, 1.0).conj()).re;

        let field = Field::new(electric, magnetic);

        assert_real_array_close(
            &field.electric_energy_density(&displacement),
            &arr1(&[expected_electric]),
        );

        assert_real_array_close(
            &field.magnetic_energy_density(&flux_density),
            &arr1(&[expected_magnetic]),
        );

        assert_real_array_close(
            &field.energy_density(&displacement, &flux_density),
            &arr1(&[expected_electric + expected_magnetic]),
        );
    }

    #[test]
    fn first_order_split_transposes_field_and_jet_layers() {
        let electric_value = scalar_vector(1.0);
        let electric_first = scalar_vector(2.0);
        let magnetic_value = scalar_vector(3.0);
        let magnetic_first = scalar_vector(4.0);

        let field: FirstField = CartesianElectromagneticField::new(
            JetFirst::from_parts(electric_value.clone(), electric_first.clone()),
            JetFirst::from_parts(magnetic_value.clone(), magnetic_first.clone()),
        );

        let (value, first) = field.split();

        assert_field_equals(&value, &electric_value, &magnetic_value);

        assert_field_equals(&first, &electric_first, &magnetic_first);
    }

    #[test]
    fn second_order_split_transposes_field_and_jet_layers() {
        let electric_value = scalar_vector(1.0);
        let electric_first = scalar_vector(2.0);
        let electric_second = scalar_vector(3.0);

        let magnetic_value = scalar_vector(4.0);
        let magnetic_first = scalar_vector(5.0);
        let magnetic_second = scalar_vector(6.0);

        let field: SecondField = CartesianElectromagneticField::new(
            Jet::from_parts(
                electric_value.clone(),
                electric_first.clone(),
                electric_second.clone(),
            ),
            Jet::from_parts(
                magnetic_value.clone(),
                magnetic_first.clone(),
                magnetic_second.clone(),
            ),
        );

        let (value, first, second) = field.split();

        assert_field_equals(&value, &electric_value, &magnetic_value);

        assert_field_equals(&first, &electric_first, &magnetic_first);

        assert_field_equals(&second, &electric_second, &magnetic_second);
    }

    #[test]
    fn bivariate_split_transposes_all_field_and_jet_components() {
        let electric_value = scalar_vector(1.0);
        let electric_x = scalar_vector(2.0);
        let electric_y = scalar_vector(3.0);
        let electric_xx = scalar_vector(4.0);
        let electric_xy = scalar_vector(5.0);
        let electric_yy = scalar_vector(6.0);

        let magnetic_value = scalar_vector(7.0);
        let magnetic_x = scalar_vector(8.0);
        let magnetic_y = scalar_vector(9.0);
        let magnetic_xx = scalar_vector(10.0);
        let magnetic_xy = scalar_vector(11.0);
        let magnetic_yy = scalar_vector(12.0);

        let field: BivariateField = CartesianElectromagneticField::new(
            JetBivariate::from_components(
                electric_value.clone(),
                electric_x.clone(),
                electric_y.clone(),
                electric_xx.clone(),
                electric_xy.clone(),
                electric_yy.clone(),
            ),
            JetBivariate::from_components(
                magnetic_value.clone(),
                magnetic_x.clone(),
                magnetic_y.clone(),
                magnetic_xx.clone(),
                magnetic_xy.clone(),
                magnetic_yy.clone(),
            ),
        );

        let parts = field.split();

        assert_field_equals(&parts.value, &electric_value, &magnetic_value);

        assert_field_equals(&parts.x, &electric_x, &magnetic_x);

        assert_field_equals(&parts.y, &electric_y, &magnetic_y);

        assert_field_equals(&parts.xx, &electric_xx, &magnetic_xx);

        assert_field_equals(&parts.xy, &electric_xy, &magnetic_xy);

        assert_field_equals(&parts.yy, &electric_yy, &magnetic_yy);
    }

    #[test]
    fn first_order_complex_poynting_obeys_product_rule() {
        let zero = c(0.0, 0.0);

        let electric_value = vector(&[c(1.0, 1.0)], &[zero], &[zero]);

        let electric_first = vector(&[c(2.0, -1.0)], &[zero], &[zero]);

        let magnetic_value = vector(&[zero], &[c(3.0, 2.0)], &[zero]);

        let magnetic_first = vector(&[zero], &[c(-1.0, 4.0)], &[zero]);

        let electric = JetFirst::from_parts(electric_value.clone(), electric_first.clone());

        let magnetic = JetFirst::from_parts(magnetic_value.clone(), magnetic_first.clone());

        let field: FirstField = CartesianElectromagneticField::new(electric, magnetic);

        let result = field.complex_poynting_vector();

        let expected_value = electric_value.cross(&magnetic_value.conjugate()) * c(0.5, 0.0);

        let expected_first = (electric_first.cross(&magnetic_value.conjugate())
            + electric_value.cross(&magnetic_first.conjugate()))
            * c(0.5, 0.0);

        assert_vector_close(result.value(), &expected_value);

        assert_vector_close(result.first(), &expected_first);
    }
}
