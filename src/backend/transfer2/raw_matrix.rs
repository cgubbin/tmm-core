use crate::{
    ComplexScalar, SpectralDerivativeVariable, StructuralDerivativeVariable, Transfer2,
    backend::{
        ComplexPlane, MatrixEvaluation, PlanarInput, RawMatrixBackend, RealAxis,
        jet::{ArrayJet, ArrayJetFirst},
        matrix::{
            ComplexMatrixBackend, ComplexMatrixSpectralDerivativeBackend,
            ComplexMatrixStructuralDerivativeBackend, MatrixDerivatives,
            RawMatrixSpectralDerivativeBackend, RawMatrixStructuralDerivativeBackend,
        },
        transfer2::{Transfer2Error, TransferMatrix2, matrix::Matrix2Entries},
    },
    material::{
        EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial,
        EvaluateMaterial, EvaluateMeromorphicMaterial,
    },
    stack::Stack,
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

impl<C, D> Matrix2Entries<ArrayJetFirst<C, D>>
where
    D: Dimension,
{
    /// Consume first-order entry jets and return value and first-derivative
    /// transfer matrices.
    pub(crate) fn into_matrix_parts(self) -> (TransferMatrix2<C, D>, TransferMatrix2<C, D>) {
        let (m11, dm11) = self.m11.into_parts();
        let (m12, dm12) = self.m12.into_parts();
        let (m21, dm21) = self.m21.into_parts();
        let (m22, dm22) = self.m22.into_parts();

        let value = TransferMatrix2::new(m11, m12, m21, m22);

        let first = TransferMatrix2::new(dm11, dm12, dm21, dm22);

        (value, first)
    }
}

impl<C, D> Matrix2Entries<ArrayJet<C, D>>
where
    D: Dimension,
{
    /// Consume second-order entry jets and return value, first-derivative, and
    /// second-derivative transfer matrices.
    pub(crate) fn into_matrix_parts(
        self,
    ) -> (
        TransferMatrix2<C, D>,
        TransferMatrix2<C, D>,
        TransferMatrix2<C, D>,
    ) {
        let (m11, dm11, ddm11) = self.m11.into_parts();

        let (m12, dm12, ddm12) = self.m12.into_parts();

        let (m21, dm21, ddm21) = self.m21.into_parts();

        let (m22, dm22, ddm22) = self.m22.into_parts();

        let value = TransferMatrix2::new(m11, m12, m21, m22);

        let first = TransferMatrix2::new(dm11, dm12, dm21, dm22);

        let second = TransferMatrix2::new(ddm11, ddm12, ddm21, ddm22);

        (value, first, second)
    }
}

impl<C, D, M> RawMatrixBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    type Matrix = TransferMatrix2<C, D>;
    type Error = Transfer2Error;

    fn solve_matrix(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let input = input.clone().to_complex();
        let matrix = self.evaluate_with::<RealAxis, _, _, _>(stack, &input)?;

        Ok(MatrixEvaluation::new(matrix))
    }
}

impl<C, D, M> RawMatrixStructuralDerivativeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn solve_matrix_structural_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let planar = input.clone().to_complex::<C>();
        let entries =
            self.evaluate_structural_first_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let (matrix, first) = entries.into_matrix_parts();

        Ok(MatrixEvaluation::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable.into(), first),
        ))
    }

    fn solve_matrix_structural_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let planar = input.clone().to_complex::<C>();
        let entries =
            self.evaluate_structural_second_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let (matrix, first, second) = entries.into_matrix_parts();

        Ok(MatrixEvaluation::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable.into(), first).with_second(second),
        ))
    }
}

impl<C, D, M> RawMatrixSpectralDerivativeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn solve_matrix_spectral_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let planar = input.clone().to_complex::<C>();
        let entries =
            self.evaluate_spectral_first_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let (matrix, first) = entries.into_matrix_parts();

        Ok(MatrixEvaluation::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable.into(), first),
        ))
    }

    fn solve_matrix_spectral_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let planar = input.clone().to_complex::<C>();
        let entries =
            self.evaluate_spectral_second_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let (matrix, first, second) = entries.into_matrix_parts();

        Ok(MatrixEvaluation::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable.into(), first).with_second(second),
        ))
    }
}

impl<C, D, M> ComplexMatrixBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    type Matrix = TransferMatrix2<C, D>;
    type Error = Transfer2Error;

    fn solve_analytic_matrix(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate_with::<ComplexPlane, _, _, _>(stack, input)
            .map(MatrixEvaluation::new)
    }
}

impl<C, D, M> ComplexMatrixStructuralDerivativeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn solve_complex_matrix_structural_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let entries =
            self.evaluate_structural_first_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let (matrix, first) = entries.into_matrix_parts();

        Ok(MatrixEvaluation::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable.into(), first),
        ))
    }

    fn solve_complex_matrix_structural_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let entries =
            self.evaluate_structural_second_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let (matrix, first, second) = entries.into_matrix_parts();

        Ok(MatrixEvaluation::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable.into(), first).with_second(second),
        ))
    }
}

impl<C, D, M> ComplexMatrixSpectralDerivativeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn solve_complex_matrix_spectral_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let entries =
            self.evaluate_spectral_first_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let (matrix, first) = entries.into_matrix_parts();

        Ok(MatrixEvaluation::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable.into(), first),
        ))
    }

    fn solve_complex_matrix_spectral_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let entries =
            self.evaluate_spectral_second_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let (matrix, first, second) = entries.into_matrix_parts();

        Ok(MatrixEvaluation::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable.into(), first).with_second(second),
        ))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr0, array};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    #[test]
    fn first_order_conversion_preserves_values_and_derivatives() {
        let entries = Matrix2Entries {
            m11: ArrayJetFirst::from_parts(arr0(c(1.0)), arr0(c(11.0))),
            m12: ArrayJetFirst::from_parts(arr0(c(2.0)), arr0(c(12.0))),
            m21: ArrayJetFirst::from_parts(arr0(c(3.0)), arr0(c(13.0))),
            m22: ArrayJetFirst::from_parts(arr0(c(4.0)), arr0(c(14.0))),
        };

        let (value, first) = entries.into_matrix_parts();

        assert_eq!(value.m11()[()], c(1.0));
        assert_eq!(value.m12()[()], c(2.0));
        assert_eq!(value.m21()[()], c(3.0));
        assert_eq!(value.m22()[()], c(4.0));

        assert_eq!(first.m11()[()], c(11.0));
        assert_eq!(first.m12()[()], c(12.0));
        assert_eq!(first.m21()[()], c(13.0));
        assert_eq!(first.m22()[()], c(14.0));
    }

    #[test]
    fn second_order_conversion_preserves_all_orders() {
        let entries = Matrix2Entries {
            m11: ArrayJet::from_parts(arr0(c(1.0)), arr0(c(11.0)), arr0(c(21.0))),
            m12: ArrayJet::from_parts(arr0(c(2.0)), arr0(c(12.0)), arr0(c(22.0))),
            m21: ArrayJet::from_parts(arr0(c(3.0)), arr0(c(13.0)), arr0(c(23.0))),
            m22: ArrayJet::from_parts(arr0(c(4.0)), arr0(c(14.0)), arr0(c(24.0))),
        };

        let (value, first, second) = entries.into_matrix_parts();

        assert_eq!(value.m11()[()], c(1.0));
        assert_eq!(value.m12()[()], c(2.0));
        assert_eq!(value.m21()[()], c(3.0));
        assert_eq!(value.m22()[()], c(4.0));

        assert_eq!(first.m11()[()], c(11.0));
        assert_eq!(first.m12()[()], c(12.0));
        assert_eq!(first.m21()[()], c(13.0));
        assert_eq!(first.m22()[()], c(14.0));

        assert_eq!(second.m11()[()], c(21.0));
        assert_eq!(second.m12()[()], c(22.0));
        assert_eq!(second.m21()[()], c(23.0));
        assert_eq!(second.m22()[()], c(24.0));
    }

    #[test]
    fn first_order_conversion_preserves_sample_shapes() {
        let entries = Matrix2Entries {
            m11: ArrayJetFirst::from_parts(array![c(1.0), c(2.0)], array![c(11.0), c(12.0)]),
            m12: ArrayJetFirst::from_parts(array![c(3.0), c(4.0)], array![c(13.0), c(14.0)]),
            m21: ArrayJetFirst::from_parts(array![c(5.0), c(6.0)], array![c(15.0), c(16.0)]),
            m22: ArrayJetFirst::from_parts(array![c(7.0), c(8.0)], array![c(17.0), c(18.0)]),
        };

        let (value, first) = entries.into_matrix_parts();

        let expected = value.m11().raw_dim();

        assert_eq!(value.m12().raw_dim(), expected);
        assert_eq!(value.m21().raw_dim(), expected);
        assert_eq!(value.m22().raw_dim(), expected);

        assert_eq!(first.m11().raw_dim(), expected);
        assert_eq!(first.m12().raw_dim(), expected);
        assert_eq!(first.m21().raw_dim(), expected);
        assert_eq!(first.m22().raw_dim(), expected);
    }

    #[test]
    fn second_order_conversion_preserves_sample_shapes() {
        let entries = Matrix2Entries {
            m11: ArrayJet::from_parts(
                array![c(1.0), c(2.0)],
                array![c(11.0), c(12.0)],
                array![c(21.0), c(22.0)],
            ),
            m12: ArrayJet::from_parts(
                array![c(3.0), c(4.0)],
                array![c(13.0), c(14.0)],
                array![c(23.0), c(24.0)],
            ),
            m21: ArrayJet::from_parts(
                array![c(5.0), c(6.0)],
                array![c(15.0), c(16.0)],
                array![c(25.0), c(26.0)],
            ),
            m22: ArrayJet::from_parts(
                array![c(7.0), c(8.0)],
                array![c(17.0), c(18.0)],
                array![c(27.0), c(28.0)],
            ),
        };

        let (value, first, second) = entries.into_matrix_parts();

        let expected = value.m11().raw_dim();

        for matrix in [&value, &first, &second] {
            assert_eq!(matrix.m11().raw_dim(), expected);
            assert_eq!(matrix.m12().raw_dim(), expected);
            assert_eq!(matrix.m21().raw_dim(), expected);
            assert_eq!(matrix.m22().raw_dim(), expected);
        }
    }

    #[test]
    fn first_order_conversion_does_not_swap_transmission_channels() {
        let entries = Matrix2Entries {
            m11: ArrayJetFirst::from_parts(arr0(c(0.0)), arr0(c(0.0))),
            m12: ArrayJetFirst::from_parts(arr0(c(12.0)), arr0(c(112.0))),
            m21: ArrayJetFirst::from_parts(arr0(c(21.0)), arr0(c(121.0))),
            m22: ArrayJetFirst::from_parts(arr0(c(0.0)), arr0(c(0.0))),
        };

        let (value, first) = entries.into_matrix_parts();

        assert_eq!(value.m12()[()], c(12.0));
        assert_eq!(value.m21()[()], c(21.0));

        assert_eq!(first.m12()[()], c(112.0));
        assert_eq!(first.m21()[()], c(121.0));
    }
}

#[cfg(test)]
mod raw_matrix_backend_tests {
    use crate::{
        Polarisation, Thickness, ValidationConfig,
        backend::derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        material::Constant,
    };

    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn planar(
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
    fn planar_re(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<Array0<f64>> {
        PlanarInput::new(
            arr0(vacuum_wavenumber),
            arr0(parallel_wavenumber),
            polarisation,
        )
    }

    fn one_layer_stack(thickness_cm: f64) -> Stack<Constant<f64>, f64> {
        Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.44, 1.0))
            .layer(
                Constant::new(2.25, 1.0),
                Thickness::from_cm(thickness_cm).unwrap(),
            )
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap()
    }

    #[test]
    fn value_solve_contains_no_derivatives() {
        let stack = one_layer_stack(0.2);
        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        let result = Transfer2::new()
            .solve_analytic_matrix(&stack, &input)
            .unwrap();

        assert!(result.derivatives().is_none());
    }

    #[test]
    fn first_derivative_solve_packages_requested_variable() {
        let stack = one_layer_stack(0.2);
        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        let variable = StructuralDerivativeVariable::Thickness(0);

        let result = Transfer2::new()
            .solve_complex_matrix_structural_first_derivative(&stack, &input, variable)
            .unwrap();

        let derivatives = result.derivatives().unwrap();

        assert_eq!(derivatives.variable(), variable.into(),);

        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_derivative_solve_packages_both_orders() {
        let stack = one_layer_stack(0.2);
        let input = planar(3.0, 0.4, Polarisation::TransverseMagnetic);

        let variable = SpectralDerivativeVariable::VacuumWavenumber;

        let result = Transfer2::new()
            .solve_complex_matrix_spectral_second_derivative(&stack, &input, variable)
            .unwrap();

        let derivatives = result.derivatives().unwrap();

        assert_eq!(derivatives.variable(), variable.into(),);

        assert!(derivatives.second().is_some());
    }

    #[test]
    fn raw_first_derivative_matches_internal_evaluation() {
        let stack = one_layer_stack(0.2);
        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        let variable = StructuralDerivativeVariable::ParallelWavenumberSquared;

        let internal = Transfer2::new()
            .evaluate_structural_first_with::<RealAxis, _, _, _>(&stack, &input, variable)
            .unwrap();

        let (expected_value, expected_first) = internal.into_matrix_parts();

        let public = Transfer2::new()
            .solve_complex_matrix_structural_first_derivative(&stack, &input, variable)
            .unwrap();

        assert_eq!(public.matrix(), &expected_value,);

        assert_eq!(public.derivatives().unwrap().first(), &expected_first,);
    }
}
