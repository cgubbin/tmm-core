use crate::backend::evaluate_exterior_wavevectors;
use crate::input::canonical::CanonicalLayer;
use crate::test_support::assertions::{FIRST_TOLERANCE, SECOND_TOLERANCE, VALUE_TOLERANCE};
use crate::test_support::jet::{HoloJ0, HoloJ1, HoloJ2, HoloJB2};
use crate::test_support::planar::{principal_exterior_wavevectors, two_layer_stack};
use crate::{
    CanonicalCoordinates, CanonicalStack, ComplexPlane, ComplexPlaneEvaluator, ExteriorWavevectors,
    SeedJet, Transfer2,
};
use crate::{
    FiniteLayerIndex, IncidentSide, Parameter, Polarisation, RealAxisEvaluator, backend::Scatter2,
};

use crate::test_support::{
    C,
    assertions::{assert_complex_close, assert_real_close},
    planar::{FILM_THICKNESS_CM, scalar_real_input, single_layer_stack},
};

use ndarray::arr0;
use num_complex::Complex64;

const TOLERANCE: f64 = 1.0e-12;

fn evaluator() -> RealAxisEvaluator<Scatter2> {
    RealAxisEvaluator::new(Scatter2)
}

#[test]
fn solve_and_retain_value_paths_have_identical_amplitudes() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let solved = evaluator
        .evaluate(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let retained = evaluator
        .retain(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    for side in [IncidentSide::Left, IncidentSide::Right] {
        let solved = solved.amplitudes(side).unwrap();

        let retained = retained.excitation(side).unwrap().amplitudes();

        assert_complex_close(
            solved.value().reflection()[()],
            retained.value().reflection()[()],
            TOLERANCE,
        );

        assert_complex_close(
            solved.value().transmission()[()],
            retained.value().transmission()[()],
            TOLERANCE,
        );
    }
}

#[test]
fn solve_and_retain_value_paths_have_identical_power() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let solved = evaluator
        .evaluate(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap()
        .power(IncidentSide::Left)
        .unwrap();

    let retained = evaluator
        .retain(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap()
        .excitation(IncidentSide::Left)
        .unwrap()
        .power();

    assert_real_close(
        solved.value().reflectance()[()],
        retained.value().reflectance()[()],
        TOLERANCE,
    );

    assert_real_close(
        solved.value().transmittance()[()],
        retained.value().transmittance()[()],
        TOLERANCE,
    );

    assert_real_close(
        solved.value().absorptance()[()],
        retained.value().absorptance()[()],
        TOLERANCE,
    );
}

#[test]
fn solve_and_retain_first_derivatives_are_identical() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(0));

    let solved = evaluator
        .evaluate_first(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            parameter,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left)
        .unwrap();

    let retained = evaluator
        .retain_first(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            parameter,
        )
        .unwrap()
        .excitation(IncidentSide::Left)
        .unwrap()
        .amplitudes();

    assert_eq!(solved.parameter(), parameter);
    assert_eq!(retained.parameter(), parameter);

    assert_complex_close(
        solved.value().reflection()[()],
        retained.value().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.first().reflection()[()],
        retained.first().reflection()[()],
        TOLERANCE,
    );
}

#[test]
fn solve_and_retain_second_derivatives_are_identical() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let solved = evaluator
        .evaluate_second(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left)
        .unwrap();

    let retained = evaluator
        .retain_second(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .excitation(IncidentSide::Left)
        .unwrap()
        .amplitudes();

    assert_complex_close(
        solved.value().reflection()[()],
        retained.value().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.first().reflection()[()],
        retained.first().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.second().reflection()[()],
        retained.second().reflection()[()],
        TOLERANCE,
    );
}

#[test]
fn solve_and_retain_bivariate_results_are_identical() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let thickness = Parameter::LayerThickness(FiniteLayerIndex::new(0));

    let solved = evaluator
        .evaluate_bivariate_second(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
            thickness,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left)
        .unwrap();

    let retained = evaluator
        .retain_bivariate_second(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
            thickness,
        )
        .unwrap()
        .excitation(IncidentSide::Left)
        .unwrap()
        .amplitudes();

    assert_eq!(solved.parameters(), retained.parameters(),);

    assert_complex_close(
        solved.gradient().axis0().reflection()[()],
        retained.gradient().axis0().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.gradient().axis1().reflection()[()],
        retained.gradient().axis1().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.hessian().axis0_axis1().reflection()[()],
        retained.hessian().axis0_axis1().reflection()[()],
        TOLERANCE,
    );
}

#[test]
fn solve_and_retain_modal_determinants_are_identical() {
    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = || {
        CanonicalCoordinates::new(
            HoloJ2::variable(arr0(Complex64::new(2.0, 0.2))),
            HoloJ2::constant(arr0(Complex64::new(0.3, 0.1))),
        )
    };

    let solved_coordinates = coordinates();

    let solved_exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ2>(
        &solved_coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let solved = evaluator
        .determinant(&solved_coordinates, &solved_exterior, polarisation)
        .unwrap()
        .into_inner();

    let retained_coordinates = coordinates();

    let retained_exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ2>(
        &retained_coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let retained = evaluator
        .retain(retained_coordinates, retained_exterior, polarisation)
        .unwrap()
        .determinant()
        .into_inner();

    assert_complex_close(solved.value()[()], retained.value()[()], TOLERANCE);

    assert_complex_close(solved.first()[()], retained.first()[()], TOLERANCE);

    assert_complex_close(solved.second()[()], retained.second()[()], TOLERANCE);
}

const COMPLEX_K0: C = C::new(2.2, 0.15);
const COMPLEX_K_PARALLEL: C = C::new(0.3, -0.08);

#[test]
fn complex_plane_bivariate_determinant_derivatives_match_between_backends() {
    let stack = single_layer_stack(1.8, 0.17);

    let transfer =
        ComplexPlaneEvaluator::<HoloJB2, _, _>::compile(&stack, Transfer2::new()).unwrap();

    let scatter = ComplexPlaneEvaluator::<HoloJB2, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = || {
        CanonicalCoordinates::new(
            <HoloJB2 as SeedJet>::variable(arr0(COMPLEX_K0), 0).unwrap(),
            <HoloJB2 as SeedJet>::variable(arr0(COMPLEX_K_PARALLEL), 1).unwrap(),
        )
    };

    let transfer_coordinates = coordinates();

    let transfer_exterior = principal_exterior_wavevectors(transfer.stack(), &transfer_coordinates);

    let transfer_determinant = transfer
        .determinant(&transfer_coordinates, &transfer_exterior, polarisation)
        .unwrap()
        .into_inner();

    let scatter_coordinates = coordinates();

    let scatter_exterior = principal_exterior_wavevectors(scatter.stack(), &scatter_coordinates);

    let scatter_determinant = scatter
        .determinant(&scatter_coordinates, &scatter_exterior, polarisation)
        .unwrap()
        .into_inner();

    assert_complex_close(
        transfer_determinant.value()[()],
        scatter_determinant.value()[()],
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        transfer_determinant.axis0()[()],
        scatter_determinant.axis0()[()],
        FIRST_TOLERANCE,
    );

    assert_complex_close(
        transfer_determinant.axis1()[()],
        scatter_determinant.axis1()[()],
        FIRST_TOLERANCE,
    );

    assert_complex_close(
        transfer_determinant.axis0_axis0()[()],
        scatter_determinant.axis0_axis0()[()],
        SECOND_TOLERANCE,
    );

    assert_complex_close(
        transfer_determinant.axis0_axis1()[()],
        scatter_determinant.axis0_axis1()[()],
        SECOND_TOLERANCE,
    );

    assert_complex_close(
        transfer_determinant.axis1_axis1()[()],
        scatter_determinant.axis1_axis1()[()],
        SECOND_TOLERANCE,
    );
}

#[test]
fn complex_plane_determinant_respects_explicit_exterior_branch() {
    let stack = two_layer_stack();

    let transfer =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Transfer2::new()).unwrap();

    let scatter = ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = || {
        CanonicalCoordinates::new(
            <HoloJ0 as SeedJet>::constant(arr0(C::new(2.5, -0.05))),
            <HoloJ0 as SeedJet>::constant(arr0(C::new(0.31, 0.02))),
        )
    };

    /*
     * Principal-branch reference.
     */
    let principal_coordinates = coordinates();

    let principal_exterior =
        principal_exterior_wavevectors(transfer.stack(), &principal_coordinates);

    let principal_determinant = transfer
        .determinant(&principal_coordinates, &principal_exterior, polarisation)
        .unwrap();

    /*
     * Deliberately reverse the left exterior branch while
     * retaining the right exterior branch.
     */
    let transfer_coordinates = coordinates();

    let transfer_principal =
        principal_exterior_wavevectors(transfer.stack(), &transfer_coordinates);

    let transfer_exterior = ExteriorWavevectors::new(
        transfer_principal.left().negate(),
        transfer_principal.right().clone(),
    );

    let transfer_determinant = transfer
        .determinant(&transfer_coordinates, &transfer_exterior, polarisation)
        .unwrap();

    let scatter_coordinates = coordinates();

    let scatter_principal = principal_exterior_wavevectors(scatter.stack(), &scatter_coordinates);

    let scatter_exterior = ExteriorWavevectors::new(
        scatter_principal.left().negate(),
        scatter_principal.right().clone(),
    );

    let scatter_determinant = scatter
        .determinant(&scatter_coordinates, &scatter_exterior, polarisation)
        .unwrap();

    /*
     * Both numerical formulations must honour the supplied
     * branch in the same way.
     */
    assert_complex_close(
        transfer_determinant.value()[()],
        scatter_determinant.value()[()],
        VALUE_TOLERANCE,
    );

    /*
     * And the explicit branch must actually affect the
     * characteristic function.
     */
    assert!(
        (transfer_determinant.value()[()] - principal_determinant.value()[()]).norm() > 1.0e-8,
        "changing the supplied exterior branch did not change the determinant",
    );
}

#[test]
fn complex_plane_canonical_thickness_derivative_matches_finite_difference() {
    let thickness_cm = 0.17;

    let canonical_stack = CanonicalStack::new(
        crate::Constant::vacuum(),
        crate::Constant::vacuum(),
        vec![CanonicalLayer::new(
            crate::Constant::dielectric(1.8 * 1.8),
            <HoloJ1 as SeedJet>::variable(arr0(C::new(thickness_cm, 0.0)), 0).unwrap(),
        )],
    );

    let evaluator = ComplexPlaneEvaluator::from_canonical_stack(canonical_stack, Scatter2::new());

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = CanonicalCoordinates::new(
        <HoloJ1 as SeedJet>::constant(arr0(COMPLEX_K0)),
        <HoloJ1 as SeedJet>::constant(arr0(COMPLEX_K_PARALLEL)),
    );

    let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

    let differentiated = evaluator
        .determinant(&coordinates, &exterior, polarisation)
        .unwrap()
        .into_inner();

    let step = 1.0e-6;

    let evaluate = |thickness_cm: f64| {
        let stack = single_layer_stack(1.8, thickness_cm);

        let evaluator =
            ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

        let coordinates = CanonicalCoordinates::new(
            <HoloJ0 as SeedJet>::constant(arr0(COMPLEX_K0)),
            <HoloJ0 as SeedJet>::constant(arr0(COMPLEX_K_PARALLEL)),
        );

        let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

        evaluator
            .determinant(&coordinates, &exterior, polarisation)
            .unwrap()
            .value()[()]
    };

    let below = evaluate(thickness_cm - step);

    let above = evaluate(thickness_cm + step);

    let finite_difference = (above - below) / (2.0 * step);

    assert_complex_close(differentiated.first()[()], finite_difference, 1.0e-8);
}
