use lamina_core::{
    Constant, CoordinateInput, Coordinates, InPlaneCoordinate, IncidentSide, MaterialStack,
    Polarisation, RealAxisEvaluator, Scatter2, SpectralCoordinate, Stack,
    units::{AngleUnit, InverseLengthUnit, LengthUnit},
};
use lamina_units::Length;
use ndarray::arr0;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let incident_side = IncidentSide::Left;
    // Intrinsic canonical-like coordinates.
    let system = Coordinates::new(
        SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
        InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerCentimetre),
    );

    let intrinsic = CoordinateInput::intrinsic(system, arr0(10_000.0), arr0(2_000.0))?;

    // Another intrinsic parameterisation.
    let system = Coordinates::new(
        SpectralCoordinate::VacuumWavelength(LengthUnit::Nanometre),
        InPlaneCoordinate::EffectiveIndex,
    );

    let effective_index = CoordinateInput::intrinsic(system, arr0(1_000.0), arr0(0.2))?;

    // Incidence angle is extrinsic because converting it to k_parallel
    // depends on the incident medium.
    let system = Coordinates::new(
        SpectralCoordinate::VacuumWavelength(LengthUnit::Nanometre),
        InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
    );

    let angle = CoordinateInput::incident_referenced(
        system,
        arr0(1_000.0),
        arr0(11.536959032815489),
        incident_side,
    )?;

    let vacuum = Constant::new(1.0, 1.0);
    let film = Constant::new(2.25, 1.0);

    let stack: MaterialStack<Complex64> = Stack::from_materials(vacuum, vacuum)
        .material_layer(film, Length::nanometres(500.0f64))
        .finalise();

    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let intrinsic_response =
        evaluator.evaluate(intrinsic, &stack, Polarisation::TransverseElectric)?;
    let effective_index_response =
        evaluator.evaluate(effective_index, &stack, Polarisation::TransverseElectric)?;
    let angle_response = evaluator.evaluate(angle, &stack, Polarisation::TransverseElectric)?;

    let intrinsic_power = intrinsic_response.power(incident_side)?;
    let effective_index_power = effective_index_response.power(incident_side)?;
    let angle_power = angle_response.power(incident_side)?;

    println!("Equivalent coordinate parameterisations");
    println!("-------------------------------------");
    println!("Vacuum wavelength:       1000 nm");
    println!("Effective index:         0.2");
    println!("Incidence angle:         11.537 deg");
    println!();

    println!("{:<28} {:>12} {:>12} {:>12}", "Coordinates", "R", "T", "A");

    println!(
        "{:<28} {:>12.8} {:>12.8} {:>12.3e}",
        "k0 + k_parallel",
        intrinsic_power.reflectance()[()],
        intrinsic_power.transmittance()[()],
        intrinsic_power.absorptance()[()],
    );

    println!(
        "{:<28} {:>12.8} {:>12.8} {:>12.3e}",
        "wavelength + n_eff",
        effective_index_power.reflectance()[()],
        effective_index_power.transmittance()[()],
        effective_index_power.absorptance()[()],
    );

    println!(
        "{:<28} {:>12.8} {:>12.8} {:>12.3e}",
        "wavelength + angle",
        angle_power.reflectance()[()],
        angle_power.transmittance()[()],
        angle_power.absorptance()[()],
    );

    let tolerance = 1.0e-12;

    for power in [&effective_index_power, &angle_power] {
        assert!((power.reflectance()[()] - intrinsic_power.reflectance()[()]).abs() < tolerance);

        assert!(
            (power.transmittance()[()] - intrinsic_power.transmittance()[()]).abs() < tolerance
        );
    }

    Ok(())
}
