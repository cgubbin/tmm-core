use lamina_core::{
    CoordinateInput, Coordinates, DifferentiableMaterialStack, FiniteLayerIndex, InPlaneCoordinate,
    IncidentSide, Parameter, Polarisation, RealAxisEvaluator, Scatter2, SpectralCoordinate, Stack,
    material::Constant,
    units::{InverseLengthUnit, Length},
};
use ndarray::arr0;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vacuum = Constant::new(1.0, 1.0);
    let film = Constant::new(2.25, 1.0);

    let stack: DifferentiableMaterialStack<Complex64> =
        Stack::from_differentiable_materials(vacuum, vacuum)
            .differentiable_layer(film, Length::nanometres(500.0))
            .finalise();

    let coordinate_system = Coordinates::new(
        SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
        InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerCentimetre),
    );

    let coordinates = CoordinateInput::intrinsic(coordinate_system, arr0(10_000.0), arr0(2_000.0))?;

    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let incident_side = IncidentSide::Left;
    let polarisation = Polarisation::TransverseElectric;

    // Differentiate with respect to the caller-facing spectral coordinate.
    let spectral = evaluator.evaluate_first(
        coordinates.clone(),
        &stack,
        polarisation,
        Parameter::Spectral,
    )?;

    let spectral_power = spectral.power(incident_side)?;

    println!("Spectral derivative");
    println!("-------------------");
    println!("R       = {:.8}", spectral_power.value().reflectance()[()]);
    println!("dR/dk0  = {:.8e}", spectral_power.first().reflectance()[()]);
    println!();

    // Differentiate with respect to the first finite-layer thickness.
    let thickness = evaluator.evaluate_first(
        coordinates,
        &stack,
        polarisation,
        Parameter::LayerThickness(FiniteLayerIndex::new(0)),
    )?;

    let thickness_power = thickness.power(incident_side)?;

    println!("Thickness derivative");
    println!("--------------------");
    println!("R       = {:.8}", thickness_power.value().reflectance()[()]);
    println!(
        "dR/dd   = {:.8e}",
        thickness_power.first().reflectance()[()]
    );

    Ok(())
}
