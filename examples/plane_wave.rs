use lamina_core::{
    CoordinateInput, Coordinates, InPlaneCoordinate, IncidentSide, MaterialStack, Polarisation,
    RealAxisEvaluator, Scatter2, SpectralCoordinate, Stack,
    material::Constant,
    units::{InverseLengthUnit, Length},
};
use ndarray::arr0;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vacuum = Constant::new(1.0, 1.0);
    let film = Constant::new(2.25, 1.0);

    let stack: MaterialStack<Complex64> = Stack::from_materials(vacuum, vacuum)
        .material_layer(film, Length::nanometres(500.0f64))
        .finalise();

    let coordinate_system = Coordinates::new(
        SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
        InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
    );

    let coordinates = CoordinateInput::intrinsic(coordinate_system, arr0(10_000.0), arr0(0.0))?;

    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let response = evaluator.evaluate(coordinates, &stack, Polarisation::TransverseElectric)?;

    let incident_side = IncidentSide::Left;
    let amplitudes = response.amplitudes(incident_side)?;
    let power = response.power(incident_side)?;
    println!("Left to right propagation:");
    println!("r = {:?}", amplitudes.reflection()[()]);
    println!("t = {:?}", amplitudes.transmission()[()]);
    println!("R = {:?}", power.reflectance()[()]);
    println!("T = {:?}", power.transmittance()[()]);
    println!("A = {:?}", power.absorptance()[()]);

    let incident_side = IncidentSide::Right;
    let amplitudes = response.amplitudes(incident_side)?;
    let power = response.power(incident_side)?;
    println!("Right to left propagation:");
    println!("r = {:?}", amplitudes.reflection()[()]);
    println!("t = {:?}", amplitudes.transmission()[()]);
    println!("R = {:?}", power.reflectance()[()]);
    println!("T = {:?}", power.transmittance()[()]);
    println!("A = {:?}", power.absorptance()[()]);

    Ok(())
}
