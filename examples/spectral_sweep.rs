use lamina_core::{
    CoordinateInput, Coordinates, InPlaneCoordinate, IncidentSide, MaterialStack, Polarisation,
    RealAxisEvaluator, Scatter2, SpectralCoordinate, Stack,
    material::Constant,
    units::{InverseLengthUnit, Length, LengthUnit},
};
use ndarray::Array1;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vacuum = Constant::new(1.0, 1.0);
    let film = Constant::new(2.25, 1.0);

    let stack: MaterialStack<Complex64> = Stack::from_materials(vacuum, vacuum)
        .material_layer(film, Length::nanometres(500.0))
        .finalise();

    let coordinate_system = Coordinates::new(
        SpectralCoordinate::VacuumWavelength(LengthUnit::Nanometre),
        InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerCentimetre),
    );

    /*
     * Sweep vacuum wavelength from 400 to 1600 nm at normal incidence.
     *
     * Both arrays have the same sampled shape. The calculation is evaluated
     * over the complete array by lamina-core.
     */
    let wavelength = Array1::linspace(400.0, 1600.0, 121);
    let parallel_wavenumber = Array1::zeros(wavelength.len());

    let coordinates =
        CoordinateInput::intrinsic(coordinate_system, wavelength.clone(), parallel_wavenumber)?;

    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let response = evaluator.evaluate(coordinates, &stack, Polarisation::TransverseElectric)?;

    let power = response.power(IncidentSide::Left)?;

    println!("{:>12} {:>12} {:>12} {:>12}", "λ / nm", "R", "T", "A",);

    for index in (0..wavelength.len()).step_by(10) {
        println!(
            "{:12.2} {:12.8} {:12.8} {:12.3e}",
            wavelength[index],
            power.reflectance()[index],
            power.transmittance()[index],
            power.absorptance()[index],
        );
    }

    Ok(())
}
