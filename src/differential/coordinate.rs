#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DirectionalCoordinate {
    VacuumWavenumber,
    ParallelWavenumber,
    Thickness(FiniteLayerIndex),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FiniteLayerIndex(usize);
