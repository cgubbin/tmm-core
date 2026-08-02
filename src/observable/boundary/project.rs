use crate::{
    ComplexScalar, IncidentSide,
    algebra::ScalarAlgebra,
    backend::{ReconstructLayerBoundaryWaves, RetainedIsotropicLayers},
    observable::{LayerBoundaries, LayerBoundaryStates, LayerBoundaryWaves},
};

use ndarray::Dimension;
use thiserror::Error;

/// Failure to construct boundary observables from a retained backend result.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BoundaryProjectionError {
    /// The backend result does not contain retained internal-layer data.
    #[error("the backend result does not retain internal layer data")]
    LayersNotRetained,

    /// Boundary waves and retained layer quantities describe different numbers
    /// of finite layers.
    #[error(
        "boundary-wave count {wave_count} does not match retained layer count \
         {layer_count}"
    )]
    LayerCountMismatch {
        wave_count: usize,
        layer_count: usize,
    },

    /// The retained workspace reported a layer count but did not provide the
    /// corresponding layer quantities.
    #[error(
        "retained layer {requested} is unavailable; retained layer count is \
         {layer_count}"
    )]
    LayerOutOfBounds {
        requested: usize,
        layer_count: usize,
    },

    #[error(
        "finite-layer boundary data are inconsistent: \
     {wave_count} wave records, and \
     {admittance_count} admittances"
    )]
    LayerDataCountMismatch {
        wave_count: usize,
        admittance_count: usize,
    },
}

/// Convert retained backend boundary waves into observable-layer containers.
pub(crate) fn project_boundary_waves<W>(
    workspace: &W,
    incident_side: IncidentSide,
) -> Result<LayerBoundaries<LayerBoundaryWaves<W::Algebra>>, BoundaryProjectionError>
where
    W: ReconstructLayerBoundaryWaves,
{
    let waves = workspace
        .reconstruct_layer_boundary_waves(incident_side)
        .ok_or(BoundaryProjectionError::LayersNotRetained)?;

    Ok(LayerBoundaries::new(
        waves.into_iter().map(Into::into).collect(),
    ))
}

/// Convert retained backend boundary waves into canonical isotropic states.
pub(crate) fn project_boundary_states<A, W>(
    workspace: &W,
    incident_side: IncidentSide,
) -> Result<LayerBoundaries<LayerBoundaryStates<A>>, BoundaryProjectionError>
where
    W: ReconstructLayerBoundaryWaves<Algebra = A> + RetainedIsotropicLayers<Algebra = A>,
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let waves = workspace
        .reconstruct_layer_boundary_waves(incident_side)
        .ok_or(BoundaryProjectionError::LayersNotRetained)?;

    let layer_count = workspace
        .retained_layer_count()
        .ok_or(BoundaryProjectionError::LayersNotRetained)?;

    if waves.len() != layer_count {
        return Err(BoundaryProjectionError::LayerCountMismatch {
            wave_count: waves.len(),
            layer_count,
        });
    }

    let states = waves
        .into_iter()
        .enumerate()
        .map(|(index, waves)| {
            let quantities = workspace.layer_quantities(index).ok_or(
                BoundaryProjectionError::LayerOutOfBounds {
                    requested: index,
                    layer_count,
                },
            )?;

            let admittance = quantities.clone().into_admittance().into_inner();

            Ok(LayerBoundaryWaves::from(waves).into_states(&admittance))
        })
        .collect::<Result<Vec<_>, BoundaryProjectionError>>()?;

    Ok(LayerBoundaries::new(states))
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation, RealAxis,
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::{
            BidirectionalWaves as BackendBoundaryWaves, IsotropicLayerQuantities,
            LayerBoundaryWaves as BackendLayerBoundaryWaves,
        },
        input::CanonicalCoordinates,
        material::Constant,
    };

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn c(real: f64) -> C {
        C::new(real, 0.0)
    }

    fn coordinates() -> CanonicalCoordinates<A> {
        CanonicalCoordinates::new(jet(c(2.3)), jet(c(0.37)))
    }

    fn quantities(epsilon: f64) -> IsotropicLayerQuantities<A> {
        IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            &Constant::new(epsilon, 1.0),
            &coordinates(),
            Polarisation::TransverseElectric,
        )
    }

    fn boundary_waves(offset: f64) -> BackendLayerBoundaryWaves<A> {
        BackendLayerBoundaryWaves::new(
            BackendBoundaryWaves::new(
                jet(C::new(offset + 1.0, 0.1)),
                jet(C::new(offset + 2.0, 0.2)),
            ),
            BackendBoundaryWaves::new(
                jet(C::new(offset + 3.0, 0.3)),
                jet(C::new(offset + 4.0, 0.4)),
            ),
        )
    }

    #[derive(Clone, Debug)]
    struct MockRetainedWorkspace {
        waves: Option<Vec<BackendLayerBoundaryWaves<A>>>,
        retained_layer_count: Option<usize>,
        quantities: Vec<IsotropicLayerQuantities<A>>,
    }

    impl ReconstructLayerBoundaryWaves for MockRetainedWorkspace {
        type Algebra = A;
        fn reconstruct_layer_boundary_waves(
            &self,
            _incident_side: IncidentSide,
        ) -> Option<Vec<BackendLayerBoundaryWaves<A>>> {
            self.waves.clone()
        }
    }

    impl RetainedIsotropicLayers for MockRetainedWorkspace {
        type Algebra = A;

        fn retained_layer_count(&self) -> Option<usize> {
            self.retained_layer_count
        }

        fn layer_quantities(&self, index: usize) -> Option<&IsotropicLayerQuantities<A>> {
            self.quantities.get(index)
        }
    }

    #[test]
    fn boundary_waves_reject_missing_retention() {
        let workspace = MockRetainedWorkspace {
            waves: None,
            retained_layer_count: None,
            quantities: Vec::new(),
        };

        let error = project_boundary_waves(&workspace, IncidentSide::Left)
            .expect_err("missing retained waves should be rejected");

        assert_eq!(error, BoundaryProjectionError::LayersNotRetained,);
    }

    #[test]
    fn boundary_states_reject_missing_boundary_wave_retention() {
        let workspace = MockRetainedWorkspace {
            waves: None,
            retained_layer_count: Some(0),
            quantities: Vec::new(),
        };

        let error = project_boundary_states(&workspace, IncidentSide::Left)
            .expect_err("missing retained waves should be rejected");

        assert_eq!(error, BoundaryProjectionError::LayersNotRetained,);
    }

    #[test]
    fn boundary_states_reject_missing_layer_quantity_retention() {
        let workspace = MockRetainedWorkspace {
            waves: Some(Vec::new()),
            retained_layer_count: None,
            quantities: Vec::new(),
        };

        let error = project_boundary_states(&workspace, IncidentSide::Left)
            .expect_err("missing retained layer quantities should be rejected");

        assert_eq!(error, BoundaryProjectionError::LayersNotRetained,);
    }

    #[test]
    fn boundary_states_reject_more_waves_than_retained_layers() {
        let workspace = MockRetainedWorkspace {
            waves: Some(vec![boundary_waves(0.0), boundary_waves(10.0)]),
            retained_layer_count: Some(1),
            quantities: vec![quantities(2.25)],
        };

        let error = project_boundary_states(&workspace, IncidentSide::Left)
            .expect_err("inconsistent retained layer counts should be rejected");

        assert_eq!(
            error,
            BoundaryProjectionError::LayerCountMismatch {
                wave_count: 2,
                layer_count: 1,
            },
        );
    }

    #[test]
    fn boundary_states_reject_fewer_waves_than_retained_layers() {
        let workspace = MockRetainedWorkspace {
            waves: Some(vec![boundary_waves(0.0)]),
            retained_layer_count: Some(2),
            quantities: vec![quantities(2.25), quantities(4.0)],
        };

        let error = project_boundary_states(&workspace, IncidentSide::Right)
            .expect_err("inconsistent retained layer counts should be rejected");

        assert_eq!(
            error,
            BoundaryProjectionError::LayerCountMismatch {
                wave_count: 1,
                layer_count: 2,
            },
        );
    }

    #[test]
    fn boundary_states_reject_missing_quantities_within_reported_range() {
        let workspace = MockRetainedWorkspace {
            waves: Some(vec![boundary_waves(0.0), boundary_waves(10.0)]),
            retained_layer_count: Some(2),

            // The workspace claims two retained layers but only exposes
            // quantities for the first.
            quantities: vec![quantities(2.25)],
        };

        let error = project_boundary_states(&workspace, IncidentSide::Left)
            .expect_err("missing quantities inside the retained range should be rejected");

        assert_eq!(
            error,
            BoundaryProjectionError::LayerOutOfBounds {
                requested: 1,
                layer_count: 2,
            },
        );
    }

    #[test]
    fn empty_retained_stack_is_valid() {
        let workspace = MockRetainedWorkspace {
            waves: Some(Vec::new()),
            retained_layer_count: Some(0),
            quantities: Vec::new(),
        };

        let waves = project_boundary_waves(&workspace, IncidentSide::Left)
            .expect("an empty retained stack should produce empty waves");

        let states = project_boundary_states(&workspace, IncidentSide::Left)
            .expect("an empty retained stack should produce empty states");

        assert!(waves.is_empty());
        assert!(states.is_empty());
    }

    #[test]
    fn consistent_retained_workspace_projects_all_layers() {
        let workspace = MockRetainedWorkspace {
            waves: Some(vec![boundary_waves(0.0), boundary_waves(10.0)]),
            retained_layer_count: Some(2),
            quantities: vec![quantities(2.25), quantities(4.0)],
        };

        let waves = project_boundary_waves(&workspace, IncidentSide::Right)
            .expect("consistent retained waves should project");

        let states = project_boundary_states(&workspace, IncidentSide::Right)
            .expect("consistent retained states should project");

        assert_eq!(waves.len(), 2);
        assert_eq!(states.len(), 2);
    }
}
