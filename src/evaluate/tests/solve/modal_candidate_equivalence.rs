// use crate::{
//     algebra::ScalarAlgebra,
//     backend::{
//         Backend, ModalSolutionSource, PlaneWaveSolutionSource, ReconstructLayerModeWaves, Scatter2,
//         scatter2::{Scatter2ProjectiveEntries, Scatter2Workspace},
//         transfer2::Transfer2Workspace,
//     },
//     test_support::{TOLERANCE, assertions::assert_complex_close, jet::J0},
// };

// fn scatter_workspace_fixture() -> Scatter2Workspace<J0> {
//     let problem = backend_comparison_problem();

//     Scatter2::default().solve(&problem).unwrap()
// }

// fn transfer_workspace_fixture() -> Transfer2Workspace<J0> {
//     let problem = backend_comparison_problem();

//     Transfer2::default().solve(&problem).unwrap()
// }

// #[test]
// fn modal_candidates_are_projectively_equivalent() {
//     let scatter = scatter_workspace_fixture();

//     let transfer = transfer_workspace_fixture();

//     let scatter_candidate = scatter.modal_boundary_solution().unwrap();

//     let transfer_candidate = transfer.modal_boundary_solution().unwrap();

//     let scale = scatter.solution().entries().n21().value()[()];

//     assert_complex_close(
//         scatter_candidate.state().field().value()[()],
//         scale * transfer_candidate.state().field().value()[()],
//         TOLERANCE,
//     );

//     assert_complex_close(
//         scatter_candidate.state().secondary().value()[()],
//         scale * transfer_candidate.state().secondary().value()[()],
//         TOLERANCE,
//     );

//     assert_complex_close(
//         scatter_candidate.projective_residual().value()[()],
//         scale * transfer_candidate.projective_residual().value()[()],
//         TOLERANCE,
//     );
// }

// #[test]
// fn reconstructed_modal_layer_waves_are_projectively_equivalent() {
//     let scatter = scatter_workspace_fixture();

//     let transfer = transfer_workspace_fixture();

//     let scatter_candidate = scatter.modal_boundary_solution().unwrap();

//     let transfer_candidate = transfer.modal_boundary_solution().unwrap();

//     let scatter_waves = scatter
//         .reconstruct_layer_mode_waves(&scatter_candidate)
//         .unwrap();

//     let transfer_waves = transfer
//         .reconstruct_layer_mode_waves(&transfer_candidate)
//         .unwrap();

//     let scale = scatter.solution().entries().n21().value()[()];

//     assert_eq!(scatter_waves.len(), transfer_waves.len(),);

//     for (scatter, transfer) in scatter_waves.iter().zip(&transfer_waves) {
//         for (scatter, transfer) in [
//             (scatter.left().forward(), transfer.left().forward()),
//             (scatter.left().backward(), transfer.left().backward()),
//             (scatter.right().forward(), transfer.right().forward()),
//             (scatter.right().backward(), transfer.right().backward()),
//         ] {
//             assert_complex_close(scatter.value()[()], scale * transfer.value()[()], TOLERANCE);
//         }
//     }
// }
