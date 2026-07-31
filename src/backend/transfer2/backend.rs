//! Internal evaluation kernel for the isotropic 2×2 transfer backend.
//!
//! This module evaluates the native transfer matrix of a finite planar stack.
//! It is responsible for:
//!
//! - evaluating isotropic quantities once per finite layer;
//! - constructing each finite-layer transfer matrix;
//! - composing layer matrices in propagation order;
//! - propagating first- and second-order derivatives;
//! - transforming primitive squared-coordinate derivatives to requested
//!   linear coordinates.
//!
//! The two semi-infinite exterior media do not contribute propagation
//! matrices and are therefore not used here. They are evaluated by the
//! plane-wave and outgoing-mode adapters, where their admittances define the
//! physical boundary conditions.
//!
//! If the finite layers are encountered as `L₁, L₂, …, Lₙ`, accumulation is:
//!
//! ```text
//! M = Lₙ … L₂ L₁
//! ```
//!
//! Value-only, first-order, and second-order calculations use distinct return
//! types so derivative arrays are allocated only when requested.

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{
        RunMode,
        isotropic::IsotropicLayerQuantities,
        transfer2::{Transfer2Entries, Transfer2Error, entries::Transfer2ExteriorContext},
    },
    input::{CanonicalCoordinates, CanonicalStack},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use super::{Transfer2, Transfer2Workspace};

impl<J> Transfer2<J> {
    pub(crate) fn accumulate<E, M>(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        stack: &CanonicalStack<M, J>,
        polarisation: Polarisation,
        request: RunMode,
    ) -> Result<Transfer2Workspace<J>, Transfer2Error>
    where
        J: ScalarAlgebra + ConstitutiveLift<E, M> + Clone,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Copy,
        J::Dimension: Dimension,
        E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
    {
        let context = Transfer2ExteriorContext::new(
            coordinates,
            stack.left_exterior(),
            stack.right_exterior(),
            polarisation,
        );

        let mut workspace = Transfer2Workspace::new(
            coordinates.vacuum_angular_wavenumber().value(),
            context,
            request,
            stack.layer_count(),
        );

        for layer in stack.layers() {
            let quantities = IsotropicLayerQuantities::evaluate::<E, M>(
                layer.material(),
                coordinates,
                polarisation,
            );

            let layer_matrix = Transfer2Entries::from_layer(&quantities, layer.thickness_cm());

            workspace.append(layer_matrix, quantities);
        }

        Ok(workspace)
    }
}
