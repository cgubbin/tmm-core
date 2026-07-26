/// One internal layer prepared for backend evaluation.
///
/// `thickness_cm` is expressed in centimetres and uses the same jet algebra as
/// the canonical coordinates passed to the backend.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalLayer<M, J> {
    material: M,
    thickness_cm: J,
}

impl<M, J> CanonicalLayer<M, J> {
    pub(crate) fn new(material: M, thickness_cm: J) -> Self {
        Self {
            material,
            thickness_cm,
        }
    }

    pub fn material(&self) -> &M {
        &self.material
    }

    pub fn thickness_cm(&self) -> &J {
        &self.thickness_cm
    }

    pub fn into_parts(self) -> (M, J) {
        (self.material, self.thickness_cm)
    }
}

/// A validated stack prepared for backend evaluation.
///
/// Every internal-layer thickness:
///
/// - is expressed in centimetres;
/// - has the same sampled shape as the canonical coordinates;
/// - uses the same jet algebra as the canonical coordinates.
///
/// Construction is restricted to the compilation module so this type can
/// represent a validated invariant.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalStack<M, J> {
    left_exterior: M,
    right_exterior: M,
    layers_left_to_right: Vec<CanonicalLayer<M, J>>,
}

impl<M, J> CanonicalStack<M, J> {
    pub(crate) fn new(
        left_exterior: M,
        right_exterior: M,
        layers_left_to_right: Vec<CanonicalLayer<M, J>>,
    ) -> Self {
        Self {
            left_exterior,
            right_exterior,
            layers_left_to_right,
        }
    }

    pub fn left_exterior(&self) -> &M {
        &self.left_exterior
    }

    pub fn right_exterior(&self) -> &M {
        &self.right_exterior
    }

    pub fn layers(&self) -> &[CanonicalLayer<M, J>] {
        &self.layers_left_to_right
    }

    pub fn layer_count(&self) -> usize {
        self.layers_left_to_right.len()
    }

    pub fn into_parts(self) -> (M, M, Vec<CanonicalLayer<M, J>>) {
        (
            self.left_exterior,
            self.right_exterior,
            self.layers_left_to_right,
        )
    }
}
