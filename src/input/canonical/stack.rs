//! Canonical planar stacks prepared for backend evaluation.
//!
//! Canonical stacks retain the physical left-to-right ordering supplied by the
//! caller. Numerical backends traverse the stack in that fixed geometric
//! direction, independent of the side from which a plane wave is incident.
//!
//! Incidence direction is therefore not encoded in these types. It is used
//! later when projecting a solved transfer or scattering representation into
//! plane-wave observables such as reflection and transmission amplitudes.
//!
//! Every finite-layer thickness:
//!
//! - is expressed in centimetres;
//! - has the same sampled shape as the canonical coordinates;
//! - uses the same jet algebra as the canonical coordinates.
//!
//! Construction is restricted to the input-compilation layer, so these types
//! represent already-established canonical invariants.

/// One finite layer prepared for backend evaluation.
///
/// The layer thickness is expressed in centimetres and uses the same sampled
/// algebraic representation as the canonical coordinates.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalLayer<M, J> {
    material: M,
    thickness_cm: J,
}

impl<M, J> CanonicalLayer<M, J> {
    /// Construct a canonical finite layer.
    pub(crate) const fn new(material: M, thickness_cm: J) -> Self {
        Self {
            material,
            thickness_cm,
        }
    }

    /// Return the layer material.
    pub(crate) fn material(&self) -> &M {
        &self.material
    }

    /// Return the layer thickness in centimetres.
    pub(crate) fn thickness_cm(&self) -> &J {
        &self.thickness_cm
    }

    /// Consume the layer and return `(material, thickness_cm)`.
    pub(crate) fn into_parts(self) -> (M, J) {
        (self.material, self.thickness_cm)
    }
}

/// A validated planar stack prepared for backend evaluation.
///
/// The stack is stored in its physical geometric order:
///
/// ```text
/// left exterior -> finite layers -> right exterior
/// ```
///
/// Backends always traverse this ordering from left to right. The side of
/// plane-wave incidence is not represented here and does not alter the stack
/// ordering. It is applied later when solved backend quantities are projected
/// into observable amplitudes and powers.
///
/// Every finite-layer thickness:
///
/// - is expressed in centimetres;
/// - has the same sampled shape as the canonical coordinates;
/// - uses the same jet algebra as the canonical coordinates.
///
/// Construction is restricted to input compilation, so this type represents a
/// validated canonical stack rather than an unchecked collection of layers.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalStack<M, J> {
    left_exterior: M,
    right_exterior: M,
    layers_left_to_right: Vec<CanonicalLayer<M, J>>,
}

impl<M, J> CanonicalStack<M, J> {
    /// Construct a canonical stack in geometric left-to-right order.
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

    /// Return the material in the left exterior half-space.
    pub(crate) fn left_exterior(&self) -> &M {
        &self.left_exterior
    }

    /// Return the material in the right exterior half-space.
    pub(crate) fn right_exterior(&self) -> &M {
        &self.right_exterior
    }

    /// Return the finite layers in geometric left-to-right order.
    pub(crate) fn layers(&self) -> &[CanonicalLayer<M, J>] {
        &self.layers_left_to_right
    }

    /// Return the number of finite layers.
    pub(crate) fn layer_count(&self) -> usize {
        self.layers_left_to_right.len()
    }

    /// Consume the stack and return its geometric components.
    ///
    /// The returned tuple contains:
    ///
    /// 1. the left exterior material;
    /// 2. the right exterior material;
    /// 3. the finite layers in left-to-right order.
    pub(crate) fn into_parts(self) -> (M, M, Vec<CanonicalLayer<M, J>>) {
        (
            self.left_exterior,
            self.right_exterior,
            self.layers_left_to_right,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_preserves_material_and_thickness() {
        let layer = CanonicalLayer::new("film", vec![0.1, 0.2]);

        assert_eq!(layer.material(), &"film");
        assert_eq!(layer.thickness_cm(), &vec![0.1, 0.2]);

        assert_eq!(layer.into_parts(), ("film", vec![0.1, 0.2]),);
    }

    #[test]
    fn stack_preserves_exteriors_and_geometric_layer_order() {
        let layers = vec![
            CanonicalLayer::new("first", 1.0),
            CanonicalLayer::new("second", 2.0),
            CanonicalLayer::new("third", 3.0),
        ];

        let stack = CanonicalStack::new("left", "right", layers.clone());

        assert_eq!(stack.left_exterior(), &"left");
        assert_eq!(stack.right_exterior(), &"right");
        assert_eq!(stack.layers(), layers.as_slice());
        assert_eq!(stack.layer_count(), 3);

        let (left_exterior, right_exterior, returned_layers) = stack.into_parts();

        assert_eq!(left_exterior, "left");
        assert_eq!(right_exterior, "right");
        assert_eq!(returned_layers, layers);
    }

    #[test]
    fn empty_stack_has_no_finite_layers() {
        let stack = CanonicalStack::<_, f64>::new("left", "right", Vec::new());

        assert_eq!(stack.left_exterior(), &"left");
        assert_eq!(stack.right_exterior(), &"right");
        assert!(stack.layers().is_empty());
        assert_eq!(stack.layer_count(), 0);
    }
}
