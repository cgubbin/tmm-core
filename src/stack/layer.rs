use lamina_units::Length;

/// A finite material layer in a planar stack.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer<M, F> {
    material: M,
    thickness: Length<F>,
}

impl<M, F> Layer<M, F> {
    /// Construct a finite layer.
    pub fn new(material: M, thickness: Length<F>) -> Self {
        Self {
            material,
            thickness,
        }
    }

    /// Return the layer material.
    pub fn material(&self) -> &M {
        &self.material
    }

    /// Return the layer thickness in its caller-selected unit.
    pub fn thickness(&self) -> Length<F>
    where
        F: Copy,
    {
        self.thickness
    }

    /// Consume the layer into its material and thickness.
    pub fn into_parts(self) -> (M, Length<F>) {
        (self.material, self.thickness)
    }
}
