use super::Thickness;

#[derive(Clone, Debug, PartialEq)]
pub struct Layer<M, F> {
    material: M,
    thickness: Thickness<F>,
}

impl<M, F> Layer<M, F> {
    pub fn new(material: M, thickness: Thickness<F>) -> Self {
        Self {
            material,
            thickness,
        }
    }

    pub fn material(&self) -> &M {
        &self.material
    }

    pub fn thickness(&self) -> Thickness<F>
    where
        F: Copy,
    {
        self.thickness
    }
}
