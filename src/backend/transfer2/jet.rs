use crate::{
    ComplexScalar,
    backend::{
        input::IncidentSide,
        jet::{ArrayJet, ChainRuleScale, Jet, JetAdditive, JetBilinear, JetZeroLike},
        transfer2::Matrix2,
    },
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

pub(crate) type Transfer2Jet<C, D> = Jet<Matrix2<C, D>>;

impl<C, D> JetZeroLike for Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn zeros_like(shape_source: &Self) -> Self {
        Self::zeros_like(shape_source.m11())
    }
}

impl<C, D> JetAdditive for Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    fn jet_negate(&self) -> Self {
        Self::new(
            -self.m11().clone(),
            -self.m12().clone(),
            -self.m21().clone(),
            -self.m22().clone(),
        )
    }

    fn jet_subtract(&self, rhs: &Self) -> Self {
        self - rhs
    }
}

impl<C, D> JetBilinear for Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_multiply(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn jet_double(&self) -> Self {
        Self::new(
            self.m11().mapv(|x| x + x),
            self.m12().mapv(|x| x + x),
            self.m21().mapv(|x| x + x),
            self.m22().mapv(|x| x + x),
        )
    }
}

impl<C, D> ChainRuleScale<ArrayBase<OwnedRepr<C>, D>> for Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn scale_by(&self, coefficient: &ArrayBase<OwnedRepr<C>, D>) -> Self {
        self.scale_by_array(coefficient)
    }
}

pub(crate) struct Matrix2EntryJets<C, D>
where
    D: Dimension,
{
    pub(crate) m11: ArrayJet<C, D>,
    pub(crate) m12: ArrayJet<C, D>,
    pub(crate) m21: ArrayJet<C, D>,
    pub(crate) m22: ArrayJet<C, D>,
}

impl<C, D> Transfer2Jet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn scalar_jets(self) -> Matrix2EntryJets<C, D>
    where
        C: ComplexScalar,
        D: Dimension,
    {
        let (value, first, second) = self.into_parts();

        let (a, b, c, d) = value.into_parts();
        let (da, db, dc, dd) = first.into_parts();
        let (dda, ddb, ddc, ddd) = second.into_parts();

        Matrix2EntryJets {
            m11: ArrayJet::with_second(a, da, dda),
            m12: ArrayJet::with_second(b, db, ddb),
            m21: ArrayJet::with_second(c, dc, ddc),
            m22: ArrayJet::with_second(d, dd, ddd),
        }
    }

    pub(super) fn amplitude_jets(
        self,
        left_admittance: &ArrayJet<C, D>,
        right_admittance: &ArrayJet<C, D>,
        incident_side: IncidentSide,
    ) -> (ArrayJet<C, D>, ArrayJet<C, D>)
    where
        C: ComplexScalar,
        D: Dimension,
    {
        let scalar = self.scalar_jets();

        let a = scalar.m11;
        let b = scalar.m12;
        let c = scalar.m21;
        let d = scalar.m22;

        let two = ArrayJet::constant_like(a.value(), C::one() + C::one());

        let b_yr = b.multiply(right_admittance);
        let d_yr = d.multiply(right_admittance);

        let u = a.subtract(&b_yr);
        let v = c.subtract(&d_yr);

        let denominator = left_admittance.multiply(&u).subtract(&v);

        match incident_side {
            IncidentSide::Left => {
                let reflection = left_admittance.multiply(&u).add(&v).divide(&denominator);

                let transmission = two.multiply(left_admittance).divide(&denominator);

                (reflection, transmission)
            }

            IncidentSide::Right => {
                let p = a.add(&b_yr);
                let q = c.add(&d_yr);

                let reflection = q
                    .subtract(&left_admittance.multiply(&p))
                    .divide(&denominator);

                let determinant = a.multiply(&d).subtract(&b.multiply(&c));

                let transmission = two
                    .multiply(right_admittance)
                    .multiply(&determinant)
                    .divide(&denominator);

                (reflection, transmission)
            }
        }
    }
}
