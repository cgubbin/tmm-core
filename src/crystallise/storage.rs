use crate::{
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet0, Jet1, Jet2,
        JetBivariate1, JetBivariate2,
    },
    differential::{
        DifferentialResponse, DirectionalCoordinate, DirectionalFirst, DirectionalSecond,
        NoDerivatives, SpectralGradient, SpectralHessian, SpectralSecond,
    },
};

use ndarray::{Array, Dimension};

pub struct ValueOnly;

pub struct FirstDirectional {
    pub coordinate: DirectionalCoordinate,
}

pub struct SecondDirectional {
    pub coordinate: DirectionalCoordinate,
}

pub struct FirstSpectral;
pub struct SecondSpectral;

pub(crate) trait Crystallise: Sized {
    fn crystallise<C>(self, crystalliser: C) -> C::Output
    where
        C: CrystallisePolicy<Self>,
    {
        crystalliser.crystallise(self)
    }
}

impl<T> Crystallise for T {}

/// Converts an internal algebraic quantity into a public result.
pub(crate) trait CrystallisePolicy<J> {
    type Output;

    fn crystallise(self, jet: J) -> Self::Output;
}

impl<I, P> CrystallisePolicy<Jet0<I, P>> for ValueOnly {
    type Output = DifferentialResponse<I, NoDerivatives>;

    fn crystallise(self, value: Jet0<I, P>) -> Self::Output {
        DifferentialResponse::new(value.into_inner(), NoDerivatives)
    }
}

impl<I, P> CrystallisePolicy<Jet1<I, P>> for ValueOnly {
    type Output = DifferentialResponse<I, NoDerivatives>;

    fn crystallise(self, jet: Jet1<I, P>) -> Self::Output {
        let (values, ..) = jet.into_parts();
        DifferentialResponse::new(values, NoDerivatives)
    }
}

impl<I, P> CrystallisePolicy<Jet1<I, P>> for FirstDirectional {
    type Output = DifferentialResponse<I, DirectionalFirst<I>>;

    fn crystallise(self, jet: Jet1<I, P>) -> Self::Output {
        let (values, first) = jet.into_parts();
        DifferentialResponse::new(values, DirectionalFirst::new(self.coordinate, first))
    }
}

impl<I, P> CrystallisePolicy<Jet2<I, P>> for ValueOnly {
    type Output = DifferentialResponse<I, NoDerivatives>;

    fn crystallise(self, jet: Jet2<I, P>) -> Self::Output {
        let (values, ..) = jet.into_parts();
        DifferentialResponse::new(values, NoDerivatives)
    }
}

impl<I, P> CrystallisePolicy<Jet2<I, P>> for FirstDirectional {
    type Output = DifferentialResponse<I, DirectionalFirst<I>>;

    fn crystallise(self, jet: Jet2<I, P>) -> Self::Output {
        let (values, first, ..) = jet.into_parts();
        DifferentialResponse::new(values, DirectionalFirst::new(self.coordinate, first))
    }
}

impl<I, P> CrystallisePolicy<Jet2<I, P>> for SecondDirectional {
    type Output = DifferentialResponse<I, DirectionalSecond<I>>;

    fn crystallise(self, jet: Jet2<I, P>) -> Self::Output {
        let (values, first, second) = jet.into_parts();
        DifferentialResponse::new(
            values,
            DirectionalSecond::new(self.coordinate, first, second),
        )
    }
}

impl<I, P> CrystallisePolicy<JetBivariate1<I, P>> for ValueOnly {
    type Output = DifferentialResponse<I, NoDerivatives>;

    fn crystallise(self, jet: JetBivariate1<I, P>) -> Self::Output {
        let (values, ..) = jet.into_parts();
        DifferentialResponse::new(values, NoDerivatives)
    }
}

impl<I, P> CrystallisePolicy<JetBivariate1<I, P>> for FirstSpectral {
    type Output = DifferentialResponse<I, SpectralGradient<I>>;

    fn crystallise(self, jet: JetBivariate1<I, P>) -> Self::Output {
        let (values, gradient) = jet.into_parts();
        let (dx, dy) = gradient.into_parts();

        DifferentialResponse::new(values, SpectralGradient::new(dx, dy))
    }
}

impl<I, P> CrystallisePolicy<JetBivariate2<I, P>> for ValueOnly {
    type Output = DifferentialResponse<I, NoDerivatives>;

    fn crystallise(self, jet: JetBivariate2<I, P>) -> Self::Output {
        let (values, ..) = jet.into_parts();
        DifferentialResponse::new(values, NoDerivatives)
    }
}

impl<I, P> CrystallisePolicy<JetBivariate2<I, P>> for FirstSpectral {
    type Output = DifferentialResponse<I, SpectralGradient<I>>;

    fn crystallise(self, jet: JetBivariate2<I, P>) -> Self::Output {
        let (values, gradient, ..) = jet.into_parts();
        let (dx, dy) = gradient.into_parts();

        DifferentialResponse::new(values, SpectralGradient::new(dx, dy))
    }
}

impl<I, P> CrystallisePolicy<JetBivariate2<I, P>> for SecondSpectral {
    type Output = DifferentialResponse<I, SpectralSecond<I>>;

    fn crystallise(self, jet: JetBivariate2<I, P>) -> Self::Output {
        let (values, gradient, hessian) = jet.into_parts();
        let (dx, dy) = gradient.into_parts();
        let (dxdx, dxdy, dydy) = hessian.into_parts();

        DifferentialResponse::new(
            values,
            SpectralSecond::new(
                SpectralGradient::new(dx, dy),
                SpectralHessian::new(dxdx, dxdy, dydy),
            ),
        )
    }
}
