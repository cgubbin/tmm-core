use num_traits::Float;

use super::model::{Drude, DrudeLorentz, Lorentz};

#[derive(Clone, Debug)]
pub struct DrudeLorentzBuilder<R> {
    epsilon_infinity: R,
    drude: Option<Drude<R>>,
    lorentz: Vec<Lorentz<R>>,
}

impl<R> DrudeLorentzBuilder<R> {
    pub fn new(epsilon_infinity: R) -> Self {
        Self {
            epsilon_infinity,
            drude: None,
            lorentz: Vec::new(),
        }
    }

    pub fn with_drude(mut self, plasma_frequency: R, damping_frequency: R) -> Self {
        self.drude = Some(Drude {
            plasma_frequency,
            damping_frequency,
        });
        self
    }

    pub fn with_lorentz(
        mut self,
        strength: R,
        transverse_frequency: R,
        damping_frequency: R,
    ) -> Self {
        self.lorentz.push(Lorentz {
            strength,
            transverse_frequency,
            damping_frequency,
        });
        self
    }

    pub fn build(self) -> DrudeLorentz<R> {
        DrudeLorentz::from_parts(self.epsilon_infinity, self.drude, self.lorentz)
    }
}

impl<R> DrudeLorentzBuilder<R>
where
    R: Float,
{
    pub fn with_lorentz_from_frequencies(
        mut self,
        epsilon_infinity: R,
        longitudinal_frequency: R,
        transverse_frequency: R,
        damping_frequency: R,
    ) -> Self {
        self.lorentz.push(Lorentz::from_frequencies(
            epsilon_infinity,
            longitudinal_frequency,
            transverse_frequency,
            damping_frequency,
        ));
        self
    }
}
