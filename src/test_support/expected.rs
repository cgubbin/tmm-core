use super::{
    C, c,
    materials::{LinearDispersion, QuadraticDispersion},
};
use crate::input::Polarisation;

pub fn linear_epsilon(material: &LinearDispersion, k0: f64) -> C {
    c(material.epsilon0 + material.epsilon_slope * k0)
}

pub fn linear_mu(material: &LinearDispersion, k0: f64) -> C {
    c(material.mu0 + material.mu_slope * k0)
}

pub fn quadratic_epsilon(material: &QuadraticDispersion, k0: f64) -> C {
    c(material.epsilon0 + material.epsilon_slope * k0 + material.epsilon_curvature * k0 * k0)
}

pub fn quadratic_mu(material: &QuadraticDispersion, k0: f64) -> C {
    c(material.mu0 + material.mu_slope * k0 + material.mu_curvature * k0 * k0)
}

pub fn linear_kappa(material: &LinearDispersion, k0: f64, k_parallel: f64) -> C {
    let epsilon = linear_epsilon(material, k0);

    let mu = linear_mu(material, k0);

    (epsilon * mu * c(k0 * k0) - c(k_parallel * k_parallel)).sqrt()
}

pub fn quadratic_kappa(material: &QuadraticDispersion, k0: f64, k_parallel: f64) -> C {
    let epsilon = quadratic_epsilon(material, k0);

    let mu = quadratic_mu(material, k0);

    (epsilon * mu * c(k0 * k0) - c(k_parallel * k_parallel)).sqrt()
}

pub fn factor(epsilon: C, mu: C, polarisation: Polarisation) -> C {
    match polarisation {
        Polarisation::TransverseElectric => mu,
        Polarisation::TransverseMagnetic => epsilon,
    }
}

pub fn linear_admittance(
    material: &LinearDispersion,
    k0: f64,
    k_parallel: f64,
    polarisation: Polarisation,
) -> C {
    let epsilon = linear_epsilon(material, k0);

    let mu = linear_mu(material, k0);

    let kappa = linear_kappa(material, k0, k_parallel);

    kappa / factor(epsilon, mu, polarisation)
}

pub fn quadratic_admittance(
    material: &QuadraticDispersion,
    k0: f64,
    k_parallel: f64,
    polarisation: Polarisation,
) -> C {
    let epsilon = quadratic_epsilon(material, k0);

    let mu = quadratic_mu(material, k0);

    let kappa = quadratic_kappa(material, k0, k_parallel);

    kappa / factor(epsilon, mu, polarisation)
}
