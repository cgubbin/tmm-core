macro_rules! delegate_analytical_material {
    ($type:ident) => {
        impl<R> crate::Material for $type<R>
        where
            R: num_traits::Float + std::fmt::Debug,
        {
            type Real = R;

            fn relative_permeability<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
            where
                C: crate::ComplexScalar<RealField = R> + Copy,
                I: crate::Sampled<Elem = R>,
            {
                self.inner.relative_permeability(vacuum_wavenumber)
            }

            fn relative_permittivity<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
            where
                C: crate::ComplexScalar<RealField = R> + Copy,
                I: crate::Sampled<Elem = R>,
            {
                self.inner.relative_permittivity(vacuum_wavenumber)
            }
        }

        impl<R> crate::DifferentiableMaterial for $type<R>
        where
            R: num_traits::Float + std::fmt::Debug,
        {
            fn relative_permittivity_derivative<I, C>(
                &self,
                vacuum_wavenumber: I,
                order: crate::DerivativeOrder,
            ) -> I::Mapped<C>
            where
                C: crate::ComplexScalar<RealField = R> + Copy,
                I: crate::Sampled<Elem = R>,
            {
                self.inner
                    .relative_permittivity_derivative(vacuum_wavenumber, order)
            }
        }

        impl<R> crate::MeromorphicMaterial for $type<R>
        where
            R: num_traits::Float + std::fmt::Debug,
        {
            fn relative_permeability_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
            where
                C: crate::ComplexScalar<RealField = R> + Copy,
                I: crate::Sampled<Elem = C>,
            {
                self.inner.relative_permeability_complex(vacuum_wavenumber)
            }

            fn relative_permittivity_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
            where
                C: crate::ComplexScalar<RealField = R> + Copy,
                I: crate::Sampled<Elem = C>,
            {
                self.inner.relative_permittivity_complex(vacuum_wavenumber)
            }
        }

        impl<R> crate::DifferentiableMeromorphicMaterial for $type<R>
        where
            R: num_traits::Float + std::fmt::Debug,
        {
            fn relative_permittivity_complex_derivative<I, C>(
                &self,
                vacuum_wavenumber: I,
                order: crate::DerivativeOrder,
            ) -> I::Mapped<C>
            where
                C: crate::ComplexScalar<RealField = R> + Copy,
                I: crate::Sampled<Elem = C>,
            {
                self.inner
                    .relative_permittivity_complex_derivative(vacuum_wavenumber, order)
            }
        }
    };
}

pub(crate) use delegate_analytical_material;
