/// Scale a value by a borrowed scalar-like factor.
pub(crate) trait ScaleBy<S> {
    fn scale_by(self, scale: &S) -> Self;
}
