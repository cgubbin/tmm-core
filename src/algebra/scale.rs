pub(crate) trait ScaleBy<S> {
    fn scale_by(self, scale: &S) -> Self;
}
