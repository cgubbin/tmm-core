/// Converts an internal algebraic quantity into a public result.
///
/// `Mode` determines the derivative payload of the result.
pub(crate) trait Crystallise<Mode> {
    type Output;

    fn crystallise(self, mode: Mode) -> Self::Output;
}
