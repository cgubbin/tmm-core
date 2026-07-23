use super::DirectionalCoordinate;

#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalFirst<T> {
    coordinate: DirectionalCoordinate,
    first: T,
}

impl<T> DirectionalFirst<T> {
    pub(crate) fn new(coordinate: DirectionalCoordinate, first: T) -> Self {
        Self { coordinate, first }
    }

    pub fn coordinate(&self) -> DirectionalCoordinate {
        self.coordinate
    }

    pub fn first(&self) -> &T {
        &self.first
    }

    pub fn into_first(self) -> T {
        self.first
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> DirectionalFirst<U> {
        DirectionalFirst {
            coordinate: self.coordinate,
            first: f(self.first),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalSecond<T> {
    coordinate: DirectionalCoordinate,
    first: T,
    second: T,
}

impl<T> DirectionalSecond<T> {
    pub(crate) fn new(coordinate: DirectionalCoordinate, first: T, second: T) -> Self {
        Self {
            coordinate,
            first,
            second,
        }
    }

    pub fn coordinate(&self) -> DirectionalCoordinate {
        self.coordinate
    }
    pub fn first(&self) -> &T {
        &self.first
    }
    pub fn second(&self) -> &T {
        &self.second
    }

    pub fn into_first(self) -> DirectionalFirst<T> {
        DirectionalFirst {
            coordinate: self.coordinate,
            first: self.first,
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> DirectionalSecond<U> {
        DirectionalSecond {
            coordinate: self.coordinate,
            first: f(self.first),
            second: f(self.second),
        }
    }
}
