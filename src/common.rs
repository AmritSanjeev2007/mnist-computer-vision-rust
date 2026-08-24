pub trait Container<T>
{
    fn iter(&self) -> std::slice::Iter<'_, T>;
    fn iter_mut(&mut self) -> std::slice::IterMut<'_, T>;
}

// Allows: for item in &container
impl<'a, T> IntoIterator for &'a dyn Container<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        return self.iter();
    }
}

// Allows: for item in &mut container
impl<'a, T> IntoIterator for &'a mut dyn Container<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        return self.iter_mut();
    }
}
