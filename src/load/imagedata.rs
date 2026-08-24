use crate::constants::{self as constants};

#[derive(Clone, Copy)]
pub struct Image
{
    _data: [u8; constants::IMAGE_SIZE],
}

impl Image
{
    /// Returns the width (in pixels) of the image.
    pub const fn width() -> usize { return constants::IMAGE_WIDTH; }
    /// Returns the height (in pixels) of the image.
    pub const fn height() -> usize { return constants::IMAGE_HEIGHT; }
    /// Returns the size (in pixels) of the image.
    pub const fn size() -> usize { return constants::IMAGE_SIZE; }

    /// Creates an empty image object filled with zeroes.
    pub fn new() -> Self {        
        return Self { _data: [0 as u8; Image::size()] };
    }


    /// Creates a new `Image` object and filling it using the specified slice.
    /// The function will **Panic** if the amount of elements in the slice is smaller than `Image::size()`.
    pub fn from_slice(slice: &[u8]) -> Self
    {
        debug_assert_eq!(slice.len(), Image::size());
        return Self { _data: slice.try_into().unwrap() };
    }
    /// Creates a new `Image` object and filling it using the specified iterator.
    /// The function will **Panic** if the iterator yields less elements than `Image::size()`.
    pub fn from_iter<T: Iterator<Item = u8>>(mut iter: T) -> Self
    {
        let mut arr = [0 as u8; Image::size()];
        arr.iter_mut().for_each(|x| *x = iter.next().unwrap());

        return Self { _data: arr };
    }
    /// Creates an `Image` object using a 1-D array. Note that the function **panics** if the size of the vector isn't the same as `IMAGE_SIZE`.
    pub fn from_vector(vec: &ndarray::Array1<f32>) -> Self
    {
        debug_assert_eq!(vec.len(), Image::size(), "Size of vector {} is not equal to required image data length {}", vec.len(), Image::size());

        let mut data = [0 as u8; Image::size()];
        data.iter_mut().enumerate().for_each(|x| *x.1 = (vec[x.0] * 255_f32) as u8);
        return Self { _data: data };
    }
    /// Creates an `Image` object from an image file. Note that this function will resize, and transform the color channels of the image if required.
    pub fn from_image_file(path: &std::path::Path) -> Result<Self, String>
    {
        let img_handle = image::open(path);
        if img_handle.is_err() {
            return Err(format!("Could not open image file due to error: {:?}", img_handle.err()));
        }

        let mut img = img_handle.unwrap();

        if (img.width(), img.height()) != (constants::IMAGE_WIDTH as u32, constants::IMAGE_HEIGHT as u32) {
            img = img.resize_exact(constants::IMAGE_WIDTH as u32, constants::IMAGE_HEIGHT as u32, image::imageops::Nearest);
        }

        debug_assert_eq!(img.width(), constants::IMAGE_WIDTH as u32, "Image width {} ≠ IMAGE_WIDTH: {}", img.width(), constants::IMAGE_WIDTH);
        debug_assert_eq!(img.height(), constants::IMAGE_HEIGHT as u32, "Image height {} ≠ IMAGE_HEIGHT: {}", img.height(), constants::IMAGE_HEIGHT);
        
        let gray = img.to_luma8();
        let mut gray_iter = gray.as_raw().iter();

        let mut data = [0 as u8; constants::IMAGE_SIZE];
        data.iter_mut().for_each(|x| *x = *gray_iter.next().unwrap());

        return Ok(Self { _data: data });
    }

    /// Converts an index into the `_data` array into an (x, y) coordinate. `x` refers to the position of the pixel along the width of the image.  
    const fn _idx_to_coord(idx: usize) -> Result<(u8, u8), &'static str> {
        if idx >= Image::size() { return Err("Out of bounds"); }
        return Ok(((idx / Image::width()) as u8, (idx % Image::width()) as u8))
    }
    /// Converts a coordinate into an index into the `_data` array into an (x, y) coordinate. `x` refers to the position of the pixel along the 
    /// width of the image.  
    const fn _coord_to_idx(point: (u8, u8)) -> Result<usize, &'static str> {
        let idx: usize = ((point.1 as usize) * Image::width()) + (point.0 as usize);

        if idx >= Image::size() { return Err("Out of bounds."); }
        return Ok(idx);
    }

    /// Tries to retrieve the value of the pixel at the specified coordinate.
    pub fn get_pixel(&self, point: (u8, u8)) -> Result<u8, &'static str> {
        return Ok(self._data[Image::_coord_to_idx(point)?]);
    }
    /// Tries to set the value of the pixel at the specified coordinate.
    pub fn set_pixel(&mut self, point: (u8, u8), value: u8) -> Result<(), &'static str> {
        self._data[Image::_coord_to_idx(point)?] = value;
        return Ok(());
    }

    /// Executes the transformer (*in a random order) for each pixel in the image. The function takes in the coordinates `(x, y)` of the pixel and a 
    /// mutable reference to it. Note that `x` refers to the position of the pixel along the width of the image.
    pub fn transform_pixels<Func>(&mut self, mut transformer: Func)
    where
        Func: FnMut((u8, u8), &mut u8)
    {
        self._data.iter_mut().enumerate().for_each(|x| transformer(Image::_idx_to_coord(x.0).unwrap(), x.1));
    }

    /// Returns the data of the image as a vector i.e., a 1-D array.
    pub fn vector(&self) -> ndarray::Array1<f32> {
        return ndarray::Array1::from_iter(self._data.iter().map(|x| (*x as f32)/255_f32));
    }

    /// Saves the content of this image to a file at the specified `path` (creating it if necessary) using the specified `image::ImageFormat`.
    pub fn write_to_file(&self, path: &std::path::Path, image_format: image::ImageFormat) -> Result<(), String> 
    {
        let img = image::GrayImage::from_raw(
            constants::IMAGE_WIDTH as u32,
            constants::IMAGE_HEIGHT as u32,
            self._data.to_vec()
        );

        if img.is_none() { return Err(format!("Could not create an image using image::GrayImage::from_raw(...)")); }

        let save_result = img.unwrap().save_with_format(path, image_format);

        if save_result.is_err() { return Err(format!("Failed to save image to path: {:?} due to error: {:?}", path, save_result.err())); }

        return Ok(());
    }
}


impl crate::common::Container<u8> for Image {
    fn iter(&self) -> std::slice::Iter<'_, u8> {
        return self._data.iter();
    }
    fn iter_mut(&mut self) -> std::slice::IterMut<'_, u8> {
        return self._data.iter_mut();
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Container;

    #[test]
    fn new_image_contains_only_zeroes() {
        let image = Image::new();

        assert_eq!(Image::size(), constants::IMAGE_SIZE);
        assert!(image.iter().all(|pixel| *pixel == 0));
    }

    #[test]
    fn from_slice_copies_correctly() {
        let arr: Vec<u8> = (0..Image::size()).map(|x| x as u8).collect();
        let image = Image::from_slice(arr.as_slice());

        assert_eq!(image._data[0], 0);
        assert_eq!(image._data[123], 123 as u8);
        assert_eq!(image._data[Image::size() - 6], (Image::size() - 6) as u8);
    }

    #[test]
    fn from_iter_preserves_pixel_data() {
        let image = Image::from_iter((0..constants::IMAGE_SIZE).map(|value| value as u8));

        assert_eq!(image._data[0], 0);
        assert_eq!(image._data[1], 1);
        assert_eq!(image._data[constants::IMAGE_SIZE - 1], 15);
    }

    #[test]
    #[should_panic]
    fn from_iter_panics_when_iterator_is_too_short() {
        Image::from_iter(std::iter::repeat_n(0, constants::IMAGE_SIZE - 1));
    }

    #[test]
    fn from_vector_scales_values_to_pixels() {
        let vector = ndarray::Array1::from_elem(constants::IMAGE_SIZE, 0.5);
        let image = Image::from_vector(&vector);

        assert_eq!(image._data[0], 127);
        assert_eq!(image.vector()[0], 127.0 / 255.0);
    }

    #[test]
    fn vector_returns_normalized_pixels() {
        let image = Image::from_iter(std::iter::once(255).chain(
            std::iter::repeat_n(0, constants::IMAGE_SIZE - 1),
        ));

        assert_eq!(image.vector()[0], 1.0);
        assert_eq!(image.vector()[1], 0.0);
    }

    #[test]
    fn missing_image_file_returns_error() {
        let result = Image::from_image_file(std::path::Path::new("does-not-exist.png"));

        assert!(result.is_err());
    }
}