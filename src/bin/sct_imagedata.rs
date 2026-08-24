use computer_vision_rust::load::{imagedata};

use const_format::formatcp;
use std::path::{Path};

fn main() {
    let mut img1 = imagedata::Image::new();
    img1.write_to_file(Path::new("tests/out/image-1.jpg"), image::ImageFormat::Jpeg).unwrap();

    let img2 = imagedata::Image::from_iter((0..784).into_iter().map(|x| ((x & 1) * 255) as u8));
    img2.write_to_file(Path::new("tests/out/image-2.jpg"), image::ImageFormat::Jpeg).unwrap();

    img1.transform_pixels(|p, x| *x = ((p.0 + p.1) & 1) * 255);
    img1.write_to_file(Path::new("tests/out/image-3.jpg"), image::ImageFormat::Jpeg).unwrap();

    let img3 = imagedata::Image::from_image_file(Path::new("tests/in/image-1.png")).unwrap();
    img3.write_to_file(Path::new("tests/out/image-4.jpg"), image::ImageFormat::Jpeg).unwrap();
}