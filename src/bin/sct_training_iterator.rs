use computer_vision_rust::load::imagedata;
use computer_vision_rust::load::training_iterator;

use std::path::Path;

const IMAGES_DATASET_PATH: &'static str = "data/train-images.idx3-ubyte";
const LABELS_DATASET_PATH: &'static str = "data/train-labels.idx1-ubyte";


fn main() {

    let specific_sample = 1238;

    for sample in training_iterator::training_iterator(
        Path::new(IMAGES_DATASET_PATH),
        Path::new(LABELS_DATASET_PATH)
    ) {
        if let Err(err) = sample { return println!("Iterator failed with error: {:?}.", err); }

        let entry = sample.unwrap();

        if entry.index < 9900 { continue; }
        if entry.index > 10000 { break; }

        let img = imagedata::Image::from_vector(&entry.input);
        img.write_to_file(Path::new(&format!("tests/out/image-{}={:?}.jpg", entry.index, entry.get_label().unwrap())), image::ImageFormat::Jpeg).unwrap();
    }
}