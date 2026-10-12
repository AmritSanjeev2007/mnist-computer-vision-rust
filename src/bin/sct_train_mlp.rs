use computer_vision_rust::archive::{archive::SerializedData, serializer::Serializer};
use computer_vision_rust::constants;
use computer_vision_rust::load::{loader::TrainingMetadata, training_iterator::training_iterator};
use computer_vision_rust::mlp::MultiLayerPerceptron;
use computer_vision_rust::update_mlp::learn;

use std::path::Path;

const IMAGES_DATASET_PATH: &str = "data/train-images.idx3-ubyte";
const LABELS_DATASET_PATH: &str = "data/train-labels.idx1-ubyte";
const OUTPUT_PATH: &str = "test.nnmlp";
const HIDDEN_LAYER_SIZE: usize = 30;
const LEARNING_RATE: f32 = 0.01;

fn run() -> Result<(), String> {
    let images_path = Path::new(IMAGES_DATASET_PATH);
    let labels_path = Path::new(LABELS_DATASET_PATH);
    let metadata = TrainingMetadata::load(images_path, labels_path)?;
    metadata.validation_info()?;

    let mut mlp = MultiLayerPerceptron::new(
        [constants::INPUT_VECTOR_LENGTH, HIDDEN_LAYER_SIZE, constants::OUTPUT_VECTOR_LENGTH]
            .into_iter(),
    );
    mlp.xavier_initialization();

    let mut samples_trained = 0;
    for sample in training_iterator(images_path, labels_path) {
        let sample = sample?;
        learn(&mut mlp, &sample.input, &sample.answer, LEARNING_RATE)?;
        samples_trained += 1;
    }

    if samples_trained != metadata.images_count {
        return Err(format!(
            "Training iterator yielded {samples_trained} samples, expected {}.",
            metadata.images_count
        ));
    }

    let archive = SerializedData::from_mlp(&mlp)?;
    archive.serialize_to_file(Path::new(OUTPUT_PATH))?;
    println!("Trained on {samples_trained} samples and saved the MLP to {OUTPUT_PATH}.");

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("MLP training failed: {err}");
        std::process::exit(1);
    }
}
