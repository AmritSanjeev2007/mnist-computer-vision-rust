use computer_vision_rust::archive::{archive::SerializedData, deserializer::Deserializer};
use computer_vision_rust::constants;
use computer_vision_rust::load::{loader::TrainingMetadata, training_iterator::training_iterator};

use std::path::Path;

const IMAGES_DATASET_PATH: &str = "data/t10k-images.idx3-ubyte";
const LABELS_DATASET_PATH: &str = "data/t10k-labels.idx1-ubyte";
const MLP_PATH: &str = "test.nnmlp";

/// Returns the index of the largest element, i.e. the digit the MLP is most confident in.
fn predicted_label(output: &ndarray::Array1<f32>) -> Option<u8> {
    return Some(output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))?
        .0 as u8
    );
}

fn run() -> Result<(), String> {
    let images_path = Path::new(IMAGES_DATASET_PATH);
    let labels_path = Path::new(LABELS_DATASET_PATH);
    let metadata = TrainingMetadata::load(images_path, labels_path)?;
    metadata.validation_info()?;

    let mut serialized_data = SerializedData::new();
    serialized_data.deserialize(Path::new(MLP_PATH))?;
    let mlp = serialized_data.to_mlp()?;

    if mlp.get_neurons_per_layer().first() != Some(&constants::INPUT_VECTOR_LENGTH)
        || mlp.get_neurons_per_layer().last() != Some(&constants::OUTPUT_VECTOR_LENGTH)
    {
        return Err(format!(
            "MLP layer sizes {:?} do not match input size {} and output size {}.",
            mlp.get_neurons_per_layer(),
            constants::INPUT_VECTOR_LENGTH,
            constants::OUTPUT_VECTOR_LENGTH
        ));
    }

    let mut samples_tested = 0;
    let mut correct = 0;
    let mut correct_per_digit = [0_usize; constants::OUTPUT_VECTOR_LENGTH];
    let mut total_per_digit = [0_usize; constants::OUTPUT_VECTOR_LENGTH];

    for sample in training_iterator(images_path, labels_path) {
        let sample = sample?;
        let expected = sample
            .get_label()
            .ok_or_else(|| format!("Sample {} has no label.", sample.index))?;
        let predicted = predicted_label(&mlp.compute(&sample.input));

        total_per_digit[expected as usize] += 1;
        if predicted == Some(expected) {
            correct += 1;
            correct_per_digit[expected as usize] += 1;
        }
        samples_tested += 1;
    }

    if samples_tested != metadata.images_count {
        return Err(format!(
            "Testing iterator yielded {samples_tested} samples, expected {}.",
            metadata.images_count
        ));
    }

    for digit in 0..constants::OUTPUT_VECTOR_LENGTH {
        let total = total_per_digit[digit];
        let accuracy = if total == 0 { 0.0 } else { correct_per_digit[digit] as f32 / total as f32 };
        println!("{digit}: {}/{total} ({:.2}%)", correct_per_digit[digit], accuracy * 100.0);
    }

    println!(
        "Overall accuracy of {MLP_PATH}: {correct}/{samples_tested} ({:.2}%).",
        correct as f32 / samples_tested as f32 * 100.0
    );

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("MLP testing failed: {err}");
        std::process::exit(1);
    }
}
