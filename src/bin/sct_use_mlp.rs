use computer_vision_rust::archive::{archive::SerializedData, deserializer::Deserializer};
use computer_vision_rust::load::imagedata::Image;
use std::path::Path;

const MLP_PATH: &str = "test.nnmlp";
const IMAGE_PATH: &str = "tests/in/image-1.png";

fn run() -> Result<(), String> {
    let mut serialized_data = SerializedData::new();
    serialized_data.deserialize(Path::new(MLP_PATH))?;

    let mlp = serialized_data.to_mlp()?;
    let image = Image::from_image_file(Path::new(IMAGE_PATH))?;
    let input = image.vector();

    if mlp.get_neurons_per_layer().first() != Some(&input.len()) {
        return Err(format!(
            "MLP input layer size {:?} does not match image vector size {}.",
            mlp.get_neurons_per_layer().first(),
            input.len()
        ));
    }

    for (number, probability) in mlp.compute(&input).iter().enumerate() {
        println!("{number}: {:.2}%", probability * 100.0);
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Failed to compute image probabilities: {err}");
        std::process::exit(1);
    }
}
