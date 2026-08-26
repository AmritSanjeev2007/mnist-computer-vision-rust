use computer_vision_rust::mlp::{MultiLayerPerceptron};
use computer_vision_rust::archive::{ archive, serializer::Serializer, deserializer::Deserializer };

use std::path::Path;

fn main() {
    let mut mlp = MultiLayerPerceptron::new([10, 7, 23].iter().map(|x| *x as usize));
    mlp.xavier_initialization();

    println!("{mlp:#?}");

    let mut serialized_data = archive::SerializedData::from_mlp(&mlp).unwrap();
    serialized_data.serialize_to_file(Path::new("tests/out/serialized.nnmlp")).unwrap();

    serialized_data.deserialize(Path::new("tests/out/serialized.nnmlp")).unwrap();

    let mlp2 = serialized_data.to_mlp().unwrap();

    println!("Same: {}.", mlp2 == mlp);
}