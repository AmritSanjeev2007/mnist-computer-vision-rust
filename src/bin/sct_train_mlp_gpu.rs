use computer_vision_rust::archive::{archive::SerializedData, serializer::Serializer};
use computer_vision_rust::constants;
use computer_vision_rust::mlp::MultiLayerPerceptron;
use computer_vision_rust::train::context::Context;
use computer_vision_rust::train::trainer::{DatasetRecord, Trainer, TrainerCreationOptions};

use std::path::{Path, PathBuf};
use std::sync::Arc;

const IMAGES_DATASET_PATH: &str = "data/train-images.idx3-ubyte";
const LABELS_DATASET_PATH: &str = "data/train-labels.idx1-ubyte";
const PROGRAM_PATH: &str = "calc/backprop.cl";
const OUTPUT_PATH: &str = "test.nnmlp";
const HIDDEN_LAYER_SIZE: usize = 30;
const BATCH_SIZE: usize = 32;
const EPOCH_COUNT: usize = 100;
const LEARNING_RATE: f32 = 0.1;

/// Passing this argument allows any OpenCL device (e.g., a CPU runtime) to be used when there is no GPU.
const ANY_DEVICE_ARG: &str = "--any-device";

fn run() -> Result<(), String> {
    let context = if std::env::args().any(|arg| arg == ANY_DEVICE_ARG) {
        Context::with_device_type(opencl3::device::CL_DEVICE_TYPE_ALL)?
    } else {
        Context::new()?
    };
    let device_name = context.get_preferred_device().name().unwrap_or_default();

    let mut mlp = MultiLayerPerceptron::new(
        [constants::INPUT_VECTOR_LENGTH, HIDDEN_LAYER_SIZE, constants::OUTPUT_VECTOR_LENGTH]
            .into_iter(),
    );
    mlp.xavier_initialization();

    let start = std::time::Instant::now();
    let mut trainer = Trainer::create_with_context(Arc::new(context), TrainerCreationOptions {
        batch_size: BATCH_SIZE,
        epoch_count: EPOCH_COUNT,
        dataset_path: DatasetRecord {
            imageset: PathBuf::from(IMAGES_DATASET_PATH),
            labelset: PathBuf::from(LABELS_DATASET_PATH),
        },
        program_path: PathBuf::from(PROGRAM_PATH),
        program_build_options: None,
        init_mlp: mlp,
    })?;
    println!("Loaded {} samples onto {device_name} in {:.2?}.", trainer.sample_count(), start.elapsed());

    for epoch in 1..=EPOCH_COUNT {
        let epoch_start = std::time::Instant::now();
        trainer.train_epoch(LEARNING_RATE)?;
        println!("Epoch {epoch}/{EPOCH_COUNT} took {:.2?}.", epoch_start.elapsed());
    }

    let mlp = trainer.current_mlp()?;
    let archive = SerializedData::from_mlp(&mlp)?;
    archive.serialize_to_file(Path::new(OUTPUT_PATH))?;
    println!(
        "Trained on {} samples for {EPOCH_COUNT} epochs in {:.2?} and saved the MLP to {OUTPUT_PATH}.",
        trainer.sample_count(),
        start.elapsed()
    );

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("GPU MLP training failed: {err}");
        std::process::exit(1);
    }
}
