use crate::load::loader::TrainingMetadata;
use crate::load::training_iterator::training_iterator;

use crate::mlp::MultiLayerPerceptron;

use crate::train::context::Context;
use crate::train::pipeline::{ Pipeline, PipelineCreationInfo };

use std::path::PathBuf;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DatasetRecord
{
    pub imageset: PathBuf,
    pub labelset: PathBuf,
}

pub struct TrainerCreationOptions
{
    /// The number of training samples considered before the weights and biases are updated. The gradient is
    /// averaged over the batch, so a batch size of 1 behaves exactly like the software-only `update_mlp::learn`.
    pub batch_size: usize,
    /// The number of times the whole dataset is traversed.
    pub epoch_count: usize,

    pub dataset_path: DatasetRecord,
    /// The path to the OpenCL program containing the training kernels (e.g., calc/backprop.cl).
    pub program_path: PathBuf,
    /// An (optional) space-separated string of options used to build the program. If `None`, the pipeline's
    /// default build options are used.
    pub program_build_options: Option<String>,

    /// The multi-layer perceptron to start training from (e.g., a Xavier-initialized one).
    pub init_mlp: MultiLayerPerceptron,
}

/// Trains a multi-layer perceptron on the GPU via OpenCL.
///
/// The whole dataset is loaded once and uploaded to the GPU on creation; training then only dispatches kernels
/// and reads the trained multi-layer perceptron back at the end.
pub struct Trainer
{
    _dataset_record: DatasetRecord,
    _epoch_count: usize,

    _ctx: std::sync::Arc<Context>,
    _pipeline: Pipeline,
}

impl Trainer
{
    /// Tries to create a trainer on the first available GPU.
    pub fn create(options: TrainerCreationOptions) -> Result<Self, String>
    {
        return Self::create_with_context(std::sync::Arc::new(Context::new()?), options);
    }

    /// Tries to create a trainer which runs on the device of the specified context.
    pub fn create_with_context(context: std::sync::Arc<Context>, options: TrainerCreationOptions) -> Result<Self, String>
    {
        let imageset = &options.dataset_path.imageset;
        let labelset = &options.dataset_path.labelset;

        let metadata = TrainingMetadata::load(imageset, labelset)?;
        metadata.validation_info()?;

        let input_len = *options.init_mlp.get_neurons_per_layer().first().unwrap_or(&0);
        let output_len = *options.init_mlp.get_neurons_per_layer().last().unwrap_or(&0);

        let mut inputs: Vec<f32> = Vec::with_capacity(metadata.images_count * input_len);
        let mut answers: Vec<f32> = Vec::with_capacity(metadata.images_count * output_len);
        for sample in training_iterator(imageset, labelset)
        {
            let sample = sample?;
            if sample.input.len() != input_len || sample.answer.len() != output_len {
                return Err(format!("The dataset's samples do not fit the input and output layers of the multi-layer perceptron."));
            }
            inputs.extend(sample.input.iter());
            answers.extend(sample.answer.iter());
        }

        let pipeline = Pipeline::create(PipelineCreationInfo {
            context: context.clone(),
            program: options.program_path,
            program_build_options: options.program_build_options,
            compilation_output: None,
            batch_size: options.batch_size,
            dataset_inputs: &inputs,
            dataset_answers: &answers,
            init_mlp: &options.init_mlp,
        })?;

        return Ok(Self {
            _dataset_record: options.dataset_path,
            _epoch_count: options.epoch_count,
            _ctx: context,
            _pipeline: pipeline,
        });
    }

    /// Returns the number of samples in the dataset being trained on.
    pub fn sample_count(&self) -> usize {
        return self._pipeline.sample_count();
    }

    /// Traverses the whole dataset once and waits for the GPU to finish.
    pub fn train_epoch(&mut self, learning_rate: f32) -> Result<(), String>
    {
        let sample_count = self._pipeline.sample_count();
        let batch_size = self._pipeline.batch_size();

        for first_sample in (0..sample_count).step_by(batch_size)
        {
            let count = batch_size.min(sample_count - first_sample);
            self._pipeline.dispatch_batch(first_sample, count, learning_rate)?;
        }

        return self._pipeline.finish();
    }

    /// Trains for every epoch and returns the trained multi-layer perceptron.
    pub fn train(&mut self, learning_rate: f32) -> Result<MultiLayerPerceptron, String>
    {
        for _ in 0..self._epoch_count {
            self.train_epoch(learning_rate)?;
        }

        return self.current_mlp();
    }

    /// Reads the multi-layer perceptron, as trained so far, back from the GPU.
    pub fn current_mlp(&mut self) -> Result<MultiLayerPerceptron, String> {
        return self._pipeline.read_mlp();
    }
}
