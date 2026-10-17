
use crate::mlp::MultiLayerPerceptron;

use crate::train::context::{ Buffer, BufferUsage, Context };
use crate::train::event::EventQueue;

use opencl3::kernel::ExecuteKernel;
use opencl3::types::cl_uint;

use std::io::Write;

pub type Program    = opencl3::program::Program;
pub type Kernel     = opencl3::kernel::Kernel;

struct _KernelData
{
    _linear_forward_kernel: Kernel,
    _relu_kernel: Kernel,
    _output_delta_kernel: Kernel,
    _backprop_delta_kernel: Kernel,
    _update_weights_kernel: Kernel,
}

struct _BufferData
{
    /// The input vectors of every sample in the dataset (batch-major).
    _dataset_input_buf: Buffer<f32>,
    /// The one-hot answer vectors of every sample in the dataset (batch-major).
    _dataset_answer_buf: Buffer<f32>,

    /// The weight matrix of each layer connection (row-major).
    _weight_bufs: Vec<Buffer<f32>>,
    /// The bias vector of each layer connection.
    _bias_bufs: Vec<Buffer<f32>>,
    /// The output of each layer connection for every sample in a batch. It holds `z = W * a - b` which is then
    /// activated in place (except for the final layer, whose softmax is folded into the `output_delta` kernel).
    _activation_out_bufs: Vec<Buffer<f32>>,
    /// The error signal (δ) of each layer connection for every sample in a batch.
    _delta_bufs: Vec<Buffer<f32>>,
}

/// The GPU-side state required to train a multi-layer perceptron: the compiled kernels, the multi-layer
/// perceptron itself and the whole dataset, which is uploaded once so that dispatching a batch needs no
/// CPU-GPU transfers.
pub struct Pipeline
{
    _ctx: std::sync::Arc<Context>,
    _event_queue: EventQueue,

    _prog: Program,

    _kernels: _KernelData,

    _neurons_per_layer: Vec<usize>,
    _batch_size: usize,
    _sample_count: usize,

    _buffers: _BufferData,
}

/// Data required for creating a new Pipeline.
pub struct PipelineCreationInfo<'a>
{
    /// A shared reference to an OpenCL context.
    pub context: std::sync::Arc<Context>,

    /// The path to the OpenCL program which contains all the required kernels.
    pub program: std::path::PathBuf,
    /// An (optional) space-separated string of the options used to control the building process of the program.
    /// If `None`, the default value used is `DEFAULT_PROGRAM_BUILD_OPTIONS`.
    pub program_build_options: Option<String>,
    /// An (optional) file stream to which to write the build output of the program to. If `None`, compilation output
    /// is discarded without writing to anything.
    pub compilation_output: Option<std::io::BufWriter<std::fs::File>>,

    /// The maximum number of training samples dispatched at once. The batch size is the number of training samples
    /// considered before the internal weights and biases of the multi-layer perceptron are updated.
    pub batch_size: usize,

    /// The input vectors of every training sample, laid out one after another.
    pub dataset_inputs: &'a [f32],
    /// The (one-hot) answer vectors of every training sample, laid out one after another.
    pub dataset_answers: &'a [f32],

    /// A multi-layer perceptron used as a starting point before training. The OpenCL buffers are initialized using
    /// the data from the supplied `MultiLayerPerceptron`.
    pub init_mlp: &'a MultiLayerPerceptron,
}

pub mod kernel_names
{
    pub const LINEAR_FORWARD: &'static str = "linear_forward";
    pub const RELU          : &'static str = "relu";
    pub const OUTPUT_DELTA  : &'static str = "output_delta";
    pub const BACKPROP_DELTA: &'static str = "backprop_delta";
    pub const UPDATE_WEIGHTS: &'static str = "update_weights";

    use frozen_collections::*;
    // The list of possible kernel names allowed.
    fz_string_set!(pub static KERNEL_NAMES: Set<&'static str>, {
        "linear_forward",
        "relu",
        "output_delta",
        "backprop_delta",
        "update_weights"
    });
}

/// The default program build options.
pub const DEFAULT_PROGRAM_BUILD_OPTIONS: &'static str = "-cl-no-signed-zeros -cl-mad-enable";

impl Pipeline
{
    /// Tries to create a new training pipeline object using the information from the specified `PipelineCreationInfo` object.
    pub fn create(mut info: PipelineCreationInfo) -> Result<Self, String>
    {
        info.init_mlp.validate()?;

        if info.batch_size == 0 { return Err(format!("The batch size must be at least 1.")); }

        let neurons_per_layer = info.init_mlp.get_neurons_per_layer().clone();
        let input_len = *neurons_per_layer.first().unwrap();
        let output_len = *neurons_per_layer.last().unwrap();

        if info.dataset_inputs.is_empty() || info.dataset_inputs.len() % input_len != 0 {
            return Err(format!("The dataset inputs must be a non-empty multiple of the input layer size ({input_len})."));
        }
        let sample_count = info.dataset_inputs.len() / input_len;
        if info.dataset_answers.len() != sample_count * output_len {
            return Err(format!("The dataset has {sample_count} inputs but {} answers.", info.dataset_answers.len() / output_len));
        }

        let mut event_queue = EventQueue::new(info.context.clone())?;

        let program = Self::_create_program(&mut info)?;
        let kernels = Self::_create_kernels(&program)?;
        let buffers = Self::_create_buffers(&info, &mut event_queue)?;

        // The buffers are written to asynchronously from slices which are only borrowed for the duration of this function.
        event_queue.finish()?;

        return Ok(Self {
            _batch_size: info.batch_size,
            _buffers: buffers,
            _ctx: info.context.clone(),
            _event_queue: event_queue,
            _kernels: kernels,
            _neurons_per_layer: neurons_per_layer,
            _prog: program,
            _sample_count: sample_count,
        });
    }

    /// Returns the number of samples in the dataset uploaded to the GPU.
    pub fn sample_count(&self) -> usize {
        return self._sample_count;
    }
    /// Returns the maximum number of samples which can be dispatched in one batch.
    pub fn batch_size(&self) -> usize {
        return self._batch_size;
    }

    /// Enqueues a single gradient descent step over the samples `[first_sample, first_sample + count)` of the dataset.
    ///
    /// This function doesn't wait for the GPU to finish; call `finish()` (or `read_mlp()`) to synchronize.
    pub fn dispatch_batch(&mut self, first_sample: usize, count: usize, learning_rate: f32) -> Result<(), String>
    {
        if count == 0 || count > self._batch_size {
            return Err(format!("Batch sample count ({count}) must be between 1 and the batch size ({}).", self._batch_size));
        }
        if first_sample + count > self._sample_count {
            return Err(format!("Batch [{first_sample}, {}) is out of the dataset's bounds ({}).", first_sample + count, self._sample_count));
        }

        let connections = self._neurons_per_layer.len() - 1;

        for layer in 0..connections {
            self._dispatch_linear_forward(layer, first_sample, count)?;
            if layer + 1 < connections {
                self._dispatch_relu(layer, count)?;
            }
        }

        self._dispatch_output_delta(first_sample, count)?;

        for layer in (0..connections).rev() {
            // The error signal of the previous layer must be computed with the weights before they are updated.
            if layer > 0 {
                self._dispatch_backprop_delta(layer, count)?;
            }
            self._dispatch_update_weights(layer, first_sample, count, learning_rate)?;
        }

        return Ok(());
    }

    /// Blocks the calling thread until every dispatched batch has completed.
    pub fn finish(&self) -> Result<(), String> {
        return self._event_queue.finish();
    }

    /// Waits for every dispatched batch to complete, then copies the weights and biases back from the GPU into a new
    /// multi-layer perceptron.
    pub fn read_mlp(&mut self) -> Result<MultiLayerPerceptron, String>
    {
        let connections = self._neurons_per_layer.len() - 1;

        let mut weights: Vec<Vec<f32>> = (0..connections).map(|l| vec![0_f32; self._buffers._weight_bufs[l].len()]).collect();
        let mut biases: Vec<Vec<f32>> = (0..connections).map(|l| vec![0_f32; self._buffers._bias_bufs[l].len()]).collect();

        for layer in 0..connections {
            self._event_queue.enqueue_read_buffer(&self._buffers._weight_bufs[layer], &mut weights[layer], 0)?;
            self._event_queue.enqueue_read_buffer(&self._buffers._bias_bufs[layer], &mut biases[layer], 0)?;
        }
        self._event_queue.finish()?;

        let weights: Vec<ndarray::Array2<f32>> = weights.into_iter().enumerate().map(|(layer, data)| {
            let shape = (self._neurons_per_layer[layer + 1], self._neurons_per_layer[layer]);
            return ndarray::Array2::from_shape_vec(shape, data)
                .map_err(|e| format!("Failed to reshape the weights of layer connection {layer} due to error: {e:?}."));
        }).collect::<Result<_, String>>()?;
        let biases: Vec<ndarray::Array1<f32>> = biases.into_iter().map(ndarray::Array1::from_vec).collect();

        return MultiLayerPerceptron::from_data(self._neurons_per_layer.clone(), weights, biases);
    }

// ===================================================================================================
// PRIVATES:        Creation
// ===================================================================================================

    fn _create_program(info: &mut PipelineCreationInfo) -> Result<Program, String>
    {
        let program_src = std::fs::read_to_string(&info.program)
            .map_err(|e| format!("Could not read program source file due to error: {e:?}."))?;
        let mut program = Program::create_from_source(
            info.context.native_context(),
            &program_src
        ).map_err(|e| format!("Failed to create program due to error: {e:?}."))?;

        // The context is only created on the preferred device, so the program is built only for that device.
        let device = info.context.get_preferred_device().id();
        let build_result = program.build(&[device], info.program_build_options.as_ref().map_or(DEFAULT_PROGRAM_BUILD_OPTIONS, |x| x));
        let build_log = program.get_build_log(device)
            .map_err(|e| format!("Failed to get build log due to error: {e:?}."))?;

        if let Some(output) = info.compilation_output.as_mut() {
            output.write_all(build_log.as_bytes())
                .and_then(|_| output.flush())
                .map_err(|e| format!("Failed to write to compilation_output file due to error: {e:?}."))?;
        }

        build_result.map_err(|e| format!("Failed to build program with error: {e:?}. Build log:\n{build_log}"))?;

        return Ok(program);
    }

    fn _create_kernels(program: &Program) -> Result<_KernelData, String>
    {
        use std::collections::HashSet;
        use frozen_collections::*;
        use kernel_names::*;

        let program_kernel_names = program.kernel_names();
        let kernel_names_set: HashSet<&str> = HashSet::from_iter(program_kernel_names.split(';'));

        if !KERNEL_NAMES.is_subset(&kernel_names_set) {
            return Err(format!("Program file doesn't contain all the required kernel names: {:?}.", KERNEL_NAMES));
        }

        let linear_forward_kernel = Kernel::create(&program, LINEAR_FORWARD)
            .map_err(|e| format!("Failed to create linear forward kernel due to error: {e:?}."))?;
        let relu_kernel = Kernel::create(&program, RELU)
            .map_err(|e| format!("Failed to create relu kernel due to error: {e:?}."))?;
        let output_delta_kernel = Kernel::create(&program, OUTPUT_DELTA)
            .map_err(|e| format!("Failed to create output delta kernel due to error: {e:?}."))?;
        let backprop_delta_kernel = Kernel::create(&program, BACKPROP_DELTA)
            .map_err(|e| format!("Failed to create backprop delta kernel due to error: {e:?}."))?;
        let update_weights_kernel = Kernel::create(&program, UPDATE_WEIGHTS)
            .map_err(|e| format!("Failed to create update weights kernel due to error: {e:?}."))?;

        return Ok(_KernelData {
            _linear_forward_kernel: linear_forward_kernel,
            _relu_kernel: relu_kernel,
            _output_delta_kernel: output_delta_kernel,
            _backprop_delta_kernel: backprop_delta_kernel,
            _update_weights_kernel: update_weights_kernel
        });
    }

    fn _create_buffers(info: &PipelineCreationInfo, queue: &mut EventQueue) -> Result<_BufferData, String>
    {
        let mut dataset_input_buf: Buffer<f32> = Buffer::create_empty(info.context.clone(), BufferUsage::ReadOnly as u64, info.dataset_inputs.len())?;
        queue.enqueue_write_buffer(&mut dataset_input_buf, info.dataset_inputs, 0)?;

        let mut dataset_answer_buf: Buffer<f32> = Buffer::create_empty(info.context.clone(), BufferUsage::ReadOnly as u64, info.dataset_answers.len())?;
        queue.enqueue_write_buffer(&mut dataset_answer_buf, info.dataset_answers, 0)?;

        let mut weight_bufs: Vec<Buffer<f32>> = Vec::with_capacity(info.init_mlp.get_weights().len());
        for mat in info.init_mlp.get_weights()
        {
            let buf_len = mat.len();
            let mut buf: Buffer<f32> = Buffer::create_empty(info.context.clone(), BufferUsage::ReadWrite as u64, buf_len)?;

            let mat_slice = mat.as_slice();
            if mat_slice.is_none() { return Err(format!("Could not get slice for weight matrix.")); }

            queue.enqueue_write_buffer(&mut buf, mat_slice.unwrap(), 0)?;
            weight_bufs.push(buf);
        }

        let mut bias_bufs: Vec<Buffer<f32>> = Vec::with_capacity(info.init_mlp.get_biases().len());
        for vec in info.init_mlp.get_biases()
        {
            let buf_len = vec.len();
            let mut buf: Buffer<f32> = Buffer::create_empty(info.context.clone(), BufferUsage::ReadWrite as u64, buf_len)?;

            let vec_slice = vec.as_slice();
            if vec_slice.is_none() { return Err(format!("Could not get slice for bias vector.")); }

            queue.enqueue_write_buffer(&mut buf, vec_slice.unwrap(), 0)?;
            bias_bufs.push(buf);
        }

        let mut activation_out_bufs: Vec<Buffer<f32>> = Vec::with_capacity(info.init_mlp.get_biases().len());
        let mut delta_bufs: Vec<Buffer<f32>> = Vec::with_capacity(info.init_mlp.get_biases().len());
        for vec in info.init_mlp.get_biases()
        {
            let buf_len = vec.len() * info.batch_size;
            activation_out_bufs.push(Buffer::create_empty(info.context.clone(), BufferUsage::ReadWrite as u64, buf_len)?);
            delta_bufs.push(Buffer::create_empty(info.context.clone(), BufferUsage::ReadWrite as u64, buf_len)?);
        }

        return Ok(_BufferData {
            _dataset_input_buf: dataset_input_buf,
            _dataset_answer_buf: dataset_answer_buf,
            _weight_bufs: weight_bufs,
            _bias_bufs: bias_bufs,
            _activation_out_bufs: activation_out_bufs,
            _delta_bufs: delta_bufs,
        });
    }

// ===================================================================================================
// PRIVATES:        Training
// ===================================================================================================

    /// Returns the buffer holding the input of the `layer`-th layer connection, along with the offset (in elements)
    /// of the batch's first sample. The first layer connection reads straight from the dataset.
    fn _layer_input<'b>(buffers: &'b _BufferData, neurons_per_layer: &[usize], layer: usize, first_sample: usize) -> (&'b Buffer<f32>, cl_uint)
    {
        if layer == 0 {
            return (&buffers._dataset_input_buf, (first_sample * neurons_per_layer[0]) as cl_uint);
        }
        return (&buffers._activation_out_bufs[layer - 1], 0);
    }

    fn _dispatch_linear_forward(&mut self, layer: usize, first_sample: usize, count: usize) -> Result<(), String>
    {
        let in_n = self._neurons_per_layer[layer] as cl_uint;
        let out_n = self._neurons_per_layer[layer + 1] as cl_uint;
        let (in_buf, in_offset) = Self::_layer_input(&self._buffers, &self._neurons_per_layer, layer, first_sample);

        let mut exec = ExecuteKernel::new(&self._kernels._linear_forward_kernel);
unsafe {
        exec.set_arg(self._buffers._weight_bufs[layer].get_native())
            .set_arg(self._buffers._bias_bufs[layer].get_native())
            .set_arg(in_buf.get_native())
            .set_arg(&in_offset)
            .set_arg(self._buffers._activation_out_bufs[layer].get_native())
            .set_arg(&in_n)
            .set_arg(&out_n)
            .set_global_work_sizes(&[out_n as usize, count]);
}
        self._event_queue.enqueue_kernel(&mut exec)?;
        return Ok(());
    }

    fn _dispatch_relu(&mut self, layer: usize, count: usize) -> Result<(), String>
    {
        let len = self._neurons_per_layer[layer + 1] * count;

        let mut exec = ExecuteKernel::new(&self._kernels._relu_kernel);
unsafe {
        exec.set_arg(self._buffers._activation_out_bufs[layer].get_native())
            .set_global_work_size(len);
}
        self._event_queue.enqueue_kernel(&mut exec)?;
        return Ok(());
    }

    fn _dispatch_output_delta(&mut self, first_sample: usize, count: usize) -> Result<(), String>
    {
        let last = self._neurons_per_layer.len() - 2;
        let out_n = *self._neurons_per_layer.last().unwrap() as cl_uint;
        let answer_offset = (first_sample * out_n as usize) as cl_uint;

        let mut exec = ExecuteKernel::new(&self._kernels._output_delta_kernel);
unsafe {
        exec.set_arg(self._buffers._activation_out_bufs[last].get_native())
            .set_arg(self._buffers._dataset_answer_buf.get_native())
            .set_arg(&answer_offset)
            .set_arg(self._buffers._delta_bufs[last].get_native())
            .set_arg(&out_n)
            .set_global_work_size(count);
}
        self._event_queue.enqueue_kernel(&mut exec)?;
        return Ok(());
    }

    /// Computes the error signal of layer connection `layer - 1` from the error signal of `layer`.
    fn _dispatch_backprop_delta(&mut self, layer: usize, count: usize) -> Result<(), String>
    {
        let in_n = self._neurons_per_layer[layer] as cl_uint;
        let out_n = self._neurons_per_layer[layer + 1] as cl_uint;

        let mut exec = ExecuteKernel::new(&self._kernels._backprop_delta_kernel);
unsafe {
        exec.set_arg(self._buffers._weight_bufs[layer].get_native())
            .set_arg(self._buffers._delta_bufs[layer].get_native())
            .set_arg(self._buffers._activation_out_bufs[layer - 1].get_native())
            .set_arg(self._buffers._delta_bufs[layer - 1].get_native())
            .set_arg(&in_n)
            .set_arg(&out_n)
            .set_global_work_sizes(&[in_n as usize, count]);
}
        self._event_queue.enqueue_kernel(&mut exec)?;
        return Ok(());
    }

    fn _dispatch_update_weights(&mut self, layer: usize, first_sample: usize, count: usize, learning_rate: f32) -> Result<(), String>
    {
        let in_n = self._neurons_per_layer[layer] as cl_uint;
        let out_n = self._neurons_per_layer[layer + 1] as cl_uint;
        let batch_count = count as cl_uint;
        let (in_buf, in_offset) = Self::_layer_input(&self._buffers, &self._neurons_per_layer, layer, first_sample);

        let mut exec = ExecuteKernel::new(&self._kernels._update_weights_kernel);
unsafe {
        exec.set_arg(self._buffers._weight_bufs[layer].get_native())
            .set_arg(self._buffers._bias_bufs[layer].get_native())
            .set_arg(self._buffers._delta_bufs[layer].get_native())
            .set_arg(in_buf.get_native())
            .set_arg(&in_offset)
            .set_arg(&in_n)
            .set_arg(&out_n)
            .set_arg(&batch_count)
            .set_arg(&learning_rate)
            .set_global_work_sizes(&[in_n as usize, out_n as usize]);
}
        self._event_queue.enqueue_kernel(&mut exec)?;
        return Ok(());
    }

}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::update_mlp::learn;

    const PROGRAM_PATH: &str = "calc/backprop.cl";

    /// Returns a context on any OpenCL device, or `None` (skipping the test) when the machine has none.
    fn _any_context() -> Option<std::sync::Arc<Context>>
    {
        // Some OpenCL runtimes fail to enumerate devices when several threads (tests) do it at the same time.
        static DEVICE_ENUMERATION: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = DEVICE_ENUMERATION.lock().unwrap_or_else(|e| e.into_inner());

        return match Context::with_device_type(opencl3::device::CL_DEVICE_TYPE_ALL) {
            Ok(ctx) => Some(std::sync::Arc::new(ctx)),
            Err(err) => { eprintln!("Skipping GPU test: {err}"); None }
        };
    }

    fn _test_mlp() -> MultiLayerPerceptron
    {
        let mut mlp = MultiLayerPerceptron::new([6, 5, 4, 3].into_iter());
        mlp.xavier_initialization();
        for layer in 0..3 {
            mlp.get_bias_indexed_iter_mut(layer).unwrap().for_each(|(i, b)| *b = 0.05 * (i as f32) - 0.1);
        }
        return mlp;
    }

    fn _test_dataset(sample_count: usize) -> (Vec<f32>, Vec<f32>)
    {
        let inputs = (0..sample_count * 6).map(|i| ((i * 7919) % 101) as f32 / 101.).collect();
        let mut answers = vec![0_f32; sample_count * 3];
        for s in 0..sample_count { answers[s * 3 + (s % 3)] = 1.; }
        return (inputs, answers);
    }

    fn _assert_mlps_roughly_equal(a: &MultiLayerPerceptron, b: &MultiLayerPerceptron)
    {
        for (wa, wb) in a.get_weights().iter().zip(b.get_weights()) {
            assert!(wa.iter().zip(wb.iter()).all(|(x, y)| (x - y).abs() < 1e-4), "weights differ:\n{wa}\n{wb}");
        }
        for (ba, bb) in a.get_biases().iter().zip(b.get_biases()) {
            assert!(ba.iter().zip(bb.iter()).all(|(x, y)| (x - y).abs() < 1e-4), "biases differ:\n{ba}\n{bb}");
        }
    }

    fn _create_pipeline<'a>(ctx: std::sync::Arc<Context>, batch_size: usize, inputs: &'a [f32], answers: &'a [f32], mlp: &'a MultiLayerPerceptron) -> Pipeline
    {
        return Pipeline::create(PipelineCreationInfo {
            context: ctx,
            program: PROGRAM_PATH.into(),
            program_build_options: None,
            compilation_output: None,
            batch_size: batch_size,
            dataset_inputs: inputs,
            dataset_answers: answers,
            init_mlp: mlp,
        }).unwrap();
    }

    #[test]
    fn read_mlp_returns_the_initial_mlp_before_training()
    {
        let Some(ctx) = _any_context() else { return; };
        let mlp = _test_mlp();
        let (inputs, answers) = _test_dataset(4);

        let mut pipeline = _create_pipeline(ctx, 2, &inputs, &answers, &mlp);

        assert_eq!(pipeline.read_mlp().unwrap(), mlp);
    }

    #[test]
    fn batch_size_of_one_matches_the_software_trainer()
    {
        let Some(ctx) = _any_context() else { return; };
        let mut cpu_mlp = _test_mlp();
        let (inputs, answers) = _test_dataset(8);
        let learning_rate = 0.1;

        let mut pipeline = _create_pipeline(ctx, 1, &inputs, &answers, &cpu_mlp);
        for s in 0..8 {
            pipeline.dispatch_batch(s, 1, learning_rate).unwrap();
        }
        let gpu_mlp = pipeline.read_mlp().unwrap();

        for s in 0..8 {
            let input = ndarray::Array1::from_vec(inputs[s * 6..(s + 1) * 6].to_vec());
            let answer = ndarray::Array1::from_vec(answers[s * 3..(s + 1) * 3].to_vec());
            learn(&mut cpu_mlp, &input, &answer, learning_rate).unwrap();
        }

        _assert_mlps_roughly_equal(&gpu_mlp, &cpu_mlp);
    }

    #[test]
    fn batch_step_matches_the_average_of_the_software_gradients()
    {
        let Some(ctx) = _any_context() else { return; };
        let mlp = _test_mlp();
        let (inputs, answers) = _test_dataset(4);
        let learning_rate = 0.2;

        let mut pipeline = _create_pipeline(ctx, 4, &inputs, &answers, &mlp);
        pipeline.dispatch_batch(0, 4, learning_rate).unwrap();
        let gpu_mlp = pipeline.read_mlp().unwrap();

        // Each per-sample step is taken from the same starting point, scaled down by the batch size, then summed.
        let mut expected = mlp.clone();
        for s in 0..4 {
            let mut step = mlp.clone();
            let input = ndarray::Array1::from_vec(inputs[s * 6..(s + 1) * 6].to_vec());
            let answer = ndarray::Array1::from_vec(answers[s * 3..(s + 1) * 3].to_vec());
            learn(&mut step, &input, &answer, learning_rate / 4.).unwrap();

            for layer in 0..3 {
                expected.get_weight_iter_mut(layer).unwrap()
                    .zip(step.get_weights()[layer].iter().zip(mlp.get_weights()[layer].iter()))
                    .for_each(|(e, (s, m))| *e += s - m);
                expected.get_bias_iter_mut(layer).unwrap()
                    .zip(step.get_biases()[layer].iter().zip(mlp.get_biases()[layer].iter()))
                    .for_each(|(e, (s, m))| *e += s - m);
            }
        }

        _assert_mlps_roughly_equal(&gpu_mlp, &expected);
    }

    #[test]
    fn dispatch_batch_rejects_out_of_bounds_batches()
    {
        let Some(ctx) = _any_context() else { return; };
        let mlp = _test_mlp();
        let (inputs, answers) = _test_dataset(4);

        let mut pipeline = _create_pipeline(ctx, 2, &inputs, &answers, &mlp);

        assert!(pipeline.dispatch_batch(0, 3, 0.1).is_err());
        assert!(pipeline.dispatch_batch(3, 2, 0.1).is_err());
        assert!(pipeline.dispatch_batch(0, 0, 0.1).is_err());
    }
}
