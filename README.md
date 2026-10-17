# mnist-computer-vision-rust
A simple implementation of a training and computing mechanism for multi-layer perceptrons. This project specifically builds the mechanism around the MNIST database example.

The network is trained to recognise handwritten digits (0-9). It can be trained either on the CPU (software only) or on a GPU through OpenCL, and a trained network is saved to a small binary archive (`test.nnmlp`) that can then be tested or used on your own images.

## Requirements

- A Rust toolchain that supports the 2024 edition (Rust 1.85 or newer).
- The MNIST dataset in `data/`. It is stored with Git LFS, so run `git lfs pull` if the files are only pointers:
  - `data/train-images.idx3-ubyte`, `data/train-labels.idx1-ubyte` (60,000 training samples)
  - `data/t10k-images.idx3-ubyte`, `data/t10k-labels.idx1-ubyte` (10,000 test samples)
- For GPU training only: an OpenCL runtime (normally installed with your GPU driver) and the OpenCL loader library (`OpenCL.lib` on Windows, `libOpenCL` elsewhere) where the linker can find it, since the `opencl3` crate links against it.

Every command below is run from the repository root, because the programs use paths relative to it (`data/`, `calc/`, `test.nnmlp`).

## Project layout

| Path | Purpose |
| --- | --- |
| `src/mlp.rs` | The `MultiLayerPerceptron` itself: layers, Xavier initialisation and the forward pass (`compute`). |
| `src/update_mlp.rs` | Software-only backpropagation (`learn`), one sample at a time. |
| `src/train/` | The OpenCL GPU trainer (`Trainer`, `Pipeline`, `Context`). |
| `calc/backprop.cl` | The OpenCL kernels used by the GPU trainer. |
| `src/load/` | Readers for the MNIST files and for ordinary image files. |
| `src/archive/` | Saving an MLP to, and loading it from, a `.nnmlp` file. |
| `src/bin/sct_*.rs` | System component tests (SCTs): small programs that exercise the whole system. |
| `neural-network.md` | A full derivation of the gradients used for backpropagation. |

## Training

Both trainers build a network with the layout **784 → 30 → 10** (one input per pixel, 30 hidden neurons, one output per digit), initialise it with Xavier initialisation, train it on the 60,000 training samples and write the result to `test.nnmlp`, overwriting any previous network.

Use `--release`: a debug build is many times slower.

### On the CPU

```sh
cargo run --release --bin sct_train_mlp
```

This walks through the training set once, updating the weights after every sample (stochastic gradient descent) with a learning rate of `0.01`.

### On the GPU (OpenCL)

```sh
cargo run --release --bin sct_train_mlp_gpu
```

The whole dataset is uploaded to the GPU once, then the kernels in `calc/backprop.cl` run the forward pass, the backpropagation and the weight updates for a whole batch in parallel. The network is only read back at the end. The settings are:

| Setting | Value |
| --- | --- |
| Batch size | 32 (the gradient is averaged over the batch) |
| Epochs | 100 (full passes over the training set) |
| Learning rate | 0.1 |

The program prints the device it uses, the time each epoch took and the total time.

By default only GPU devices are considered. If your machine has no GPU but does have another OpenCL runtime (for example a CPU runtime such as Intel's or PoCL), allow any OpenCL device:

```sh
cargo run --release --bin sct_train_mlp_gpu -- --any-device
```

The hidden layer size, batch size, epoch count and learning rate are constants at the top of `src/bin/sct_train_mlp.rs` and `src/bin/sct_train_mlp_gpu.rs`; edit them to experiment.

## Testing

```sh
cargo run --release --bin sct_test_mlp
```

This loads `test.nnmlp`, runs every one of the 10,000 test images (`data/t10k-*`, which the network never saw during training) through it, and takes the digit with the highest output as the prediction. It prints the accuracy for each digit and the overall accuracy, for example:

```text
0: 968/980 (98.78%)
...
Overall accuracy of test.nnmlp: 9612/10000 (96.12%).
```

(The numbers above are illustrative; yours depend on which trainer you used and on the random initialisation.)

The unit tests, including checks that the GPU trainer produces the same gradients as the software trainer, run with:

```sh
cargo test
```

The GPU tests need an OpenCL device to be available.

### Running SCTs by name

The SCTs listed in `src/test/sct-list.jsonc` can also be started through the main binary:

```sh
cargo run test sct_train_mlp_gpu
```

This simply runs `cargo run --bin <name>` with the options listed for it in that file (a debug build, with no extra arguments unless you add them there).

## Using a trained MLP

```sh
cargo run --release --bin sct_use_mlp
```

This loads `test.nnmlp`, reads `tests/in/image-1.png`, and prints the network's confidence for each digit:

```text
0: 0.01%
1: 0.20%
2: 97.85%
...
```

To classify your own image, change `IMAGE_PATH` in `src/bin/sct_use_mlp.rs`. Any format the `image` crate can open works; the image is resized to 28×28 and converted to grayscale. Draw the digit in **white on a black background**, roughly centred, like the MNIST images, since colours are not inverted.

From your own Rust code the same steps look like this:

```rust
use computer_vision_rust::archive::{archive::SerializedData, deserializer::Deserializer};
use computer_vision_rust::load::imagedata::Image;
use std::path::Path;

let mut archive = SerializedData::new();
archive.deserialize(Path::new("test.nnmlp"))?;
let mlp = archive.to_mlp()?;

let input = Image::from_image_file(Path::new("my-digit.png"))?.vector(); // 784 values in [0, 1]
let probabilities = mlp.compute(&input);                                 // 10 values summing to 1
```

## How an MLP works

### Computing an answer (the forward pass)

A multi-layer perceptron is a chain of layers of "neurons", where every neuron in one layer is connected to every neuron in the next. Here the input layer has 784 neurons, one per pixel of the 28×28 image, each holding that pixel's brightness scaled to between 0 and 1.

Each connection between two layers is described by a **weight matrix** $W_l$ and a **bias vector** $b_l$. To get the next layer, every neuron takes a weighted sum of all the neurons before it and shifts it by its bias:

$$ z_l = W_l\,a_l - b_l $$

(this project subtracts the bias; adding it is the more common convention, and the two are equivalent).

Without anything else, stacking layers would still only produce one big linear function. So each hidden layer passes $z_l$ through a non-linear **activation function**. This project uses ReLU, which keeps positive values and turns negative ones into 0:

$$ a_{l+1} = \mathrm{relu}(z_l) = \max(0, z_l) $$

The final layer instead uses **softmax**, which turns its 10 raw scores into probabilities that are all positive and add up to 1:

$$ \hat y_k = \frac{e^{z_k}}{\sum_j e^{z_j}} $$

$\hat y_k$ is the network's confidence that the image shows digit $k$, and the largest one is its answer. All the network "knows" is stored in the numbers inside the $W$s and $b$s; for a 784 → 30 → 10 network that is 784×30 + 30 + 30×10 + 10 = 23,860 numbers.

### Learning (backpropagation)

A freshly initialised network gives essentially random answers. Learning means nudging every weight and bias so that the answers improve.

1. **Measure how wrong the answer is.** For a training image whose true digit is known, the correct answer is a "one-hot" vector $y$ (1 for the right digit, 0 elsewhere). The **cross-entropy loss** measures how far the prediction is from it:

   $$ L = -\sum_k y_k \ln \hat y_k $$

   It is 0 when the network is 100% confident in the right digit and grows the less confident it is.

2. **Find the error at the output.** With softmax and cross-entropy together, the derivative of the loss with respect to the output layer's raw scores is remarkably simple:

   $$ \delta_{\text{out}} = \frac{\partial L}{\partial z_{\text{out}}} = \hat y - y $$

   That is, the error signal is just "what the network said minus what it should have said".

3. **Pass the error backwards.** Using the chain rule, the error signal of a layer is sent back through the same weights it came forward through, and is kept only where the ReLU was active (ReLU's slope is 1 for positive inputs and 0 otherwise):

   $$ \delta_{l-1} = \left(W_l^{\mathsf T}\,\delta_l\right) \odot \mathrm{relu}'(z_{l-1}) $$

   This is the "back" in backpropagation: each layer learns how much it contributed to the final mistake.

4. **Adjust the weights and biases.** The error signal tells us the gradient, i.e. which direction increases the loss, for every parameter: $\partial L/\partial W_l = \delta_l\,a_l^{\mathsf T}$ for the weights, and $\delta_l$ for the biases (with the opposite sign here because the bias is subtracted). **Gradient descent** takes a small step the other way, scaled by the learning rate $\eta$:

   $$ W_l \leftarrow W_l - \eta\,\delta_l\,a_l^{\mathsf T}, \qquad b_l \leftarrow b_l + \eta\,\delta_l $$

Repeating this over tens of thousands of examples gradually shapes the weights into detectors for strokes and shapes that distinguish one digit from another.

The CPU trainer (`src/update_mlp.rs`) performs these steps one sample at a time. The GPU trainer (`src/train/`, `calc/backprop.cl`) performs the same steps for a whole batch of 32 samples at once, with one GPU work-item per neuron and sample, then averages the batch's gradients before updating. That massive parallelism is where its speed-up comes from. With a batch size of 1 it produces exactly the same updates as the CPU trainer.

For the full derivation of these formulas, including the softmax and cross-entropy derivatives, see [neural-network.md](neural-network.md).

## License

See [LICENSE](LICENSE).
