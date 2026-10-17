// Kernels used by the GPU trainer (src/train/) to train a multi-layer perceptron.
//
// Every per-sample vector is stored batch-major: the value of neuron `i` for the `b`-th sample of a batch
// lives at `vec[b * n + i]`, where `n` is the number of neurons in that layer.
//
// Every weight matrix is stored row-major with dimensions (out_n, in_n), exactly like the ndarray matrices
// of `MultiLayerPerceptron`, i.e. `W[r][c]` lives at `W[r * in_n + c]`.
//
// The layer equations match the software-only implementation (src/mlp.rs, src/update_mlp.rs):
//
//      z_l     = W_l * a_l - b_l
//      a_{l+1} = relu(z_l)             (hidden layers)
//      output  = softmax(z_l)          (final layer)

/// z = W * a - b for every sample in the batch.
///
/// - Global work size: (out_n, batch_count).
/// - `in_offset` is the offset (in elements) into `in_act` of the first sample of the batch.
__kernel void linear_forward(
    __global const float* weights,
    __global const float* biases,
    __global const float* in_act,
    const uint in_offset,
    __global float* out_z,
    const uint in_n,
    const uint out_n)
{
    const uint r = get_global_id(0);
    const uint b = get_global_id(1);

    __global const float* w_row = weights + r * in_n;
    __global const float* a = in_act + in_offset + b * in_n;

    float sum = 0.0f;
    for (uint c = 0; c < in_n; ++c) {
        sum = fma(w_row[c], a[c], sum);
    }

    out_z[b * out_n + r] = sum - biases[r];
}

/// x = relu(x) in place.
///
/// - Global work size: (n * batch_count).
__kernel void relu(__global float* x)
{
    const uint i = get_global_id(0);
    x[i] = fmax(x[i], 0.0f);
}

/// Computes the error signal of the output layer for the softmax + cross-entropy loss:
///
///     delta = softmax(z) - answer
///
/// - Global work size: (batch_count).
/// - `answer_offset` is the offset (in elements) into `answers` of the first sample of the batch.
__kernel void output_delta(
    __global const float* z,
    __global const float* answers,
    const uint answer_offset,
    __global float* delta,
    const uint out_n)
{
    const uint b = get_global_id(0);

    __global const float* zb = z + b * out_n;
    __global const float* yb = answers + answer_offset + b * out_n;
    __global float* db = delta + b * out_n;

    // Subtracting the max does not change the softmax but keeps exp() from overflowing.
    float max_z = zb[0];
    for (uint i = 1; i < out_n; ++i) { max_z = fmax(max_z, zb[i]); }

    float sum = 0.0f;
    for (uint i = 0; i < out_n; ++i) { sum += exp(zb[i] - max_z); }

    for (uint i = 0; i < out_n; ++i) {
        db[i] = exp(zb[i] - max_z) / sum - yb[i];
    }
}

/// Backpropagates the error signal of layer connection `l` to the previous one:
///
///     delta_prev = (W_l^T * delta) .* relu'(a_l)
///
/// - Global work size: (in_n, batch_count).
/// - `prev_act` is the activated vector a_l (the input of layer connection `l`).
__kernel void backprop_delta(
    __global const float* weights,
    __global const float* delta,
    __global const float* prev_act,
    __global float* delta_prev,
    const uint in_n,
    const uint out_n)
{
    const uint c = get_global_id(0);
    const uint b = get_global_id(1);

    __global const float* db = delta + b * out_n;

    float sum = 0.0f;
    for (uint r = 0; r < out_n; ++r) {
        sum = fma(weights[r * in_n + c], db[r], sum);
    }

    const uint i = b * in_n + c;
    delta_prev[i] = prev_act[i] > 0.0f ? sum : 0.0f;
}

/// Performs a gradient descent step on layer connection `l`, averaging the gradient over the batch:
///
///     W_l -= learning_rate / batch_count * sum_b(delta_b * a_b^T)
///     b_l += learning_rate / batch_count * sum_b(delta_b)
///
/// The bias moves the opposite way to the weights because the forward pass subtracts it.
///
/// - Global work size: (in_n, out_n).
/// - `in_offset` is the offset (in elements) into `in_act` of the first sample of the batch.
__kernel void update_weights(
    __global float* weights,
    __global float* biases,
    __global const float* delta,
    __global const float* in_act,
    const uint in_offset,
    const uint in_n,
    const uint out_n,
    const uint batch_count,
    const float learning_rate)
{
    const uint c = get_global_id(0);
    const uint r = get_global_id(1);

    const float step = learning_rate / (float)batch_count;
    __global const float* a = in_act + in_offset;

    float grad = 0.0f;
    for (uint b = 0; b < batch_count; ++b) {
        grad = fma(delta[b * out_n + r], a[b * in_n + c], grad);
    }
    weights[r * in_n + c] -= step * grad;

    if (c == 0) {
        float bias_grad = 0.0f;
        for (uint b = 0; b < batch_count; ++b) { bias_grad += delta[b * out_n + r]; }
        biases[r] += step * bias_grad;
    }
}
