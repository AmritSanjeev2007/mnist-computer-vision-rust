use crate::mlp::{ MultiLayerPerceptron, relu };

use ndarray::Array2 as Matrix;
use ndarray::Array1 as Vector;


/// Returns an array of the activated vectors at each layer (except the final output). Specifically, for a
/// multi-layer perceptron:
/// 
/// `input -> a_1 -> a_2 -> ... -> a_{n-1} -> output`,
/// 
/// the function returns `[input, a_1, a_2, ..., a_{n-1}]`
/// 
fn get_activated_vectors(mlp: &MultiLayerPerceptron, input: &Vector<f32>) -> Vec<Vector<f32>>
{
    let mut output = input.clone();
    return std::iter::once(input.clone()).chain(
        (0..mlp.num_layers() - 2).map(|i| {
            let tmp = mlp.get_weights()[i].dot(&output) - &mlp.get_biases()[i];
            output = relu(&tmp);
            return output.clone();
        })
    ).collect();
}

/// Returns relu'(x) i.e., d/dx relu(x)
fn _relu_derivative(x: &Vector<f32>) -> Vector<f32>
{
    return x.iter().map(|u| if *u > 0. { 1. } else { 0. }).collect();
}

/// Gets the error signal (δ<sub>L-1</sub>) for the previous layer given:
/// * The ``weight`` of the current layer (W<sub>L</sub>).
/// * The error signal of the current layer (δ<sub>L</sub>).
/// * The activated vector of the previous layer (a<sub>L-1</sub>).
fn _get_backpropagated_err_signal(weight: &Matrix<f32>, err_signal: &Vector<f32>, prev_activated_vector: &Vector<f32>) -> Vector<f32>
{
    return weight.t().dot(err_signal) * _relu_derivative(prev_activated_vector);
}

/// Updates the weight ( W<sub>L</sub> ) and bias ( b<sub>L</sub> ) of the current layer `L` given:
/// * The error signal of the current layer (δ<sub>L</sub>).
/// * The activated vector of the previous layer (a<sub>L-1</sub>).
fn _update_layer(
    mlp: &mut MultiLayerPerceptron,
    layer_index: usize,
    err_signal: &Vector<f32>,
    prev_activated_vector: &Vector<f32>,
    learning_rate: f32,
)
    -> Result<(), String>
{
    let weight_grad = Matrix::from_shape_fn(
        (err_signal.len(), prev_activated_vector.len()),
        |(row, column)| err_signal[row] * prev_activated_vector[column],
    );

    mlp.get_weight_iter_mut(layer_index)?
        .zip(weight_grad.iter())
        .for_each(|(weight, gradient)| *weight -= learning_rate * *gradient);
    // The forward pass subtracts stored biases, so gradient descent adds the error signal here.
    mlp.get_bias_iter_mut(layer_index)?
        .zip(err_signal.iter())
        .for_each(|(bias, gradient)| *bias += learning_rate * *gradient);

    return Ok(());
}


/// Tries to teach the specified multi-layer perceptron to produce an output closer to the true answer. The
/// function uses the cross-entropy loss method to calculate how far the output is from the true answer. It is
/// defined as follows:
/// 
/// $$ L = -\sum_k y_k\ln(\hat y_k) $$
/// 
/// where $y$ is the true answer and $\hat y$ is the output of the multi-layer perceptron. 
/// 
/// This function is software-only. This function doesn't make use of the GPU.
///
pub fn learn(mlp: &mut MultiLayerPerceptron, input: &Vector<f32>, answer: &Vector<f32>, learning_rate: f32) -> Result<(), String>
{
    mlp.validate()?;

    if *mlp.get_neurons_per_layer().first().unwrap() != input.len() {
        return Err(format!("Length of input vector doesn't match the number of neurons in the input layer of the multi-layer perceptron."));
    }

    if *mlp.get_neurons_per_layer().last().unwrap() != answer.len() {
        return Err(format!("Length of output vector doesn't match the number of neurons in the output layer of the multi-layer perceptron."));
    }

    let activated = get_activated_vectors(mlp, input);
    let mut err_signal = mlp.compute(input) - answer;

    for layer_index in (0..mlp.num_layers() - 1).rev() {
        let backpropagated_err_signal = if layer_index > 0 {
            Some(_get_backpropagated_err_signal(
                &mlp.get_weights()[layer_index],
                &err_signal,
                &activated[layer_index],
            ))
        } else {
            None
        };

        _update_layer(
            mlp,
            layer_index,
            &err_signal,
            &activated[layer_index],
            learning_rate,
        )?;

        if let Some(backpropagated_err_signal) = backpropagated_err_signal {
            err_signal = backpropagated_err_signal;
        }
    }

    return Ok(());
}

#[cfg(test)]
mod tests
{
    use super::learn;
    use crate::mlp::MultiLayerPerceptron;
    use ndarray::{ array, Array1 as Vector };

    #[test]
    fn learn_updates_a_network_without_hidden_layers()
    {
        let mut mlp = MultiLayerPerceptron::from_data(
            vec![2, 2],
            vec![array![[0.0, 0.0], [0.0, 0.0]]],
            vec![array![0.0, 0.0]],
        ).unwrap();

        learn(&mut mlp, &array![1.0, 2.0], &array![1.0, 0.0], 0.1).unwrap();

        assert_eq!(mlp.get_weights()[0], array![[0.05, 0.1], [-0.05, -0.1]]);
        assert_eq!(mlp.get_biases()[0], array![-0.05, 0.05]);
    }

    #[test]
    fn learn_backpropagates_through_hidden_layers()
    {
        let mut mlp = MultiLayerPerceptron::from_data(
            vec![1, 1, 2],
            vec![array![[1.0]], array![[1.0], [0.0]]],
            vec![array![0.0], array![0.0, 0.0]],
        ).unwrap();
        let output_0 = 1.0_f32.exp() / (1.0_f32.exp() + 1.0);
        let learning_rate = 0.1;

        learn(
            &mut mlp,
            &array![1.0],
            &array![0.0, 1.0],
            learning_rate,
        ).unwrap();

        assert!((mlp.get_weights()[0][[0, 0]] - (1.0 - learning_rate * output_0)).abs() < 1e-6);
        assert!((mlp.get_biases()[0][0] - (learning_rate * output_0)).abs() < 1e-6);
        assert!((mlp.get_weights()[1][[0, 0]] - (1.0 - learning_rate * output_0)).abs() < 1e-6);
        assert!((mlp.get_weights()[1][[1, 0]] - (learning_rate * output_0)).abs() < 1e-6);
    }

    #[test]
    fn learn_rejects_input_and_answer_dimensions_that_do_not_match()
    {
        let mut mlp = MultiLayerPerceptron::new([2, 2].into_iter());
        let input: Vector<f32> = array![1.0];
        let answer: Vector<f32> = array![1.0, 0.0];

        assert!(learn(&mut mlp, &input, &answer, 0.1).is_err());
        assert!(learn(&mut mlp, &array![1.0, 2.0], &array![1.0], 0.1).is_err());
    }
}
