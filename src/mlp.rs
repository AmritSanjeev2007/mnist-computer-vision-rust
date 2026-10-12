/// Given an input vector, `x`, this function returns a new vector `relu(x)`.
pub fn relu(x: &ndarray::Array1<f32>) -> ndarray::Array1<f32>
{
    fn _relu_single(n: &mut f32) -> () { *n = n.max(0.0); }

    let mut x_copy = x.clone();
    x_copy.iter_mut().for_each(_relu_single);

    return x_copy;
}

/// Given an input vector, `x`, this function returns a new vector `softmax(x)`.
pub fn softmax(x: &ndarray::Array1<f32>) -> ndarray::Array1<f32>
{
    fn _exp_accumulator(total: f32, x: &f32) -> f32 { return total + x.exp(); }

    let sum: f32 = x.fold(0.0, _exp_accumulator);

    let mut x_copy = x.clone();
    x_copy.iter_mut().for_each(move |n| *n = n.exp()/sum);
    return x_copy;
}

/// A simple multi-layer perceptron (MLP) with a customizable structure (neurons per layer).
#[derive(Debug)]
pub struct MultiLayerPerceptron
{
    _neurons_per_layer  : Vec<usize>,
    _weights            : Vec<ndarray::Array2<f32>>,
    _biases             : Vec<ndarray::Array1<f32>>,
}

impl MultiLayerPerceptron
{
    /// Constructs a new (uninitialized) MultiLayerPerceptron using the specified neurons per layer (via a iterator).
    pub fn new<T: Iterator<Item = usize>>(neurons: T) -> Self
    {
        let neurons_per_layer: Vec<usize>           = Vec::from_iter(neurons);
        let mut weights: Vec<ndarray::Array2<f32>>  = Vec::with_capacity(neurons_per_layer.len() - 1);
        let mut biases: Vec<ndarray::Array1<f32>>   = Vec::with_capacity(neurons_per_layer.len() - 1);

        for i in 0..(neurons_per_layer.len() - 1)
        {
            let columns: usize  = neurons_per_layer[i];
            let rows: usize     = neurons_per_layer[i + 1];

            weights.push(ndarray::Array2::from_elem((rows, columns), 0.));
            biases.push(ndarray::Array1::from_elem(rows, 0.));
        }

        return Self {
            _neurons_per_layer  : neurons_per_layer,
            _weights            : weights,
            _biases             : biases
        };
    }

    /// This function validates whether the structure of this MLP is logically correct.
    /// That is, it checks whether the number of layer connections satisfies the number of layers and
    /// whether dimensions of the layer connections (weights) truly related two layers.
    pub fn validate(&self) -> Result<(), String>
    {
        if self._neurons_per_layer.len() < 2 { return Err(format!("A valid MLP requires at least 2 layers.")); }
        if self._weights.len() != self._biases.len() { return Err(format!("Count of Weights and Biases must be the same.")); }
        if self._neurons_per_layer.len() != self._weights.len() + 1 {
            return Err(format!("Number of layer connections do not satisfy the number of layers."));
        }
        if self._neurons_per_layer.iter().any(|x| *x == 0) {
            return Err(format!("An MLP cannot have a layer with 0 neurons."));
        }

        for i in 0..(self._neurons_per_layer.len() - 1) {
            let rows = self._neurons_per_layer[i + 1];
            let columns = self._neurons_per_layer[i];

            if self._weights[i].dim() != (rows, columns) {
                return Err(format!("Layer connection {i} weights do not have the correct dimentions."));
            }
            if self._biases[i].len() != rows {
                return Err(format!("Layer connection {i} biases do not have the correct dimentions."));
            }
        }

        return Ok(());
    } 

    /// Tries to create a new multi-layer perceptron from its struct data members.
    /// Note that this function validates whether the structure of the specified data is logically correct.
    pub fn from_data(neurons_per_layer: Vec<usize>, weights: Vec<ndarray::Array2<f32>>, biases: Vec<ndarray::Array1<f32>>) -> Result<Self, String> 
    {
        let mlp = Self { _neurons_per_layer: neurons_per_layer, _weights: weights, _biases: biases };
        mlp.validate()?;

        return Ok(mlp);
    }

    /// Returns the number of layers in the multi-layer perceptron.
    pub fn num_layers(&self) -> usize { return self._neurons_per_layer.len(); }

    /// Retrieves the neurons per layer of this multi-layer perceptron.
    pub fn get_neurons_per_layer(&self) -> &Vec<usize> {
        return &self._neurons_per_layer;
    }

    /// Returns whether this MLP is "structurally similar" to another MLP. Two MLPs are considered similar if they have the same number of layers
    /// and the same number of neurons on each corresponding layer.
    pub fn is_similar_to(&self, other: &Self) -> bool
    {
        if self.num_layers() != other.num_layers() { return false; } 
        return self._neurons_per_layer.eq(&other._neurons_per_layer);
    }

    /// Performs a Xavier initialization on the weight matrices such that every entry is a random number 
    /// which follows a normal distribution with a mean of 0 and a standard deviation depending on the layer.
    pub fn xavier_initialization(&mut self) -> ()
    {
        use rand_distr::Distribution;

        static MEAN: f32 = 0.;

        let mut rng: rand::prelude::ThreadRng = rand::rng();

        for i in 0..self._weights.len()
        {
            let stddev: f32 = (2./((self._neurons_per_layer[i] as f32) + (self._neurons_per_layer[i + 1] as f32))).sqrt();
            let normal = rand_distr::Normal::new(MEAN, stddev);
            
            if normal.is_err() { return println!("Failed to create normal distribution due to error: {:?}.", normal.err()); }

            self._weights[i].iter_mut().for_each(|x| *x = normal.unwrap().sample(&mut rng));
        }
    }

    /// Returns an immutable slice to the weights in this multi-layer perceptron.
    pub fn get_weights(&self) -> &[ndarray::Array2<f32>] { return &self._weights; }
    /// Returns an immutable slice to the biases in this multi-layer perceptron.
    pub fn get_biases(&self) -> &[ndarray::Array1<f32>] { return &self._biases; }

    /// Returns an iterator that yields mutable references to the entries for the weight matrix in the `index`-th layer.
    pub fn get_weight_iter_mut(&mut self, index: usize) -> Result<impl Iterator<Item = &mut f32>, String>
    {
        if index >= self._weights.len() { 
            return Err(format!("Error: Out of bounds (max: {}), cannot get weight matrix in the {index}-th layer", self._weights.len()));
        }

        return Ok(self._weights[index].iter_mut());
    }
    /// Returns an iterator that yields mutable references to the entries for the bias vector in the `index`-th layer.
    pub fn get_bias_iter_mut(&mut self, index: usize) -> Result<impl Iterator<Item = &mut f32>, String>
    {
        if index >= self._biases.len() { 
            return Err(format!("Error: Out of bounds (max: {}), cannot get bias vector in the {index}-th layer", self._weights.len()));
        }

        return Ok(self._biases[index].iter_mut());
    }

    /// Returns an iterator that yields mutable references and its indices to the entries for the weight matrix in
    /// the `index`-th layer.
    pub fn get_weight_indexed_iter_mut(&mut self, index: usize) -> Result<impl Iterator<Item = ((usize, usize), &mut f32)>, String>
    {
        if index >= self._weights.len() { 
            return Err(format!("Error: Out of bounds (max: {}), cannot get weight matrix in the {index}-th layer", self._weights.len()));
        }

        return Ok(self._weights[index].indexed_iter_mut());
    }
    /// Returns an iterator that yields mutable references and its indices to the entries for the bias vector in
    /// the `index`-th layer.
    pub fn get_bias_indexed_iter_mut(&mut self, index: usize) -> Result<impl Iterator<Item = (usize, &mut f32)>, String>
    {
        if index >= self._biases.len() { 
            return Err(format!("Error: Out of bounds (max: {}), cannot get bias vector in the {index}-th layer", self._weights.len()));
        }

        return Ok(self._biases[index].indexed_iter_mut());
    }

    /// Computes the result of this multi-layer perceptron for the specified `input`.
    pub fn compute(&self, input: &ndarray::Array1<f32>) -> ndarray::Array1<f32>
    {
        debug_assert!(self._neurons_per_layer.len() >= 2, "Number of layers must be at least 2.");
        debug_assert_eq!(input.len(), self._neurons_per_layer[0], "Input vector size {} must match the number of neurons in the input layer {}.", input.len(), self._neurons_per_layer.len());

        let mut output: ndarray::Array1<f32> = input.clone();
        for i in 0..(self.num_layers() - 2)
        {
            let tmp: ndarray::Array1<f32> = self._weights[i].dot(&output) - &self._biases[i];
            output = relu(&tmp);
        }

        let tmp: ndarray::Array1<f32> = self._weights.last().unwrap().dot(&output) - self._biases.last().unwrap();
        output = softmax(&tmp);

        return output;
    }

}

impl Clone for MultiLayerPerceptron
{
    /// Clones this multi-layer perceptron. Note: MLPs are large objects, cloning will most likely incur a heavy cost.
    fn clone(&self) -> MultiLayerPerceptron
    {
        return MultiLayerPerceptron {
            _neurons_per_layer  : self._neurons_per_layer.clone(),
            _weights            : self._weights.clone(),
            _biases             : self._biases.clone()
        };
    }
}

impl Default for MultiLayerPerceptron
{
    /// Creates a new (default-initialized) multi-layer perceptron.
    fn default() -> Self {
        return MultiLayerPerceptron {
            _neurons_per_layer  : Vec::default(),
            _weights            : Vec::default(),
            _biases             : Vec::default()
        };
    }
}

impl PartialEq for MultiLayerPerceptron
{
    /// Returns true if two multi-layer perceptrons are equal.
    fn eq(&self, other: &Self) -> bool {
        return  self.is_similar_to(other) &&
                self._weights == other._weights &&
                self._biases == other._biases
    }
}


#[cfg(test)]
mod tests
{
    use super::*;

    fn _is_roughly_equal(a: f32, b: f32, acceptable_error: f32) -> bool {
        return (a-b).abs() < acceptable_error;
    }


    #[test]
    fn test_relu_calculation() {
        let vec1: ndarray::Array1<f32> = ndarray::Array1::from_iter(vec![1., 5., 7., 6.].into_iter());
        assert_eq!(relu(&vec1), ndarray::Array1::from_iter(vec![1., 5., 7., 6.].into_iter()));

        let vec2: ndarray::Array1<f32> = ndarray::Array1::from_iter(vec![1., -5., -7., 6.].into_iter());
        assert_eq!(relu(&vec2), ndarray::Array1::from_iter(vec![1., 0., 0., 6.].into_iter()));
    }

    #[test]
    fn test_softmax_calculation() {
        let vec1: ndarray::Array1<f32> = ndarray::Array1::from_iter(vec![0.4, 0.5, -0.2, 0.4].into_iter());

        let output1 = softmax(&vec1);
        let answer1 = ndarray::Array1::from_iter(vec![0.274, 0.302, 0.150, 0.274].into_iter());

        assert!(output1.into_iter().zip(answer1.into_iter()).all(|x| _is_roughly_equal(x.0, x.1, 0.002)));

        let vec2: ndarray::Array1<f32> = ndarray::Array1::from_iter(vec![-0.25, 1.45, -78.2, 0.06].into_iter());

        let output2 = softmax(&vec2);
        let answer2 = ndarray::Array1::from_iter(vec![0.128, 0.698, 0., 0.174].into_iter());

        assert!(output2.into_iter().zip(answer2.into_iter()).all(|x| _is_roughly_equal(x.0, x.1, 0.002)));
    }

    #[test]
    fn test_mlp_instantiation_and_getters() {
        let mlp = MultiLayerPerceptron::new(vec![24, 5, 46].into_iter());

        assert_eq!(mlp.num_layers(), 3);
        assert_eq!(*mlp.get_neurons_per_layer().get(2).unwrap(), 46);
        assert!(mlp.get_neurons_per_layer().get(25).is_none());

        assert_eq!(mlp.get_weights()[0].len(), 24 * 5);
        assert_eq!(mlp.get_weights()[1].dim(), (46, 5));
        assert!(mlp.get_weights().get(99).is_none());

        assert_eq!(mlp.get_biases()[0].dim(), 5);
        assert_eq!(mlp.get_biases()[1].dim(), 46);
        assert!(mlp.get_biases().get(1289).is_none());
    }

    #[test]
    fn test_similarity_checks_layer_shape() {
        let a1 = MultiLayerPerceptron::new(vec![2, 4, 1].into_iter());
        let b1 = MultiLayerPerceptron::new(vec![2, 4, 1].into_iter());

        assert!(a1.is_similar_to(&b1));

        let a2 = MultiLayerPerceptron::new(vec![2, 4, 1].into_iter());
        let b2 = MultiLayerPerceptron::new(vec![2, 20, 1].into_iter());

        assert!(!a2.is_similar_to(&b2));
    }

    #[test]
    fn test_weight_and_bias_mut_iterators() {
        let mut mlp = MultiLayerPerceptron::new(vec![12, 64, 198].into_iter());

        mlp.get_weight_indexed_iter_mut(1).unwrap().for_each(|((r,c),x)| *x = (r * c) as f32);

        assert_eq!(*mlp.get_weights()[1].get((54, 9)).unwrap(), 54. * 9.);
        assert_ne!(*mlp.get_weights()[1].get((24, 11)).unwrap(), 24. * 10.);

        mlp.get_bias_indexed_iter_mut(0).unwrap().for_each(|(i,x)| *x = (i as f32).exp());

        assert!(_is_roughly_equal(*mlp.get_biases()[0].get(49).unwrap(), 49_f32.exp(), 0.002));
        assert!(_is_roughly_equal(*mlp.get_biases()[0].get(7).unwrap(), 7_f32.exp(), 0.002));

        assert!(mlp.get_weight_iter_mut(2).is_err());
        assert!(mlp.get_bias_iter_mut(2).is_err());
    }

    #[test]
    fn test_mlp_computation() {
        let mut mlp = MultiLayerPerceptron::new(vec![2, 2, 2].into_iter());

        mlp.get_weight_indexed_iter_mut(0).unwrap().for_each(|((row, column), value)| {
            *value = match (row, column) {
                (0, 0) => 1.,
                (0, 1) => -1.,
                (1, 0) => 0.5,
                (1, 1) => 0.5,
                _ => unreachable!(),
            };
        });
        mlp.get_bias_indexed_iter_mut(0).unwrap().for_each(|(row, value)| {
            *value = if row == 0 { 0. } else { 1. };
        });
        mlp.get_weight_indexed_iter_mut(1).unwrap().for_each(|((row, _column), value)| {
            *value = if row == 0 { 1. } else { -1. };
        });
        mlp.get_bias_iter_mut(1).unwrap().for_each(|value| *value = 0.);

        let output = mlp.compute(&ndarray::Array1::from_vec(vec![3., 1.]));
        let expected = ndarray::Array1::from_vec(vec![0.9975, 0.0025]);

        assert!(_is_roughly_equal(output[0], expected[0], 0.001));
        assert!(_is_roughly_equal(output[1], expected[1], 0.001));
        assert!(_is_roughly_equal(output.sum(), 1., 0.001));
    }

    #[test]
    fn test_from_data_accepts_matching_shapes() {
        let weights = vec![ndarray::Array2::from_elem((2, 3), 1.)];
        let biases = vec![ndarray::Array1::from_vec(vec![0.5, -0.5])];

        let mlp = MultiLayerPerceptron::from_data(vec![3, 2], weights.clone(), biases.clone()).unwrap();

        assert_eq!(mlp.get_neurons_per_layer(), &vec![3, 2]);
        assert_eq!(mlp.get_weights()[0], weights[0]);
        assert_eq!(mlp.get_biases()[0], biases[0]);
    }

    #[test]
    fn test_from_data_rejects_mismatched_shapes() {
        let valid_weights = vec![ndarray::Array2::from_elem((2, 3), 1.)];
        let valid_biases = vec![ndarray::Array1::from_elem(2, 0.)];

        assert!(MultiLayerPerceptron::from_data(
            vec![3, 2],
            vec![ndarray::Array2::from_elem((1, 3), 1.)],
            valid_biases.clone(),
        ).is_err());
        assert!(MultiLayerPerceptron::from_data(
            vec![3, 2],
            valid_weights,
            vec![ndarray::Array1::from_elem(1, 0.)],
        ).is_err());
    }

    #[test]
    fn test_clone_copies_network_data() {
        let mut original = MultiLayerPerceptron::new(vec![2, 2].into_iter());
        original.get_weight_iter_mut(0).unwrap().for_each(|value| *value = 3.);
        original.get_bias_iter_mut(0).unwrap().for_each(|value| *value = -2.);

        let mut clone = original.clone();
        clone.get_weight_iter_mut(0).unwrap().for_each(|value| *value = 9.);
        clone.get_bias_iter_mut(0).unwrap().for_each(|value| *value = 7.);

        assert_eq!(*original.get_weights()[0].get((0, 0)).unwrap(), 3.);
        assert_eq!(*original.get_biases()[0].get(0).unwrap(), -2.);
        assert_eq!(*clone.get_weights()[0].get((0, 0)).unwrap(), 9.);
        assert_eq!(*clone.get_biases()[0].get(0).unwrap(), 7.);
    }
}