/// Given an input vector, `x`, this function returns a new vector `relu(x)`.
fn relu(x: &ndarray::Array1<f32>) -> ndarray::Array1<f32>
{
    fn _relu_single(n: &mut f32) -> () { *n = n.max(0.0); }

    let mut x_copy = x.clone();
    x_copy.iter_mut().for_each(_relu_single);

    return x_copy;
}

/// Given an input vector, `x`, this function returns a new vector `softmax(x)`.
fn softmax(x: &ndarray::Array1<f32>) -> ndarray::Array1<f32>
{
    fn _exp_accumulator(total: f32, x: &f32) -> f32 { return total + x.exp(); }

    let sum: f32 = x.fold(0.0, _exp_accumulator);

    let mut x_copy = x.clone();
    x_copy.iter_mut().for_each(move |n| *n = n.exp()/sum);
    return x_copy;
}

/// A simple multi-layer perceptron (MLP) with a customizable structure (neurons per layer).
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

    /// Returns the number of layers in the multi-layer perceptron.
    pub fn num_layers(&self) -> usize { return self._neurons_per_layer.len(); }

    /// Tries to retrieve the number of neurons in the queried `layer`. Note: `layer := 0` corresponds to the input/first layer.
    pub fn neurons_per_layer(&self, layer: usize) -> Result<usize, String>
    {
        if layer >= self._neurons_per_layer.len() 
        { 
            return Err(format!("Error: Out of bounds (max: {}), cannot get neurons in the {}-th layer", self._neurons_per_layer.len(), layer)); 
        }
        return Ok(*self._neurons_per_layer.get(layer).unwrap()); 
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

    /// Tries to return a const reference to the weights at the specified index. Note: an index of 0 corresponds 
    /// to the weight matrix relating the input/first layer and the next layer.
    pub fn get_weight_matrix(&self, index: usize) -> Result<&ndarray::Array2<f32>, String>
    {
        if index >= self._weights.len()
        { 
            return Err(format!("Error: Out of bounds (max: {}), cannot get weight matrix in the {}-th layer", self._weights.len(), index));
        }

        return Ok(self._weights.get(index).unwrap());
    }
    /// Tries to return a const reference to the bias vector at the specified index. Note: an index of 0 corresponds 
    /// to the bias vector relating the input/first layer and the next layer.
    pub fn get_bias_vector(&self, index: usize) -> Result<&ndarray::Array1<f32>, String>
    {
        if index >= self._biases.len()
        { 
            return Err(format!("Error: Out of bounds (max: {}), cannot get bias vector in the {}-th layer", self._biases.len(), index));
        }

        return Ok(self._biases.get(index).unwrap());
    }

    /// Tries to transform the element of the weight matrix at the specified index using the specified callback. 
    /// Note: an index of 0 corresponds to the weight matrix relating the input/first layer and the next layer.
    pub fn set_weight_matrix<Func>(&mut self, index: usize, setter: Func) -> Result<(), String>
    where
        Func: FnMut(((usize, usize), &mut f32))
    {
        if index >= self._weights.len() { 
            return Err(format!("Error: Out of bounds (max: {}), cannot set weight matrix in the {}-th layer", self._weights.len(), index));
        }

        self._weights[index].indexed_iter_mut().for_each(setter);
        return Ok(());
    }

    /// Tries to transform the element of the bias vector at the specified index using the specified callback. 
    /// Note: an index of 0 corresponds to the bias vector relating the input/first layer and the next layer.
    pub fn set_bias_vector<Func>(&mut self, index: usize, setter: Func) -> Result<(), String>
    where
        Func: FnMut((usize, &mut f32))
    {
        if index >= self._biases.len() { 
            return Err(format!("Error: Out of bounds (max: {}), cannot set bias vector in the {}-th layer", self._biases.len(), index));
        }

        self._biases[index].indexed_iter_mut().for_each(setter);
        return Ok(());
    }

    /// Computes the result of this multi-layer perceptron for the specified `input`.
    pub fn compute(&self, input: &ndarray::Array1<f32>) -> ndarray::Array1<f32>
    {
        debug_assert!(self._neurons_per_layer.len() > 0, "Number of layers must be non-zero.");
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
        assert_eq!(mlp.neurons_per_layer(2).unwrap(), 46);
        assert!(mlp.neurons_per_layer(25).is_err());

        assert_eq!(mlp.get_weight_matrix(0).unwrap().len(), 24 * 5);
        assert_eq!(mlp.get_weight_matrix(1).unwrap().dim(), (46, 5));
        assert!(mlp.get_weight_matrix(99).is_err());

        assert_eq!(mlp.get_bias_vector(0).unwrap().dim(), 5);
        assert_eq!(mlp.get_bias_vector(1).unwrap().dim(), 46);
        assert!(mlp.get_bias_vector(1289).is_err());
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
    fn test_weight_and_bias_setters() {
        let mut mlp = MultiLayerPerceptron::new(vec![12, 64, 198].into_iter());

        mlp.set_weight_matrix(1, |x| *x.1 = (x.0.0 as f32) * (x.0.1 as f32)).unwrap();

        assert_eq!(*mlp.get_weight_matrix(1).unwrap().get((54, 9)).unwrap(), 54. * 9.);
        assert_ne!(*mlp.get_weight_matrix(1).unwrap().get((24, 11)).unwrap(), 24. * 10.);

        mlp.set_bias_vector(0, |x| *x.1 = (x.0 as f32).exp()).unwrap();

        assert!(_is_roughly_equal(*mlp.get_bias_vector(0).unwrap().get(49).unwrap(), 49_f32.exp(), 0.002));
        assert!(_is_roughly_equal(*mlp.get_bias_vector(0).unwrap().get(7).unwrap(), 7_f32.exp(), 0.002));
    }

    #[test]
    fn test_mlp_computation() {
        
    }
}