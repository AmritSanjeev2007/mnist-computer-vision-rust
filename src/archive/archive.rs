//! Serializes a multi-layer perceptron into a binary .nnmlp file.
/// 
/// The format of a .nnmlp file is as follows:
/// 
/// =============== METADATA ===============
/// 
/// (Arbitrary letters are used to represent the following attributes for brevity)
/// 
/// Magic Number: u32           = C
/// Number of Layers: u32       = N
/// 
/// =============== DATA ===============
/// 
/// Neurons per Layer: [u32; N] = Z
/// 
/// (Layer 0 corresponds to the input layer of the multi-layer perceptron)
/// (An index Z.0 corresponds to Z[0])
/// 
/// Weights layer 0 to 1        : [f32; Z.0 * Z.1]
/// Biases layer 0 to 1         : [f32; Z.1]
/// 
/// Weights layer 1 to 2        : [f32; Z.1 * Z.2]
/// Biases layer 1 to 2         : [f32; Z.2]
/// 
/// ...
/// 
/// Weights layer N-2 to N-1    : [f32; (N-2)*(N-1)]
/// Biases layer N-2 to N-1     : [f32; N-1]
///
/// * All data in the file is stored as big-endian.
/// * This module assumes that the MLP is small enough to be completely loaded into RAM.
///

use crate::mlp::MultiLayerPerceptron;

/// The magic number for .nnmlp files. 
pub const NNMLP_MAGIC_NUMBER: u32 = u32::from_be_bytes(*b"NMLP");
/// The size of the metadata (in bytes) of a .nnmlp file.
pub const NNMLP_METADATA_SIZE_BYTES: usize = 2 * size_of::<u32>();

/// Represents the data of the weight and bias connecting two layers.
#[derive(Clone)]
pub struct LayerConnectionData
{
    /// The weights connecting the two layers.
    pub weight  : ndarray::Array2<f32>,
    // The bias relating the two layers.
    pub bias    : ndarray::Array1<f32>
}

impl Default for LayerConnectionData
{
    /// Creates a new (defualt-initialized) layer connection data object.
    fn default() -> Self {
        return Self { 
            weight  : ndarray::Array2::default((1, 1)),
            bias    : ndarray::Array1::default(1),
        };
    }
}

/// Represents the serialized data retrieved after deserializing a .nnmlp file.
#[derive(Clone)]
pub struct SerializedData
{
    /// The magic number of the .nnmlp file.
    pub magic_number: u32,
    /// The neurons per layer in the multi-layer perceptron.
    pub neurons_per_layer: Vec<u32>,
    /// The data (weights and biases) connecting each layer.
    pub layer_connections: Vec<LayerConnectionData>,
}

impl SerializedData
{
    /// Creates a new (uninitialized) serialized data object.
    pub fn new() -> Self {
        return Self { magic_number: NNMLP_MAGIC_NUMBER, neurons_per_layer: Vec::new(), layer_connections: Vec::new() };
    }

    /// This function validates whether the structure of this serialized data object is logically correct.
    /// That is, it checks whether the number of layer connections satisfies the number of layers and
    /// whether dimensions of the layer connections (weights) truly related two layers.
    pub fn validate(&self) -> Result<(), String>
    {
        if self.magic_number != NNMLP_MAGIC_NUMBER { return Err(format!("Incorrect magic number.")); }

        if self.neurons_per_layer.len() < 2 { return Err(format!("A valid MLP requires at least 2 layers.")); }
        if self.neurons_per_layer.len() != self.layer_connections.len() + 1 {
            return Err(format!("Number of layer connections do not satisfy the number of layers."));
        }
        if self.neurons_per_layer.iter().any(|x| *x == 0) {
            return Err(format!("An MLP cannot have a layer with 0 neurons."));
        }

        for i in 0..(self.neurons_per_layer.len() - 1) {
            let rows = self.neurons_per_layer[i + 1] as usize;
            let columns = self.neurons_per_layer[i] as usize;

            if self.layer_connections[i].weight.dim() != (rows, columns) {
                return Err(format!("Layer connection {i} weights do not have the correct dimentions."));
            }
            if self.layer_connections[i].bias.len() != rows {
                return Err(format!("Layer connection {i} biases do not have the correct dimentions."));
            }
        }

        return Ok(());
    }

    /// Creates a serialized data object from a multi-layer perceptron.
    pub fn from_mlp(mlp: &MultiLayerPerceptron) -> Result<Self, String>
    {
        mlp.validate()?;

        let layer_connections: Vec<LayerConnectionData> = (0..mlp.num_layers() - 1).map(|i| LayerConnectionData{
            weight: mlp.get_weights()[i].clone(),
            bias: mlp.get_biases()[i].clone()
        }).collect();

        return Ok(Self {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: mlp.get_neurons_per_layer().iter().map(|x| *x as u32).collect(),
            layer_connections: layer_connections
        });
    }

    /// Tries to return a multi-layer perceptron using the serialized data.
    pub fn to_mlp(&self) -> Result<MultiLayerPerceptron, String>
    {
        use ndarray::{Array1, Array2};

        let neurons_per_layer: Vec<usize> = self.neurons_per_layer.iter().map(|x| *x as usize).collect();
        let (weights, biases): (Vec<Array2<f32>>, Vec<Array1<f32>>) = 
            self.layer_connections.iter().map(|x| (x.weight.clone(), x.bias.clone())).collect();

        return Ok(MultiLayerPerceptron::from_data(neurons_per_layer, weights, biases)?);
    }

    /// Returns the total size of this object if it were to be serialized into a bytes array. It is just the
    /// total size of its data on the stack and the heap.
    pub fn total_size_bytes(&self) -> Result<usize, String>
    {
        let metadata_len = NNMLP_METADATA_SIZE_BYTES;
        let neurons_per_layer_len = self.neurons_per_layer.len().checked_mul(size_of::<u32>());
        if neurons_per_layer_len.is_none() { return Err(format!("Too many layers.")); }

        let data_len = self.layer_connections.iter()
            .fold(0usize, |tot, x| tot + x.weight.len() + x.bias.len())
            .checked_mul(size_of::<f32>());
        if data_len.is_none() { return Err(format!("MLP parameters (weights and biases) are too large.")); }

        return Ok(metadata_len + neurons_per_layer_len.unwrap() + data_len.unwrap());
    }

}

impl Default for SerializedData
{
    /// Creates a new (defualt-initialized) serialized data object.
    fn default() -> Self {
        return Self { magic_number: NNMLP_MAGIC_NUMBER, neurons_per_layer: Vec::default(), layer_connections: Vec::default() };
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn new_creates_empty_data_with_magic_number() {
        let data = SerializedData::new();

        assert_eq!(data.magic_number, NNMLP_MAGIC_NUMBER);
        assert!(data.neurons_per_layer.is_empty());
        assert!(data.layer_connections.is_empty());
    }

    #[test]
    fn from_mlp_copies_network_structure_and_parameters() {
        let mut mlp = MultiLayerPerceptron::new([2, 3, 1].into_iter());

        mlp.get_weight_indexed_iter_mut(0).unwrap().for_each(|((row, column), x)| *x = (row * 2 + column + 1) as f32);
        mlp.get_bias_indexed_iter_mut(0).unwrap().for_each(|(index, x)| *x = (index + 7) as f32);
        mlp.get_weight_indexed_iter_mut(1).unwrap().for_each(|((row, column), x)| *x = (row * 3 + column + 10) as f32);
        mlp.get_bias_indexed_iter_mut(1).unwrap().for_each(|(index, x)| *x = (index + 13) as f32);

        let data = SerializedData::from_mlp(&mlp).unwrap();

        assert_eq!(data.magic_number, NNMLP_MAGIC_NUMBER);
        assert_eq!(data.neurons_per_layer, vec![2, 3, 1]);
        assert_eq!(data.layer_connections.len(), 2);
        assert_eq!(data.layer_connections[0].weight, ndarray::array![[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!(data.layer_connections[0].bias, ndarray::array![7., 8., 9.]);
        assert_eq!(data.layer_connections[1].weight, ndarray::array![[10., 11., 12.]]);
        assert_eq!(data.layer_connections[1].bias, ndarray::array![13.]);
    }

    #[test]
    fn to_mlp_reconstructs_network_structure_and_parameters() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 3, 1],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::array![[1., 2.], [3., 4.], [5., 6.]],
                    bias: ndarray::array![7., 8., 9.],
                },
                LayerConnectionData {
                    weight: ndarray::array![[10., 11., 12.]],
                    bias: ndarray::array![13.],
                },
            ],
        };

        let mlp = data.to_mlp().unwrap();

        assert_eq!(mlp.get_neurons_per_layer(), &vec![2, 3, 1]);
        assert_eq!(mlp.get_weights()[0], ndarray::array![[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!(mlp.get_biases()[0], ndarray::array![7., 8., 9.]);
        assert_eq!(mlp.get_weights()[1], ndarray::array![[10., 11., 12.]]);
        assert_eq!(mlp.get_biases()[1], ndarray::array![13.]);
    }

    #[test]
    fn total_size_bytes_counts_metadata_structure_and_parameters() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 3, 1],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::Array2::zeros((3, 2)),
                    bias: ndarray::Array1::zeros(3),
                },
                LayerConnectionData {
                    weight: ndarray::Array2::zeros((1, 3)),
                    bias: ndarray::Array1::zeros(1),
                },
            ],
        };

        assert_eq!(data.total_size_bytes().unwrap(), 72);
    }

    #[test]
    fn to_mlp_rejects_parameters_with_wrong_dimensions() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 3],
            layer_connections: vec![LayerConnectionData {
                weight: ndarray::Array2::zeros((1, 2)),
                bias: ndarray::Array1::zeros(3),
            }],
        };

        assert!(data.to_mlp().is_err());
    }

    #[test]
    fn validate_accepts_valid_serialized_data() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 3, 1],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::array![[1., 2.], [3., 4.], [5., 6.]],
                    bias: ndarray::array![7., 8., 9.],
                },
                LayerConnectionData {
                    weight: ndarray::array![[10., 11., 12.]],
                    bias: ndarray::array![13.],
                },
            ],
        };

        assert!(data.validate().is_ok());
    }

    #[test]
    fn validate_rejects_incorrect_magic_number() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER + 1,
            neurons_per_layer: vec![2, 3, 1],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::array![[1., 2.], [3., 4.], [5., 6.]],
                    bias: ndarray::array![7., 8., 9.],
                },
                LayerConnectionData {
                    weight: ndarray::array![[10., 11., 12.]],
                    bias: ndarray::array![13.],
                },
            ],
        };

        assert!(data.validate().is_err());
    }

    #[test]
    fn validate_rejects_fewer_than_two_layers() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2],
            layer_connections: vec![],
        };

        assert!(data.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_neuron_layers() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 0, 1],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::Array2::zeros((0, 2)),
                    bias: ndarray::Array1::zeros(0),
                },
                LayerConnectionData {
                    weight: ndarray::Array2::zeros((1, 0)),
                    bias: ndarray::Array1::zeros(1),
                },
            ],
        };

        assert!(data.validate().is_err());
    }

    #[test]
    fn validate_rejects_connection_count_mismatch() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 3, 1, 4],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::array![[1., 2.], [3., 4.], [5., 6.]],
                    bias: ndarray::array![7., 8., 9.],
                },
                LayerConnectionData {
                    weight: ndarray::array![[10., 11., 12.]],
                    bias: ndarray::array![13.],
                },
            ],
        };

        assert!(data.validate().is_err());
    }

    #[test]
    fn validate_rejects_weight_dimension_mismatch() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 3, 1],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::array![[1., 2.], [3., 4.]],
                    bias: ndarray::array![7., 8., 9.],
                },
                LayerConnectionData {
                    weight: ndarray::array![[10., 11., 12.]],
                    bias: ndarray::array![13.],
                },
            ],
        };

        assert!(data.validate().is_err());
    }

    #[test]
    fn validate_rejects_bias_dimension_mismatch() {
        let data = SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 3, 1],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::array![[1., 2.], [3., 4.], [5., 6.]],
                    bias: ndarray::array![7., 8.],
                },
                LayerConnectionData {
                    weight: ndarray::array![[10., 11., 12.]],
                    bias: ndarray::array![13.],
                },
            ],
        };

        assert!(data.validate().is_err());
    }
}