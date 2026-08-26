use crate::archive::archive::{LayerConnectionData, SerializedData};

/// Represents a generic object which can deserialize a (binary) file into a object which is convertible to a multi-layer perceptron.
pub trait Deserializer
{
    /// Tries to deserialize bytes into an object which is convertible to a multi-layer perceptron.
    fn deserialize_bytes(&mut self, bytes: &[u8]) -> Result<(), String>;
    /// Tries to deserialize a (binary) file into a object which is convertible to a multi-layer perceptron.
    fn deserialize(&mut self, file: &std::path::Path) -> Result<(), String>;
}

fn _deserialize_metadata<Iter: ExactSizeIterator<Item = u32>>(this: &mut SerializedData, u32_iter: &mut Iter) -> Result<(), String> {
    use crate::archive::archive::{NNMLP_MAGIC_NUMBER, NNMLP_METADATA_SIZE_BYTES};
    
    if u32_iter.len() < (NNMLP_METADATA_SIZE_BYTES/size_of::<u32>()) { return Err(format!("Not enough bytes for loading metadata.")); }

    this.magic_number = u32_iter.next().unwrap();
    if this.magic_number != NNMLP_MAGIC_NUMBER { return Err(format!("Magic number of file is not {NNMLP_MAGIC_NUMBER}.")); }

    let number_of_layers = u32_iter.next().unwrap() as usize;
    if number_of_layers < 2 { return Err(format!("MLP must contain at least 2 layers.")); }
    if u32_iter.len() < number_of_layers { return Err(format!("Not enough bytes for loading neurons per layer.")); }

    this.neurons_per_layer = (0..number_of_layers).map(|_| u32_iter.next().unwrap()).collect();
    if this.neurons_per_layer.iter().any(|x| *x == 0) { return Err(format!("MLP layers must have at least one neuron.")); }

    this.layer_connections = Vec::with_capacity(number_of_layers - 1);

    return Ok(());
}

fn _deserialize_layer_connections<Iter>(this: &mut SerializedData, u32_iter: &mut Iter) -> Result<(), String>
where Iter: ExactSizeIterator<Item = u32>
{
    let number_of_connections = this.neurons_per_layer.len() - 1;

    for i in 0..number_of_connections {
        let rows = this.neurons_per_layer[i + 1] as usize;
        let columns = this.neurons_per_layer[i] as usize;

        let required_len = rows.checked_mul(columns).and_then(|x| x.checked_add(rows)).ok_or_else(|| format!("Layer connection {i} is too large."))?;
        if u32_iter.len() < required_len { return Err(format!("Iterator doesn't have sufficient size for layer connection {i}.")); }

        let weights = ndarray::Array2::from_shape_simple_fn((rows, columns), || f32::from_bits(u32_iter.next().unwrap()));
        let biases = ndarray::Array1::from_shape_simple_fn(rows, || f32::from_bits(u32_iter.next().unwrap()));

        this.layer_connections.push(LayerConnectionData { weight: weights, bias: biases });
    }

    return Ok(());
}

impl Deserializer for SerializedData {

    /// Deserializes the bytes of a .nnmlp file.
    fn deserialize_bytes(&mut self, bytes: &[u8]) -> Result<(), String> 
    {
        if bytes.len() & 0b11 != 0 { return Err(format!("Size of bytes slice must be a multiple of 4 (= sizeof u32).")); }

        let mut u32_iter = bytes.chunks_exact(4).map(|x| u32::from_be_bytes(x.try_into().unwrap()));
        let mut tmp_serialized_data = SerializedData::default();

        _deserialize_metadata(&mut tmp_serialized_data, &mut u32_iter)?;
        _deserialize_layer_connections(&mut tmp_serialized_data, &mut u32_iter)?;

        if u32_iter.len() > 0 { return Err(format!("Bytes slice contains trailing data.")); }

        *self = tmp_serialized_data;

        return Ok(());
    }

    /// Deserializes a .nnmlp file.
    fn deserialize(&mut self, file: &std::path::Path) -> Result<(), String> {
        let bytes_vec = std::fs::read(file);
        if let Err(err) = bytes_vec { return Err(format!("Failed to read from file: {:?} due to error: {err}.", file)); }

        return self.deserialize_bytes(bytes_vec.unwrap().as_slice());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::archive::NNMLP_MAGIC_NUMBER;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn bytes(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
        values.into_iter().flat_map(u32::to_be_bytes).collect()
    }

    fn valid_bytes() -> Vec<u8> {
        bytes([
            NNMLP_MAGIC_NUMBER, 3, 2, 3, 1,
            1.0f32.to_bits(), 2.0f32.to_bits(),
            3.0f32.to_bits(), 4.0f32.to_bits(),
            5.0f32.to_bits(), 6.0f32.to_bits(),
            7.0f32.to_bits(), 8.0f32.to_bits(), 9.0f32.to_bits(),
            10.0f32.to_bits(), 11.0f32.to_bits(), 12.0f32.to_bits(),
            13.0f32.to_bits(),
        ])
    }

    fn temporary_path(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        std::env::temp_dir().join(format!(
            "deserializer_{}_{}_{}",
            name,
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn deserialize_bytes_reads_metadata_weights_and_biases() {
        let mut data = SerializedData::new();

        data.deserialize_bytes(&valid_bytes()).unwrap();

        assert_eq!(data.magic_number, NNMLP_MAGIC_NUMBER);
        assert_eq!(data.neurons_per_layer, vec![2, 3, 1]);
        assert_eq!(data.layer_connections.len(), 2);
        assert_eq!(data.layer_connections[0].weight, ndarray::array![[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!(data.layer_connections[0].bias, ndarray::array![7., 8., 9.]);
        assert_eq!(data.layer_connections[1].weight, ndarray::array![[10., 11., 12.]]);
        assert_eq!(data.layer_connections[1].bias, ndarray::array![13.]);
    }

    #[test]
    fn deserialize_bytes_rejects_invalid_metadata() {
        let cases = [
            (bytes([0]), "Not enough bytes"),
            (bytes([0, 2, 1, 1]), "Magic number"),
            (bytes([NNMLP_MAGIC_NUMBER, 1]), "at least 2 layers"),
            (bytes([NNMLP_MAGIC_NUMBER, 2, 0, 1]), "at least one neuron"),
            (bytes([NNMLP_MAGIC_NUMBER, 3, 1, 2]), "neurons per layer"),
        ];

        for (input, message) in cases {
            let mut data = SerializedData::new();
            let error = data.deserialize_bytes(&input).unwrap_err();
            assert!(error.contains(message), "unexpected error: {error}");
        }
    }

    #[test]
    fn deserialize_bytes_rejects_malformed_length_and_trailing_data() {
        let mut data = SerializedData::new();

        let error = data.deserialize_bytes(&[0, 0, 0]).unwrap_err();
        assert!(error.contains("multiple of 4"));

        let mut incomplete = valid_bytes();
        incomplete.pop();
        incomplete.truncate(incomplete.len() - 3);
        let error = data.deserialize_bytes(&incomplete).unwrap_err();
        assert!(error.contains("sufficient size"));

        let mut trailing = valid_bytes();
        trailing.extend_from_slice(&0u32.to_be_bytes());
        let error = data.deserialize_bytes(&trailing).unwrap_err();
        assert!(error.contains("trailing data"));
    }

    #[test]
    fn deserialize_bytes_does_not_replace_receiver_on_error() {
        let mut data = SerializedData::new();
        data.neurons_per_layer = vec![9, 8];
        let original = data.neurons_per_layer.clone();

        let mut invalid = valid_bytes();
        invalid.extend_from_slice(&0u32.to_be_bytes());
        assert!(data.deserialize_bytes(&invalid).is_err());

        assert_eq!(data.neurons_per_layer, original);
        assert!(data.layer_connections.is_empty());
    }

    #[test]
    fn deserialize_reads_file_and_reports_file_errors() {
        let path = temporary_path("valid");
        std::fs::write(&path, valid_bytes()).unwrap();

        let mut data = SerializedData::new();
        data.deserialize(&path).unwrap();
        assert_eq!(data.neurons_per_layer, vec![2, 3, 1]);
        std::fs::remove_file(&path).unwrap();

        let missing = temporary_path("missing");
        let error = data.deserialize(&missing).unwrap_err();
        assert!(error.contains("Failed to read"));
    }
}

