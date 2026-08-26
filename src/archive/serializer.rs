use crate::archive::archive::{ SerializedData };

/// Represents a generic object which can serialize an object which is convertible to a multi-layer perceptron into a (binary) file.
pub trait Serializer
{
    /// Tries to serialize an object which is convertible to a multi-layer perceptron into a (big-endian) bytes array.
    fn serialize_to_bytes(&self) -> Result<Vec<u8>, String>;
    /// Tries to serialize an object which is convertible to a multi-layer perceptron into the specified (binary) file.
    /// The bytes are stored in a (big-endian) ordering. 
    fn serialize_to_file(&self, path: &std::path::Path) -> Result<(), String>;
}

fn _serialize_metadata(this: &SerializedData, bytes: &mut Vec<u8>) -> Result<(), String>
{
    use crate::archive::archive::{ NNMLP_METADATA_SIZE_BYTES };

    if bytes.spare_capacity_mut().len() < NNMLP_METADATA_SIZE_BYTES {
        return Err(format!("Not enough space for serializing metadata."));
    }
    bytes.extend(this.magic_number.to_be_bytes());
    bytes.extend((this.neurons_per_layer.len() as u32).to_be_bytes());

    if bytes.spare_capacity_mut().len() < this.neurons_per_layer.len() * size_of::<u32>() {
        return Err(format!("Not enough space for serializing neurons per layer."));
    }
    bytes.extend(this.neurons_per_layer.iter().map(|x| x.to_be_bytes()).flatten());

    return Ok(());
}

fn _serialize_layer_connections(this: &SerializedData, bytes: &mut Vec<u8>) -> Result<(), String>
{
    for i in 0..this.layer_connections.len() {
        if bytes.spare_capacity_mut().len() < this.layer_connections[i].weight.len() * size_of::<f32>() {
            return Err(format!("Not enough space for serializing layer connection {i} weights."));
        }
        bytes.extend(this.layer_connections[i].weight.iter().map(|x| x.to_be_bytes()).flatten());

        if bytes.spare_capacity_mut().len() < this.layer_connections[i].bias.len() * size_of::<f32>() {
            return Err(format!("Not enough space for serializing layer connection {i} biases."));
        }
        bytes.extend(this.layer_connections[i].bias.iter().map(|x| x.to_be_bytes()).flatten());
    }

    return Ok(());
}

impl Serializer for SerializedData
{
    fn serialize_to_bytes(&self) -> Result<Vec<u8>, String> 
    {
        self.validate()?;

        let size = self.total_size_bytes()?;
        if size & 0b11 != 0 { return Err(format!("Size of the serialized data object must be a multiple of 4.")); }

        let mut bytes: Vec<u8> = Vec::with_capacity(size);

        _serialize_metadata(self, &mut bytes)?;
        _serialize_layer_connections(self, &mut bytes)?;

        if bytes.len() != size { return Err(format!("Serialized data object has trailing data?!")); }

        return Ok(bytes);
    }

    fn serialize_to_file(&self, path: &std::path::Path) -> Result<(), String>
    {
        let bytes = self.serialize_to_bytes()?;
        
        if let Err(err) = std::fs::write(path, bytes) {
            return Err(format!("Writing to {path:?} failed with error: {err}."));
        }
        
        return Ok(());
    }
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::archive::archive::{LayerConnectionData, NNMLP_MAGIC_NUMBER};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn simple_network() -> SerializedData {
        SerializedData {
            magic_number: NNMLP_MAGIC_NUMBER,
            neurons_per_layer: vec![2, 2, 1],
            layer_connections: vec![
                LayerConnectionData {
                    weight: ndarray::array![[1.0, 2.0], [3.0, 4.0]],
                    bias: ndarray::array![5.0, 6.0],
                },
                LayerConnectionData {
                    weight: ndarray::array![[7.0, 8.0]],
                    bias: ndarray::array![9.0],
                },
            ],
        }
    }

    #[test]
    fn serialize_to_bytes_writes_expected_layout_for_simple_network() {
        let data = simple_network();
        let expected = [
            NNMLP_MAGIC_NUMBER.to_be_bytes(),
            3u32.to_be_bytes(),
            2u32.to_be_bytes(),
            2u32.to_be_bytes(),
            1u32.to_be_bytes(),
            1.0f32.to_be_bytes(),
            2.0f32.to_be_bytes(),
            3.0f32.to_be_bytes(),
            4.0f32.to_be_bytes(),
            5.0f32.to_be_bytes(),
            6.0f32.to_be_bytes(),
            7.0f32.to_be_bytes(),
            8.0f32.to_be_bytes(),
            9.0f32.to_be_bytes(),
        ]
        .concat();

        let bytes = data.serialize_to_bytes().unwrap();

        assert_eq!(bytes, expected);
    }

    #[test]
    fn serialize_to_file_rejects_invalid_data_before_writing() {
        let data = SerializedData {
            magic_number: 0,
            neurons_per_layer: vec![2, 2],
            layer_connections: vec![LayerConnectionData {
                weight: ndarray::array![[1.0, 2.0], [3.0, 4.0]],
                bias: ndarray::array![5.0, 6.0],
            }],
        };

        let path = std::env::temp_dir().join(format!(
            "nnmlp-invalid-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::remove_file(&path);

        assert!(data.serialize_to_bytes().is_err());
        assert!(data.serialize_to_file(&path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn serialize_to_bytes_length_matches_total_size_bytes() {
        let data = simple_network();

        let bytes = data.serialize_to_bytes().unwrap();

        assert_eq!(bytes.len(), data.total_size_bytes().unwrap());
    }

    #[test]
    fn serialize_to_file_writes_same_bytes_to_disk() {
        let data = simple_network();
        let expected = data.serialize_to_bytes().unwrap();
        let path = std::env::temp_dir().join(format!(
            "nnmlp-output-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::remove_file(&path);

        data.serialize_to_file(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes, expected);

        let _ = std::fs::remove_file(&path);
    }
}