/// Constants describing the training data.
/// See https://www.kaggle.com/code/hojjatk/read-mnist-dataset for more information on how to read the dataset.


/// The magic number for the images dataset file. (e.g., data\train-images.idx3-ubyte)
pub const IMAGE_DATASET_MAGIC_NUMBER: usize = 2051;
/// The magic number for the images dataset file. (e.g., data\train-labels.idx3-ubyte)
pub const LABEL_DATASET_MAGIC_NUMBER: usize = 2049;

/// The width (in px) of the images in the training dataset.
pub const IMAGE_WIDTH: usize = 28;
/// The height (in px) of the images in the training dataset.
pub const IMAGE_HEIGHT: usize = 28;
/// The total number of pixels in the images in the training dataset.
pub const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

/// The size of the label in bytes.
pub const LABEL_SIZE: usize = 1;

/// The length of the input vector passed into the multi-layer perceptron.
pub const INPUT_VECTOR_LENGTH: usize = IMAGE_SIZE;
/// The length of the output vector produced by the multi-layer perceptron.
pub const OUTPUT_VECTOR_LENGTH: usize = 10;

/// The size of the metadata (in bytes) of the images dataset file.
pub const IMAGE_DATASET_METADATA_SIZE_BYTES: usize = 16;
/// The size of the metadata (in bytes) of the label dataset file.
pub const LABEL_DATASET_METADATA_SIZE_BYTES: usize = 8;

/// The function used to convert (big endian) training dataset metadata into regular `usize`s.   
pub const fn usize_from_training_data_endian_bytes(bytes: &[u8]) -> usize {
    assert!(bytes.len() == 4);

    let _bytes: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
    return u32::from_be_bytes(_bytes) as usize;
}
