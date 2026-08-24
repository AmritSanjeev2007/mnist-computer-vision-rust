use std::io::Read;

use crate::constants;

#[derive(Clone, Copy)]
pub struct TrainingMetadata
{
    /// The magic number for the images dataset file format.
    pub imageset_magic_number: usize,
    /// The magic number for the labels dataset file format.
    pub labelset_magic_number: usize,

    /// The number of images in the images dataset.
    pub images_count: usize,
    /// The number of labels in the labels dataset.
    pub labels_count: usize,
    
    /// The number of rows in an image.
    pub image_rows_count: usize,
    /// The number of columns in an image.
    pub image_columns_count: usize
}

/// Represents the reader used while traversing the training dataset.
pub type TrainingReader = std::io::BufReader<std::fs::File>;

/// Returns readers for traversing or just peeking at the training dataset. Returns a tuple of readers: `(imageset_reader, labelset_reader)`.
pub fn get_file_readers_for_training_data(imageset: &std::path::Path, labelset: &std::path::Path) -> Result<(TrainingReader, TrainingReader), String>
{
    let images_file_desc = std::fs::OpenOptions::new().read(true).open(imageset);
    if let Err(err) = images_file_desc {
        return Err(format!("Failed to open images dataset due to error: {:?}.", err));
    }

    let labels_file_desc = std::fs::OpenOptions::new().read(true).open(labelset);
    if let Err(err) = labels_file_desc {
        return Err(format!("Failed to open labels dataset due to error: {:?}", err));
    }

    let mut images_reader = std::io::BufReader::new(images_file_desc.unwrap());
    let mut labels_reader = std::io::BufReader::new(labels_file_desc.unwrap());

    return Ok((images_reader, labels_reader));
}

impl TrainingMetadata
{
    /// Tries to load the metadata from the image and label datasets.
    pub fn load(imageset: &std::path::Path, labelset: &std::path::Path) -> Result<Self, String> {
        let result = get_file_readers_for_training_data(imageset, labelset);
        if let Err(err) = result { return Err(err); }

        let (mut images_reader, mut labels_reader) = result.unwrap();

        let mut imageset_metadata = [0 as u8; constants::IMAGE_DATASET_METADATA_SIZE_BYTES];
        let mut labelset_metadata = [0 as u8; constants::LABEL_DATASET_METADATA_SIZE_BYTES];

        if let Err(err) = images_reader.read_exact(&mut imageset_metadata).and(labels_reader.read_exact(&mut labelset_metadata)) {
            return Err(format!("Failed to read metadata due to error: {:?}.", err));
        }

        let imageset_magic_number   = constants::usize_from_training_data_endian_bytes(&imageset_metadata[0..4]);
        let images_count            = constants::usize_from_training_data_endian_bytes(&imageset_metadata[4..8]);
        let rows_count              = constants::usize_from_training_data_endian_bytes(&imageset_metadata[8..12]);
        let columns_count           = constants::usize_from_training_data_endian_bytes(&imageset_metadata[12..16]);

        let labelset_magic_number   = constants::usize_from_training_data_endian_bytes(&labelset_metadata[0..4]);
        let label_count             = constants::usize_from_training_data_endian_bytes(&labelset_metadata[4..8]);

        return Ok(Self {
            imageset_magic_number: imageset_magic_number,
            images_count: images_count,
            image_rows_count: rows_count,
            image_columns_count: columns_count,

            labelset_magic_number: labelset_magic_number,
            labels_count: label_count
        });
    }

    /// Validates the metadata with respect to the datasets.
    pub const fn is_valid(&self) -> bool {
        return (
            self.imageset_magic_number == constants::IMAGE_DATASET_MAGIC_NUMBER &&
            self.labelset_magic_number == constants::LABEL_DATASET_MAGIC_NUMBER &&
            self.images_count == self.labels_count &&
            self.image_rows_count == constants::IMAGE_HEIGHT &&
            self.image_columns_count == constants::IMAGE_WIDTH
        );
    } 
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temporary_path(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        std::env::temp_dir().join(format!(
            "loader_{}_{}_{}",
            name,
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn write_training_metadata(images_count: usize, labels_count: usize) -> (PathBuf, PathBuf) {
        let images_path = temporary_path("images");
        let labels_path = temporary_path("labels");

        let mut imageset_metadata = Vec::with_capacity(constants::IMAGE_DATASET_METADATA_SIZE_BYTES);
        imageset_metadata.extend_from_slice(&u32::to_be_bytes(constants::IMAGE_DATASET_MAGIC_NUMBER as u32));
        imageset_metadata.extend_from_slice(&u32::to_be_bytes(images_count as u32));
        imageset_metadata.extend_from_slice(&u32::to_be_bytes(constants::IMAGE_HEIGHT as u32));
        imageset_metadata.extend_from_slice(&u32::to_be_bytes(constants::IMAGE_WIDTH as u32));

        let mut labelset_metadata = Vec::with_capacity(constants::LABEL_DATASET_METADATA_SIZE_BYTES);
        labelset_metadata.extend_from_slice(&u32::to_be_bytes(constants::LABEL_DATASET_MAGIC_NUMBER as u32));
        labelset_metadata.extend_from_slice(&u32::to_be_bytes(labels_count as u32));

        fs::write(&images_path, &imageset_metadata).unwrap();
        fs::write(&labels_path, &labelset_metadata).unwrap();

        (images_path, labels_path)
    }

    #[test]
    fn get_file_readers_for_training_data_opens_both_files() {
        let (images_path, labels_path) = write_training_metadata(2, 2);

        let result = get_file_readers_for_training_data(&images_path, &labels_path);

        assert!(result.is_ok());

        let _ = result.unwrap();
        let _ = fs::remove_file(images_path);
        let _ = fs::remove_file(labels_path);
    }

    #[test]
    fn get_file_readers_for_training_data_reports_missing_files() {
        let images_path = temporary_path("missing_images");
        let labels_path = temporary_path("missing_labels");

        let result = get_file_readers_for_training_data(&images_path, &labels_path);

        assert!(result.is_err());
    }

    #[test]
    fn training_metadata_load_reads_and_parses_valid_metadata() {
        let (images_path, labels_path) = write_training_metadata(7, 7);

        let metadata = TrainingMetadata::load(&images_path, &labels_path).unwrap();

        assert_eq!(metadata.imageset_magic_number, constants::IMAGE_DATASET_MAGIC_NUMBER);
        assert_eq!(metadata.labelset_magic_number, constants::LABEL_DATASET_MAGIC_NUMBER);
        assert_eq!(metadata.images_count, 7);
        assert_eq!(metadata.labels_count, 7);
        assert_eq!(metadata.image_rows_count, constants::IMAGE_HEIGHT);
        assert_eq!(metadata.image_columns_count, constants::IMAGE_WIDTH);
        assert!(metadata.is_valid());

        let _ = fs::remove_file(images_path);
        let _ = fs::remove_file(labels_path);
    }

    #[test]
    fn training_metadata_load_rejects_unreadable_metadata() {
        let images_path = temporary_path("broken_images");
        let labels_path = temporary_path("broken_labels");

        fs::write(&images_path, b"short").unwrap();
        fs::write(&labels_path, b"tiny").unwrap();

        let result = TrainingMetadata::load(&images_path, &labels_path);

        assert!(result.is_err());

        let _ = fs::remove_file(images_path);
        let _ = fs::remove_file(labels_path);
    }

    #[test]
    fn training_metadata_is_valid_rejects_mismatched_dataset_metadata() {
        let (images_path, labels_path) = write_training_metadata(10, 9);

        let metadata = TrainingMetadata::load(&images_path, &labels_path).unwrap();

        assert!(!metadata.is_valid());

        let _ = fs::remove_file(images_path);
        let _ = fs::remove_file(labels_path);
    }
}

