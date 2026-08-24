use crate::constants;

/// Represents the content of a single entry in the training dataset.
pub struct TrainingSample
{
    /// The position of this sample/entry with respect to the beginning of the dataset.
    pub index   : usize,
    /// The input vector passed into the multi-layer perceptron.
    pub input   : ndarray::Array1<f32>,
    /// The answer vector that the multi-layer perceptron must aim to match.
    pub answer  : ndarray::Array1<f32>
}

impl TrainingSample
{
    /// Creates a training sample using raw data (slices).
    pub fn from_raw(idx: usize, input: &[u8], answer: &[u8]) -> Result<Self, String> {
        if answer.len() != constants::LABEL_SIZE { 
            return Err(format!("Length of answer slice is not the same as label-size.")); 
        }
        if (*answer.get(0).unwrap() as usize) >= constants::OUTPUT_VECTOR_LENGTH { 
            return Err(format!("label must less than output vector length (one-hot vector).")); 
        }

        let mut answer_vec = ndarray::Array1::from_elem(constants::OUTPUT_VECTOR_LENGTH, 0_f32);
        *answer_vec.get_mut(*answer.get(0).unwrap() as usize).unwrap() = 1_f32;

        return Ok(Self {
            index: idx,
            input: ndarray::Array1::from_iter(input.iter().map(|x| (*x as f32)/255_f32)),
            answer: answer_vec
        });
    }

    /// Tries to convert the label as a scalar value by returning the index of the largest element.
    pub fn get_label(&self) -> Option<u8> {
        return Some(self.answer
            .iter()
            .enumerate()
            .max_by(|(a1,a2), (b1, b2)| a2.total_cmp(b2))?
            .0 as u8
        );
    }
}

impl Clone for TrainingSample
{
    fn clone(&self) -> Self {
        return Self {
            index   : self.index,
            input   : self.input.clone(),
            answer  : self.answer.clone()
        };
    }
}

use crate::load::loader::TrainingReader;

/// Tries to seek past the training dataset metadata.
fn _seek_past_metadata(imageset_reader: &mut TrainingReader, labelset_reader: &mut TrainingReader) -> Result<(), String> {
    if let Err(err) =  imageset_reader.seek_relative(constants::IMAGE_DATASET_METADATA_SIZE_BYTES as i64).and(
            labelset_reader.seek_relative(constants::LABEL_DATASET_METADATA_SIZE_BYTES as i64)
        ) {
        return Err(format!("Failed to seek past dataset metadata due to error: {:?}", err));
    }

    return Ok(());
}

/// Creates an input iterator that traverses through the image and label datasets via a `BufReader` in chunks.'
///
/// - `images_dataset`: A path to a file containing the image data.
/// - `labels_dataset`: A path to a file containing the corresponding label data.
///
/// Returns an iterator which yields (by value) each training sample/entry in the dataset.
/// 
/// ### Notes:
/// 
/// **This iterator must only be called after validating the dataset.** i.e., 
/// 
pub fn training_iterator(images_dataset: &std::path::Path, labels_dataset: &std::path::Path) -> impl Iterator<Item = Result<TrainingSample, String>> {
    use genawaiter::sync::Gen   as Generator;
    use genawaiter::sync::Co    as Coroutine;

    use std::io::Read;

    use crate::load::loader::get_file_readers_for_training_data;

    return Generator::new(|co: Coroutine<Result<TrainingSample, String>>| async move {

        let result = get_file_readers_for_training_data(images_dataset, labels_dataset);
        if let Err(err) = result { co.yield_(Err(err)).await; return; }

        let (mut images_reader, mut labels_reader) = result.unwrap();
        if let Err(err) = _seek_past_metadata(&mut images_reader, &mut labels_reader) { co.yield_(Err(err)).await; return; }

        let mut idx: usize = 0;
        let mut input_buf = [0 as u8; constants::IMAGE_SIZE];
        let mut answer_buf = [0 as u8; constants::LABEL_SIZE];

        loop {
            if let Err(err) = images_reader.read_exact(&mut input_buf).and(labels_reader.read_exact(&mut answer_buf)) {
                if err.kind() == std::io::ErrorKind::UnexpectedEof { return; }

                co.yield_(Err(format!("Reading failed at index: {} due to error: {:?}.", idx, err))).await;
                return;
            }

            co.yield_(TrainingSample::from_raw(idx, &input_buf, &answer_buf)).await;
            idx += 1;
        };

    }).into_iter();
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
            "training_iterator_{}_{}_{}",
            name,
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn from_raw_creates_normalized_input_and_one_hot_answer() {
        let input = [0, 128, 255];
        let answer = [3];

        let sample = TrainingSample::from_raw(7, &input, &answer).unwrap();

        assert_eq!(sample.index, 7);
        assert_eq!(sample.input.len(), input.len());
        assert!((sample.input[0] - 0.0).abs() < f32::EPSILON);
        assert!((sample.input[1] - (128.0 / 255.0)).abs() < f32::EPSILON);
        assert!((sample.input[2] - 1.0).abs() < f32::EPSILON);

        assert_eq!(sample.answer.len(), constants::OUTPUT_VECTOR_LENGTH);
        assert_eq!(sample.answer[3], 1.0);
        assert_eq!(
            sample.answer.iter().filter(|&&value| value == 1.0).count(),
            1
        );
    }

    #[test]
    fn from_raw_rejects_invalid_answer_length() {
        let answer = vec![0; constants::LABEL_SIZE.saturating_sub(1)];

        let result = TrainingSample::from_raw(0, &[], &answer);

        assert!(result.is_err());
    }

    #[test]
    fn from_raw_rejects_out_of_range_label() {
        let answer = [constants::OUTPUT_VECTOR_LENGTH as u8];

        let result = TrainingSample::from_raw(0, &[], &answer);

        assert!(result.is_err());
    }

    #[test]
    fn get_label_returns_index_of_largest_answer() {
        let mut sample = TrainingSample::from_raw(0, &[], &[0]).unwrap();
        sample.answer.fill(0.0);
        sample.answer[5] = 0.75;
        sample.answer[2] = 0.25;

        assert_eq!(sample.get_label(), Some(5));
    }

    #[test]
    fn clone_preserves_sample_contents() {
        let sample = TrainingSample::from_raw(12, &[10, 20], &[1]).unwrap();
        let cloned = sample.clone();

        assert_eq!(cloned.index, sample.index);
        assert_eq!(cloned.input, sample.input);
        assert_eq!(cloned.answer, sample.answer);
    }

    #[test]
    fn iterator_reads_images_and_labels() {
        let images_path = temporary_path("images");
        let labels_path = temporary_path("labels");

        let image_one = vec![0x00; constants::IMAGE_SIZE];
        let image_two = vec![0xff; constants::IMAGE_SIZE];

        let mut image_data =
            vec![0; constants::IMAGE_DATASET_METADATA_SIZE_BYTES];
        image_data.extend_from_slice(&image_one);
        image_data.extend_from_slice(&image_two);

        let mut label_data =
            vec![0; constants::LABEL_DATASET_METADATA_SIZE_BYTES];
        label_data.extend_from_slice(&vec![1; constants::LABEL_SIZE]);
        label_data.extend_from_slice(&vec![2; constants::LABEL_SIZE]);

        fs::write(&images_path, image_data).unwrap();
        fs::write(&labels_path, label_data).unwrap();

        let samples: Vec<_> = training_iterator(&images_path, &labels_path)
            .map(|result| result.unwrap())
            .collect();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].index, 0);
        assert_eq!(samples[1].index, 1);
        assert_eq!(samples[0].get_label(), Some(1));
        assert_eq!(samples[1].get_label(), Some(2));
        assert!(samples[0].input.iter().all(|&value| value == 0.0));
        assert!(samples[1].input.iter().all(|&value| value == 1.0));

        let _ = fs::remove_file(images_path);
        let _ = fs::remove_file(labels_path);
    }

    #[test]
    fn iterator_reports_missing_images_file() {
        let images_path = temporary_path("missing_images");
        let labels_path = temporary_path("labels");

        fs::write(&labels_path, []).unwrap();

        let mut iterator = training_iterator(&images_path, &labels_path);
        assert!(iterator.next().unwrap().is_err());

        let _ = fs::remove_file(&labels_path);
    }

    #[test]
    fn iterator_reports_missing_labels_file() {
        let images_path = temporary_path("images");
        let labels_path = temporary_path("missing_labels");

        fs::write(&images_path, []).unwrap();

        let mut iterator = training_iterator(&images_path, &labels_path);
        assert!(iterator.next().unwrap().is_err());

        let _ = fs::remove_file(&images_path);
    }
}