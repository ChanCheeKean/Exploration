"""Input functions used in dual encoder SMITH model."""

import sys
import os
sys.path.append(os.path.abspath(os.getcwd()) + '/smith/config')
import tensorflow.compat.v1 as tf  # tf
import constants

def input_fn_builder(input_files,
                     is_training,
                     drop_remainder,
                     max_seq_length=32,
                     max_predictions_per_seq=5,
                     num_cpu_threads=4,
                     batch_size=16,
                     is_prediction=False,
                     train_mode='finetune'):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):  # pylint: disable=unused-argument
    """The actual input function."""
    name_to_features = {
        "input_ids_1": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask_1": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_ids_2": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask_2": tf.FixedLenFeature([max_seq_length], tf.int64),
        "documents_match_labels": tf.FixedLenFeature([1], tf.float32, 0)
    }
    if (train_mode == constants.TRAIN_MODE_PRETRAIN or train_mode == constants.TRAIN_MODE_JOINT_TRAIN):
      # Add some features related to word masked LM losses.
      name_to_features["masked_lm_positions_1"] = tf.FixedLenFeature([max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_ids_1"] = tf.FixedLenFeature([max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_weights_1"] = tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
      name_to_features["masked_lm_positions_2"] = tf.FixedLenFeature([max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_ids_2"] = tf.FixedLenFeature([max_predictions_per_seq], tf.int64)
      name_to_features["masked_lm_weights_2"] = tf.FixedLenFeature([max_predictions_per_seq], tf.float32)

    # For training, we want a lot of parallel reading and shuffling, but not for eval
    if is_training:
      file_list = tf.data.Dataset.list_files(tf.constant(input_files))
      file_list = file_list.shuffle(buffer_size=len(input_files))
      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))
      # `sloppy` mode means that the interleaving is not exact. This adds more randomness to the training pipeline.
      d = file_list.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    else:
      d = tf.data.TFRecordDataset(tf.constant(input_files))
      # prediction should raise an end-of-input exception (OutOfRangeError or StopIteration)
      # which serves as the stopping signal to TPUEstimator.
      if not is_prediction:
        # Since we evaluate for a fixed number of steps we don't want to encounter out-of-range exceptions.
        d = d.repeat()

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features, train_mode),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=drop_remainder))
    return d

  return input_fn


def _decode_record(record, name_to_features, train_mode='finetune'):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)
  example["input_ids_1"] = tf.cast(example["input_ids_1"], tf.int32)
  example["input_ids_2"] = tf.cast(example["input_ids_2"], tf.int32)
  example["documents_match_labels"] = tf.cast(example["documents_match_labels"], tf.float32)
  example["input_mask_1"] = tf.cast(example["input_mask_1"], tf.int32)
  example["input_mask_2"] = tf.cast(example["input_mask_2"], tf.int32)
  if (train_mode == constants.TRAIN_MODE_PRETRAIN or train_mode == constants.TRAIN_MODE_JOINT_TRAIN):
    example["masked_lm_ids_1"] = tf.cast(example["masked_lm_ids_1"], tf.int32)
    example["masked_lm_ids_2"] = tf.cast(example["masked_lm_ids_2"], tf.int32)
    example["masked_lm_weights_1"] = tf.cast(example["masked_lm_weights_1"], tf.float32)
    example["masked_lm_weights_2"] = tf.cast(example["masked_lm_weights_2"], tf.float32)
    example["masked_lm_positions_1"] = tf.cast(example["masked_lm_positions_1"], tf.int32)
    example["masked_lm_positions_2"] = tf.cast(example["masked_lm_positions_2"], tf.int32)
  return example
  
def get_actual_max_seq_len(max_doc_length_by_sentence, max_sent_length_by_word, max_predictions_per_seq):
  """Get the actual maximum sequence length.

  Args:
    max_doc_length_by_sentence: The maximum document length by the number of sentences.
    max_sent_length_by_word: The maximum sentence length by the number of words.
    max_predictions_per_seq: The maximum number of predicted masked tokens insequence

  Returns:
    The actual maximum sequence length and maximum number of masked LM predictions per sequence. 

  Raises:
    ValueError: if the arguments are not usable.

  """
  max_seq_length_actual = max_doc_length_by_sentence * max_sent_length_by_word
  max_predictions_per_seq_actual = max_doc_length_by_sentence * max_predictions_per_seq
  return (max_seq_length_actual, max_predictions_per_seq_actual)