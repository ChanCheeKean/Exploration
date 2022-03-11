"""Dual encoder SMITH models."""


import sys
import os
sys.path.append(os.path.abspath(os.getcwd()) + '/smith/config')
import json
import pickle
import os
from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
import constants
from smith import input_fns
from smith import modeling as smith_modeling
from smith import utils
from config import dictConveter

flags.DEFINE_string("dual_encoder_config_file", None,
                    "The proto config file for dual encoder SMITH models.")

flags.DEFINE_enum(
    "train_mode", 'finetune', ["finetune", "pretrain", "joint_train"],
    "Whether it is joint_train, pretrain or finetune. The difference is "
    "about total_loss calculation and input files for eval and training.")

flags.DEFINE_enum(
    "schedule", 'predict', ["train", "continuous_eval", "predict", "export"],
    "The run schedule which can be any one of train, continuous_eval, "
    "predict or export.")

FLAGS = flags.FLAGS

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  train_mode = FLAGS.train_mode
  with open(FLAGS.dual_encoder_config_file, 'rb') as f:
     exp_config = pickle.load(f)

  max_seq_length_actual, max_predictions_per_seq_actual = \
        input_fns.get_actual_max_seq_len(
            exp_config.encoder_config.max_doc_length_by_sentence,
            exp_config.encoder_config.max_sent_length_by_word,
            exp_config.encoder_config.max_predictions_per_seq)

  # Prepare input for train and eval.
  input_files = exp_config.train_eval_config.processed_file
  tf.gfile.MakeDirs(exp_config.train_eval_config.model_output_dir)
  
  input_fn_builder = input_fns.input_fn_builder
  eval_drop_remainder = True if exp_config.train_eval_config.use_tpu else False
  
  train_input_fn = input_fn_builder(
      input_files=input_files,
      is_training=True,
      drop_remainder=True,
      max_seq_length=max_seq_length_actual,
      max_predictions_per_seq=max_predictions_per_seq_actual,
      num_cpu_threads=4,
      batch_size=exp_config.train_eval_config.batch_size,
      train_mode=FLAGS.train_mode)
  
  eval_input_fn = input_fn_builder(
      input_files=input_files,
      max_seq_length=max_seq_length_actual,
      max_predictions_per_seq=max_predictions_per_seq_actual,
      is_training=False,
      drop_remainder=eval_drop_remainder,
      batch_size=exp_config.train_eval_config.batch_size,
      train_mode=FLAGS.train_mode)
      
  predict_input_fn = input_fn_builder(
      input_files=input_files,
      max_seq_length=max_seq_length_actual,
      max_predictions_per_seq=max_predictions_per_seq_actual,
      is_training=False,
      drop_remainder=eval_drop_remainder,
      batch_size=exp_config.train_eval_config.batch_size,
      is_prediction=True,
      train_mode=FLAGS.train_mode)

  # Build and run the model.
  tpu_cluster_resolver = None
  if exp_config.train_eval_config.use_tpu and exp_config.train_eval_config.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        exp_config.train_eval_config.tpu_name, 
        zone=exp_config.train_eval_config.tpu_zone, 
        project=exp_config.train_eval_config.gcp_project)

  is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=exp_config.train_eval_config.master,
      model_dir=exp_config.train_eval_config.model_output_dir,
      save_checkpoints_steps=exp_config.train_eval_config.save_checkpoints_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=1000,
          num_shards=exp_config.train_eval_config.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = smith_modeling.model_fn_builder(
      dual_encoder_config=exp_config,
      train_mode=FLAGS.train_mode,
      learning_rate=exp_config.train_eval_config.learning_rate,
      num_train_steps=exp_config.train_eval_config.num_train_steps,
      num_warmup_steps=exp_config.train_eval_config.num_warmup_steps,
      use_tpu=exp_config.train_eval_config.use_tpu,
      use_one_hot_embeddings=exp_config.train_eval_config.use_tpu,
      debugging=False)

  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=exp_config.train_eval_config.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=exp_config.train_eval_config.batch_size,
      eval_batch_size=exp_config.train_eval_config.batch_size,
      predict_batch_size=exp_config.train_eval_config.batch_size)

  if FLAGS.schedule == "train":
    tf.logging.info("***** Running training Final*****")
    estimator.train(input_fn=train_input_fn, max_steps=exp_config.train_eval_config.num_train_steps)
    
  elif FLAGS.schedule == "continuous_eval":
    tf.logging.info("***** Running continuous evaluation *****")
    # checkpoints_iterator blocks until a new checkpoint appears.
    for ckpt in tf.train.checkpoints_iterator(estimator.model_dir):
      try:
        # Estimator automatically loads and evaluates the latest checkpoint.
        result = estimator.evaluate(input_fn=eval_input_fn, steps=50)
        tf.logging.info("***** Eval results for %s *****", ckpt)
        for key, value in result.items():
          tf.logging.info("  %s = %s", key, str(value))

      except tf.errors.NotFoundError:
        # Checkpoint might get garbage collected before the eval can run.
        tf.logging.error("Checkpoint path '%s' no longer exists.", ckpt)
        
  elif FLAGS.schedule == "predict":
    # Load the model checkpoint and run the prediction process
    tf.logging.info("***** Running prediction with ckpt {} *****".format(exp_config.encoder_config.predict_checkpoint))
    output_predict_file = exp_config.train_eval_config.pred_output_file
    # Output the prediction results in json format.
    pred_res_list = []
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      written_line_index = 0
      tf.logging.info("***** Predict results *****")
      for result in estimator.predict(
          input_fn=predict_input_fn,
          checkpoint_path=exp_config.encoder_config.predict_checkpoint,
          yield_single_examples=True):
        pred_item_dict = utils.get_pred_res_list_item_smith_de(result)
        pred_res_list.append(pred_item_dict)
        written_line_index += 1
        if written_line_index % 100 == 0:
          tf.logging.info("Current written_line_index: {} *****".format(written_line_index))
      tf.logging.info("***** Output prediction results into %s *****",output_predict_file)
      json.dump(pred_res_list, writer)

  else:
    raise ValueError("Unsupported schedule : %s" % FLAGS.schedule)

if __name__ == "__main__":
  app.run(main)
