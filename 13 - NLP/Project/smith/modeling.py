"""Dual encoder SMITH models."""

import sys
import os
sys.path.append(os.path.abspath(os.getcwd()) + '/smith/config')
import tensorflow.compat.v1 as tf
import constants
from smith import layers
from smith import loss_fns
from smith import metric_fns
from smith import utils
from smith.bert import modeling
from smith.bert import optimization

def build_smith_dual_encoder(dual_encoder_config,
                             train_mode,
                             is_training,
                             input_ids_1,
                             input_mask_1,
                             masked_lm_positions_1,
                             masked_lm_ids_1,
                             masked_lm_weights_1,
                             input_ids_2,
                             input_mask_2,
                             masked_lm_positions_2,
                             masked_lm_ids_2,
                             masked_lm_weights_2,
                             use_one_hot_embeddings,
                             documents_match_labels,
                             debugging=False):
  
  """Build the dual encoder SMITH model.
  Args:
    dual_encoder_config: the configuration file for the dual encoder model.
    train_mode: string. The train mode of the current. It can be finetune, pretrain or joint_train.
    is_training: bool. Whether it in training mode.
    input_ids_1: int Tensor with shape [batch, max_seq_length]. The input ids of input 1.
    input_mask_1: int Tensor with shape [batch, max_seq_length]. The input masks of input 1.
    masked_lm_positions_1: int Tensor with shape [batch, max_predictions_per_seq]. The input masked LM prediction positions of input 1. 
    masked_lm_ids_1: int Tensor with shape [batch, max_predictions_per_seq]. The input masked LM prediction ids (ground Truth) of input 1. 
    masked_lm_weights_1: float Tensor with shape [batch, max_predictions_per_seq]. The input masked LM prediction weights of input 1.
    input_ids_2: int Tensor with shape [batch, max_seq_length]. The input ids of input 2.
    input_mask_2: int Tensor with shape [batch, max_seq_length]. The input masks of input 2.
    masked_lm_positions_2: int Tensor with shape [batch, max_predictions_per_seq]. The input masked LM prediction positions of input 2.
    masked_lm_ids_2: int Tensor with shape [batch, max_predictions_per_seq]. The input masked LM prediction ids (ground truth) of input 2. 
    masked_lm_weights_2: float Tensor with shape [batch, max_predictions_per_seq]. The input masked LM prediction weights of input 2.
    use_one_hot_embeddings: bool. Whether use one hot embeddings.
    documents_match_labels: float Tensor with shape [batch]. The ground truth labels for the input examples.
    debugging: bool. Whether it is in the debugging mode.

  Returns:
    The masked LM loss, per example LM loss, masked sentence LM loss, per
    example masked sentence LM loss, sequence representations, text matching
    loss, per example text matching loss, text matching logits, text matching
    probabilities and text matching log probabilities.
  Raises:
    ValueError: if the doc_rep_combine_mode in dual_encoder_config is invalid.
  """

  bert_config = modeling.BertConfig.from_json_file(dual_encoder_config.encoder_config.bert_config_file)
  doc_bert_config = modeling.BertConfig.from_json_file(dual_encoder_config.encoder_config.doc_bert_config_file)
  
  # to generate sentence repressentation for max_seq_length
  (input_sent_reps_doc_1_unmask, input_mask_doc_level_1_tensor,
   input_sent_reps_doc_2_unmask, input_mask_doc_level_2_tensor,
   masked_lm_loss_doc_1, masked_lm_loss_doc_2, masked_lm_example_loss_doc_1,
   masked_lm_example_loss_doc_2, masked_lm_weights_doc_1, masked_lm_weights_doc_2) = \
   layers.learn_sent_reps_normal_loop(
       dual_encoder_config, is_training, train_mode, input_ids_1, input_mask_1,
       masked_lm_positions_1, masked_lm_ids_1, masked_lm_weights_1, input_ids_2,
       input_mask_2, masked_lm_positions_2, masked_lm_ids_2,
       masked_lm_weights_2, use_one_hot_embeddings)

  if debugging:
    input_mask_doc_level_1_tensor = tf.Print(
        input_mask_doc_level_1_tensor,
        [input_mask_doc_level_1_tensor, input_mask_doc_level_2_tensor],
        message="input_mask_doc_level_1_tensor in build_smith_dual_encoder",
        summarize=30)

  if dual_encoder_config.encoder_config.use_masked_sentence_lm_loss:
    batch_size_static = (dual_encoder_config.train_eval_config.train_batch_size if is_training
                          else dual_encoder_config.train_eval_config.eval_batch_size)

    # Generates the sentence masked document representations.
    with tf.variable_scope("mask_sent_in_doc", reuse=tf.AUTO_REUSE):
      sent_mask_embedding = tf.get_variable(
        name="sentence_mask_embedding",
        shape=[bert_config.hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=bert_config.initializer_range)
        )

      # Output Shape: [batch, loop_sent_number_per_doc, hidden].
      (input_sent_reps_doc_1_masked, masked_sent_index_1, masked_sent_weight_1) = \
      layers.get_doc_rep_with_masked_sent(
        input_sent_reps_doc=input_sent_reps_doc_1_unmask,
        sent_mask_embedding=sent_mask_embedding,
        input_mask_doc_level=input_mask_doc_level_1_tensor,
        batch_size_static=batch_size_static,
        max_masked_sent_per_doc=dual_encoder_config.encoder_config.max_masked_sent_per_doc,
        loop_sent_number_per_doc=dual_encoder_config.encoder_config.loop_sent_number_per_doc)

      (input_sent_reps_doc_2_masked, masked_sent_index_2, masked_sent_weight_2) = \
      layers.get_doc_rep_with_masked_sent(
        input_sent_reps_doc=input_sent_reps_doc_2_unmask,
        sent_mask_embedding=sent_mask_embedding,
        input_mask_doc_level=input_mask_doc_level_2_tensor,
        batch_size_static=batch_size_static,
        max_masked_sent_per_doc=dual_encoder_config.encoder_config.max_masked_sent_per_doc,
        loop_sent_number_per_doc=dual_encoder_config.encoder_config.loop_sent_number_per_doc)

    # Learn the document representations based on masked sentence embeddings.
    model_doc_1 = modeling.DocBertModel(
        config=doc_bert_config,
        is_training=is_training,
        input_reps=input_sent_reps_doc_1_masked,
        input_mask=input_mask_doc_level_1_tensor)

    model_doc_2 = modeling.DocBertModel(
        config=doc_bert_config,
        is_training=is_training,
        input_reps=input_sent_reps_doc_2_masked,
        input_mask=input_mask_doc_level_2_tensor)

    # Shape of masked_sent_lm_loss_1 [1].
    # Shape of masked_sent_lm_example_loss_1 is [batch * max_predictions_per_seq].
    (masked_sent_lm_loss_1, masked_sent_per_example_loss_1, _) = \
    layers.get_masked_sent_lm_output(doc_bert_config,
                                      model_doc_1.get_sequence_output(),
                                      input_sent_reps_doc_1_unmask,
                                      masked_sent_index_1,
                                      masked_sent_weight_1)

    (masked_sent_lm_loss_2, masked_sent_per_example_loss_2,_) = \
    layers.get_masked_sent_lm_output(doc_bert_config,
                                      model_doc_2.get_sequence_output(),
                                      input_sent_reps_doc_2_unmask,
                                      masked_sent_index_2,
                                      masked_sent_weight_2)
  else:
    # Learn the document representations based on unmasked sentence embeddings.
    model_doc_1 = modeling.DocBertModel(
      config=doc_bert_config,
      is_training=is_training,
      input_reps=input_sent_reps_doc_1_unmask,
      input_mask=input_mask_doc_level_1_tensor)

    model_doc_2 = modeling.DocBertModel(
      config=doc_bert_config,
      is_training=is_training,
      input_reps=input_sent_reps_doc_2_unmask,
      input_mask=input_mask_doc_level_2_tensor)

    masked_sent_lm_loss_1, masked_sent_lm_loss_2 = 0, 0
    masked_sent_per_example_loss_1, masked_sent_per_example_loss_2 = tf.zeros(1), tf.zeros(1)
    masked_sent_weight_1, masked_sent_weight_2 = tf.zeros(1), tf.zeros(1)

  with tf.variable_scope("seq_rep_from_bert_doc_dense", reuse=tf.AUTO_REUSE):
    normalized_doc_rep_1 = layers.get_seq_rep_from_bert(model_doc_1)
    normalized_doc_rep_2 = layers.get_seq_rep_from_bert(model_doc_2)

    # contextualized sentence embedding output by document level Transformer model
    output_sent_reps_doc_1 = model_doc_1.get_sequence_output()
    output_sent_reps_doc_2 = model_doc_2.get_sequence_output()

  # Generate the final document representations representations based on the word/sentence/document level representations
  # 1. normal: only use the document level representation as the final document representations.
  # 2. sum_concat:  compute the sum of all sentence level repsentations then concatenate the sum vector with the document representations.
  # 3. mean_concat: compute the mean of all sentence level repsentations then concatenate the mean vector with the document representations
  # 4. attention: compute the weighted sum of sentence level representations with attention mechanism, then concatenate with the document level representations.

  # The document level mask is to indicate whether each sentence is a real sentence (1) or a paded sentence (0). 
  # The shape of input_mask_doc_level_1_tensor is [batch, max_doc_length_by_sentence]. 
  # The shape of input_sent_reps_doc_1_unmask is [batch, max_doc_length_by_sentence, hidden].
  final_doc_rep_combine_mode = dual_encoder_config.encoder_config.doc_rep_combine_mode

  if final_doc_rep_combine_mode == constants.DOC_COMBINE_NORMAL:
    final_doc_rep_1 = normalized_doc_rep_1
    final_doc_rep_2 = normalized_doc_rep_2

  elif final_doc_rep_combine_mode == constants.DOC_COMBINE_SUM_CONCAT:
    # Output Shape: [batch, 2*hidden].
    final_doc_rep_1 = tf.concat([tf.reduce_sum(input_sent_reps_doc_1_unmask, 1), normalized_doc_rep_1],axis=1)
    final_doc_rep_2 = tf.concat([tf.reduce_sum(input_sent_reps_doc_2_unmask, 1), normalized_doc_rep_2], axis=1)

  elif final_doc_rep_combine_mode == constants.DOC_COMBINE_MEAN_CONCAT:
    final_doc_rep_1 = tf.concat([tf.reduce_mean(input_sent_reps_doc_1_unmask, 1), normalized_doc_rep_1], axis=1)
    final_doc_rep_2 = tf.concat([tf.reduce_mean(input_sent_reps_doc_2_unmask, 1), normalized_doc_rep_2], axis=1)

  elif final_doc_rep_combine_mode == constants.DOC_COMBINE_ATTENTION:
    final_doc_rep_1 = tf.concat([
      layers.get_attention_weighted_sum(input_sent_reps_doc_1_unmask, bert_config, is_training, dual_encoder_config.encoder_config.doc_rep_combine_attention_size),
        normalized_doc_rep_1
    ], axis=1)

    final_doc_rep_2 = tf.concat([
        layers.get_attention_weighted_sum(
          input_sent_reps_doc_2_unmask, bert_config, is_training, dual_encoder_config.encoder_config.doc_rep_combine_attention_size),
        normalized_doc_rep_2
    ], axis=1)

  else:
    raise ValueError("Only normal, sum_concat, mean_concat and attention are"
                     " supported: %s" % final_doc_rep_combine_mode)

  (siamese_loss, siamese_example_loss, siamese_logits) = \
  loss_fns.get_prediction_loss_cosine(
       input_tensor_1=final_doc_rep_1,
       input_tensor_2=final_doc_rep_2,
       labels=documents_match_labels,
       similarity_score_amplifier=dual_encoder_config.loss_config.similarity_score_amplifier,
       neg_to_pos_example_ratio=dual_encoder_config.train_eval_config.neg_to_pos_example_ratio)

  # The shape of masked_lm_loss_doc is [1].
  # The shape of masked_lm_example_loss_doc is [batch * max_predictions_per_seq,
  # max_doc_length_by_sentence].

  return (masked_lm_loss_doc_1, masked_lm_loss_doc_2,
          masked_lm_example_loss_doc_1, masked_lm_example_loss_doc_2,
          masked_lm_weights_doc_1, masked_lm_weights_doc_2,
          masked_sent_lm_loss_1, masked_sent_lm_loss_2,
          masked_sent_per_example_loss_1, masked_sent_per_example_loss_2,
          masked_sent_weight_1, masked_sent_weight_2, final_doc_rep_1,
          final_doc_rep_2, input_sent_reps_doc_1_unmask,
          input_sent_reps_doc_2_unmask, output_sent_reps_doc_1,
          output_sent_reps_doc_2, siamese_loss, siamese_example_loss,
          siamese_logits)

def model_fn_builder(dual_encoder_config,
                     train_mode,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     use_one_hot_embeddings,
                     debugging=False):

  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    tf.logging.info("*** Current mode: %s ***" % mode)

    input_ids_1 = features["input_ids_1"]
    input_mask_1 = features["input_mask_1"]
    input_ids_2 = features["input_ids_2"]
    input_mask_2 = features["input_mask_2"]

    # to transfer these labels like 0/1/2 to binary labels like 0/1.
    documents_match_labels = features["documents_match_labels"]
    documents_match_labels = tf.cast(documents_match_labels > 0, tf.float32)
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if train_mode == constants.TRAIN_MODE_FINETUNE:
      masked_lm_positions_1 = tf.zeros([1])
      masked_lm_ids_1 = tf.zeros([1])
      masked_lm_weights_1 = tf.zeros([1])
      masked_lm_positions_2 = tf.zeros([1])
      masked_lm_ids_2 = tf.zeros([1])
      masked_lm_weights_2 = tf.zeros([1])
    else:
      masked_lm_positions_1 = features["masked_lm_positions_1"]
      masked_lm_ids_1 = features["masked_lm_ids_1"]
      masked_lm_weights_1 = features["masked_lm_weights_1"]
      masked_lm_positions_2 = features["masked_lm_positions_2"]
      masked_lm_ids_2 = features["masked_lm_ids_2"]
      masked_lm_weights_2 = features["masked_lm_weights_2"]
    
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(documents_match_labels), dtype=tf.float32)

    (masked_lm_loss_1, masked_lm_loss_2, masked_lm_example_loss_1,
    masked_lm_example_loss_2, masked_lm_weights_1, masked_lm_weights_2,
    masked_sent_lm_loss_1, masked_sent_lm_loss_2,
    masked_sent_per_example_loss_1, masked_sent_per_example_loss_2,
    masked_sent_weight_1, masked_sent_weight_2, seq_embed_1, seq_embed_2,
    input_sent_embed_1, input_sent_embed_2, output_sent_embed_1,
    output_sent_embed_2, siamese_loss, siamese_example_loss, siamese_logits) = \
    build_smith_dual_encoder(
          dual_encoder_config, train_mode, is_training, input_ids_1,
          input_mask_1, masked_lm_positions_1, masked_lm_ids_1,
          masked_lm_weights_1, input_ids_2, input_mask_2,
          masked_lm_positions_2, masked_lm_ids_2, masked_lm_weights_2,
          use_one_hot_embeddings, documents_match_labels, debugging)

    # 1. joint_train: combines the masked word LM losses for doc1/doc2 and the siamese matching loss.
    # 2. pretrain: only contains the masked word LM losses for doc1/doc2. (exclude NSP Loss)
    # 3. finetune: fine tune the model with loaded pretrained checkpoint only with the siamese matching loss.

    if train_mode == constants.TRAIN_MODE_JOINT_TRAIN:
      total_loss = masked_lm_loss_1 + masked_lm_loss_2 + siamese_loss
    elif train_mode == constants.TRAIN_MODE_PRETRAIN:
      total_loss = masked_lm_loss_1 + masked_lm_loss_2
    elif train_mode == constants.TRAIN_MODE_FINETUNE:
      total_loss = siamese_loss
    else:
      raise ValueError("Only joint_train, pretrain, finetune are supported.")

    # add the masked sentence LM losses for the two documents.
    if dual_encoder_config.encoder_config.use_masked_sentence_lm_loss:
      total_loss += (masked_sent_lm_loss_1 + masked_sent_lm_loss_2)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    init_checkpoint = dual_encoder_config.encoder_config.init_checkpoint
    # Load pretrained BERT checkpoints if there is a specified path.
    if init_checkpoint:
      tf.logging.info("**** Passed pretrained BERT checkpoint = %s ****", init_checkpoint)
      (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      if use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()
        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
              
    output_spec = None
    predicted_score = tf.sigmoid(siamese_logits)
    predicted_class = tf.round(predicted_score)

    _, prediction_dict = utils.get_export_outputs_prediction_dict_smith_de(
                          seq_embed_1, 
                          seq_embed_2, 
                          predicted_score, 
                          predicted_class,
                          documents_match_labels, 
                          input_sent_embed_1, 
                          input_sent_embed_2,
                          output_sent_embed_1, 
                          output_sent_embed_2)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:
      if (train_mode == constants.TRAIN_MODE_JOINT_TRAIN or train_mode == constants.TRAIN_MODE_PRETRAIN):
        eval_metrics = (metric_fns.metric_fn_pretrain, [
            masked_lm_example_loss_1, masked_lm_weights_1,
            masked_sent_per_example_loss_1, masked_sent_weight_1,
            masked_lm_example_loss_2, masked_lm_weights_2,
            masked_sent_per_example_loss_2, masked_sent_weight_2,
            predicted_class, documents_match_labels, is_real_example])
        
      elif train_mode == constants.TRAIN_MODE_FINETUNE:
        eval_metrics = (metric_fns.metric_fn_finetune, [
            predicted_class, documents_match_labels, siamese_example_loss,
            is_real_example])

      else:
        raise ValueError("Only joint_train, pretrain, finetune are supported.")
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(mode=mode, loss=total_loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=prediction_dict, scaffold_fn=scaffold_fn)

    else:
      raise ValueError("Only TRAIN, EVAL, PREDICT modes are supported: %s" %mode)
    return output_spec
  return model_fn
