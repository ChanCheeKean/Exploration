'''Configuration of Smith Model'''

class dictConveter(object):
    '''converting dict into instance attribute'''
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [dictConveter(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, dictConveter(b) if isinstance(b, dict) else b)

encoder_config_dict = {
      "encoder_config": {
      "model_name": "smith_dual_encoder",
      "init_checkpoint": "smith/pretrained_model/SMITHWPSP/model.ckpt-400000",
      "predict_checkpoint": "smith/pretrained_model/SMITHWPSP/model.ckpt-400000",
      "bert_config_file": "smith/config/sent_bert_4l_config.json",
      "doc_bert_config_file": "smith/config/doc_bert_3l_256h_config.json",
      "add_masks_lm": False,
      "max_predictions_per_seq": 0,
      "max_sent_length_by_word": 32,
      "max_doc_length_by_sentence": 64,
      "loop_sent_number_per_doc": 64,
      "sent_bert_trainable": True,
      "max_masked_sent_per_doc": 0,
      "use_masked_sentence_lm_loss": False,
      "doc_rep_combine_mode": "normal",
      "doc_rep_combine_attention_size": 256,},

  "train_eval_config": {
     "processed_file": "smith/data/processed_data_test.tfrecord",
     "model_output_dir": "smith/pretrained_model/export/",
     "pred_output_file": "smith/data/output/prediction_results.json",
     "learning_rate": 5e-5,
     "num_train_steps": 2,
     "num_warmup_steps": 1,
     "batch_size": 32,
     "save_checkpoints_steps": 10e9,
     "neg_to_pos_example_ratio": 1.0,
     "use_tpu": False,
     "tpu_name": None,
     "tpu_zone": None,
     "gcp_project": None,
     "master": None,
     "num_tpu_cores": 8,
     },
 
 "loss_config" : {
     "similarity_score_amplifier": 6.0},
}

# encoder_config = dictConveter(encoder_config_dict)