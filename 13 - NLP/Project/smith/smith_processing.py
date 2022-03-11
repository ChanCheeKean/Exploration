
import sys
import os
sys.path.append(os.path.abspath(os.getcwd()) + '/smith/config')
import random
import collections
import tempfile
import tensorflow.compat.v1 as tf
from smith.bert import tokenization
import wiki_doc_pair_pb2
import tqdm
import pandas as pd
import nltk
# tf.app.flags.DEFINE_string('f', '', 'kernel')

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

def create_bytes_feature(values):
  feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
  return feature

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates masked tokens
  Arg:
  ============================
    token: single sentence block after adding CLS ans SEP token token

  Return:
  ============================
    output_tokens: full sentence block with masked token
    masked_lm_positions: position of masked token
    masked_lm_labels: True label of masked token
  """

  cand_indexes = []

  # avoid masking CLS or SEP
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  # rehuffle
  rng.shuffle(cand_indexes)
  output_tokens = list(tokens)
  num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
  masked_lms = []
  covered_indexes = set()

  # mask the first n tokens
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict: break
    if index in covered_indexes: continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original, # 10% of the time, replace with random word
      if rng.random() < 0.5: 
        masked_token = tokens[index]
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    # maskedinstance for to store label position and label
    MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)
  return (output_tokens, masked_lm_positions, masked_lm_labels)

def get_smith_model_tokens(input_text, tokenizer, sent_token_counter):
  """Generate tokens given an input text"""
  res_tokens = []
  for sent in nltk.tokenize.sent_tokenize(input_text):
    # remove empty sent
    if not sent:
      continue
    tokens = [w for w in tokenizer.tokenize(sent) if w]
    sent_token_counter[0] += 1  # Track number of sentences.
    sent_token_counter[1] += len(tokens)  # Track number of tokens.
    res_tokens.append(tokens)
  return (res_tokens, sent_token_counter)

def get_token_masks_paddings(block_tokens, max_sent_length_by_word, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, block_index):
  """Generates tokens, masks and paddings for the input block tokens at max length. 

    Arg:
    ============================
      token: single sentence block after adding CLS ans SEP token token

    Return:
    ============================
      tokens: full sentence block with CLS, SEP, PAD and masked token at max_seq_len
      segment_ids: list of 0
      masked_lm_positions: position of masked token
      masked_lm_labels: True label of masked token
      input_mask: 1 for word in token and 0 for PAD
      masked_lm_weights: 1 if there is label else 0 for PAD
  """

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_sent_length_by_word - 3

  # Truncates the sequence if sequence length is longer than max_num_tokens.
  if len(block_tokens) > max_num_tokens:
    block_tokens = block_tokens[0:max_num_tokens]

  # Add CLS
  tokens, segment_ids = [], []
  tokens_a = block_tokens
  tokens.append("[CLS]")
  segment_ids.append(0)

  # add SEP token
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  # generate masked token for each sentence block
  masked_lm_positions = []
  masked_lm_labels = []
  masked_lm_weights = []
  if max_predictions_per_seq > 0:
    (tokens, masked_lm_positions, masked_lm_labels) = \
    create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

  # Add [PAD] to tokens to max sentence length
  input_mask = [1] * len(tokens)
  while len(tokens) < max_sent_length_by_word:
    tokens.append("[PAD]")
    input_mask.append(0)
    segment_ids.append(0)

  assert len(tokens) == max_sent_length_by_word
  assert len(input_mask) == max_sent_length_by_word
  assert len(segment_ids) == max_sent_length_by_word

  # Transfer local positions in masked_lm_positions to global positions
  if max_predictions_per_seq > 0:
    masked_lm_positions = [(i + max_sent_length_by_word * block_index) for i in masked_lm_positions]
    masked_lm_weights = [1.0] * len(masked_lm_labels)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_labels.append("[PAD]")
      masked_lm_weights.append(0.0)
  return (tokens, segment_ids, masked_lm_positions, masked_lm_labels, input_mask, masked_lm_weights)

def get_tokens_segment_ids_masks(max_sent_length_by_word,
                                 max_doc_length_by_sentence, 
                                 doc_one_tokens,
                                 masked_lm_prob, 
                                 max_predictions_per_seq,
                                 vocab_words, 
                                 rng,
                                 tokenizer,
                                 greedy_sentence_filling=True):
  
  """Get the tokens, segment ids and masks of an input sequence.
    Arg:
  ============================
    doc_one_tokens: input documents tokens with the shape (num_sentences, num_words)

    Return:
  ============================
      tokens_doc: full doc tokens with CLS\SEP\PAD in size max_doc_length * max_sent_length
      token_ids_doc: Word Embedding for tokenized text
      segment_ids_doc: full 0 in size max_doc_length * max_sent_length
      masked_lm_positions_doc: position of masked token in size max_doc_length * max_predictions_per_seq
      masked_lm_labels_doc: True label of masked token in size max_doc_length * max_predictions_per_seq
      input_mask_doc: 1 for word in token and 0 for PAD in size max_doc_length * max_sent_length
      masked_lm_weights_doc: 1 if there is masked label else 0 for PAD in size max_doc_length * max_sent_length
      masked_lm_ids_doc: Word Embedding for masked label
  """

  sentence_num = len(doc_one_tokens)
  # sent_block_token_list is a 2D list to contain sentence block tokens.
  sent_block_token_list = []
  natural_sentence_index = -1

  while natural_sentence_index + 1 < sentence_num:
    natural_sentence_index += 1
    sent_tokens = doc_one_tokens[natural_sentence_index]
    if not sent_tokens:
      continue

    # Fill as many senteces as possible in the current sentence block in a greedy way.
    if greedy_sentence_filling:
      cur_sent_block_length = 0
      cur_sent_block = []

      while natural_sentence_index < sentence_num:
        cur_natural_sent_tokens = doc_one_tokens[natural_sentence_index]
        if not cur_natural_sent_tokens:
          natural_sentence_index += 1
          continue
        cur_sent_len = len(cur_natural_sent_tokens)

        # keep adding if current block lenth < max_sent_length_by_word
        # exception: current block is empty but sentence block going across the boundary
        # put this long natural sentence in the current sentence block and cut off later
        if ((cur_sent_block_length + cur_sent_len) <= (max_sent_length_by_word - 3)) or cur_sent_block_length == 0:
          cur_sent_block.extend(cur_natural_sent_tokens)
          cur_sent_block_length += cur_sent_len
          natural_sentence_index += 1
        else:
          # If cur_sent_block_length + cur_sent_len > max_sent_length_by_word-3
          # and the current sentence block is not empty, the sentence which
          # goes across the boundary will be put into the next sentence block.
          natural_sentence_index -= 1
          break
    sent_tokens = cur_sent_block
    sent_block_token_list.append(sent_tokens)

    # Skip more sentence blocks if the document is too long.
    if len(sent_block_token_list) >= max_doc_length_by_sentence:
      break

  # For each sentence block, generate the token sequences, masks and paddings.
  (tokens_doc, tokens_ids_doc, segment_ids_doc, input_mask_doc) = [[] for _ in range(4)]
  (masked_lm_positions_doc, masked_lm_labels_doc, masked_lm_weights_doc, masked_lm_ids_doc) = [[] for _ in range(4)]
  
  for block_index in range(len(sent_block_token_list)):
    tokens_block, segment_ids_block, masked_lm_positions_block, masked_lm_labels_block, input_mask_block, masked_lm_weights_block = \
    get_token_masks_paddings(sent_block_token_list[block_index], 
                             max_sent_length_by_word, 
                             masked_lm_prob, 
                             max_predictions_per_seq, 
                             vocab_words,
                             rng,
                             block_index)
    
    tokens_doc.extend(tokens_block)
    segment_ids_doc.extend(segment_ids_block)
    masked_lm_positions_doc.extend(masked_lm_positions_block)
    masked_lm_labels_doc.extend(masked_lm_labels_block)
    input_mask_doc.extend(input_mask_block)
    masked_lm_weights_doc.extend(masked_lm_weights_block)

  # Pad sentence blocks if the actual number of sentence blocks is less than max_doc_length_by_sentence.
  sentence_block_index = len(sent_block_token_list)

  while sentence_block_index < max_doc_length_by_sentence:
    for _ in range(max_sent_length_by_word):
      tokens_doc.append("[PAD]")
      segment_ids_doc.append(0)
      input_mask_doc.append(0)
    for _ in range(max_predictions_per_seq):
      masked_lm_positions_doc.append(0)
      masked_lm_labels_doc.append("[PAD]")
      masked_lm_weights_doc.append(0.0)
    sentence_block_index += 1

  assert len(tokens_doc) == max_sent_length_by_word * max_doc_length_by_sentence
  assert len(masked_lm_labels_doc) == max_predictions_per_seq * max_doc_length_by_sentence
  
  tokens_ids_doc = tokenizer.convert_tokens_to_ids(tokens_doc)
  masked_lm_ids_doc = tokenizer.convert_tokens_to_ids(masked_lm_labels_doc)
  
  return (tokens_doc, tokens_ids_doc, segment_ids_doc, masked_lm_positions_doc, 
  masked_lm_labels_doc, input_mask_doc, masked_lm_weights_doc, masked_lm_ids_doc)

class TrainingInstance(object):
  """A single training instance (sentence pair as dual encoder model inputs)."""

  def __init__(self,
               tokens_1,
               tokens_ids_1,
               segment_ids_1,
               masked_lm_positions_1,
               masked_lm_labels_1,
               input_mask_1,
               masked_lm_weights_1,
               masked_lm_ids_1,
               tokens_2,
               tokens_ids_2,
               segment_ids_2,
               masked_lm_positions_2,
               masked_lm_labels_2,
               input_mask_2,
               masked_lm_weights_2,
               masked_lm_ids_2,
               instance_id,
               documents_match_labels=-1.0):
               
    self.tokens_1 = tokens_1
    self.input_ids_1 = tokens_ids_1
    self.segment_ids_1 = segment_ids_1
    self.masked_lm_positions_1 = masked_lm_positions_1
    self.masked_lm_labels_1 = masked_lm_labels_1
    self.input_mask_1 = input_mask_1
    self.masked_lm_weights_1 = masked_lm_weights_1
    self.masked_lm_ids_1 = masked_lm_ids_1
    self.tokens_2 = tokens_2
    self.input_ids_2 = tokens_ids_2
    self.segment_ids_2 = segment_ids_2
    self.masked_lm_positions_2 = masked_lm_positions_2
    self.masked_lm_labels_2 = masked_lm_labels_2
    self.input_mask_2 = input_mask_2
    self.masked_lm_weights_2 = masked_lm_weights_2
    self.masked_lm_ids_2 = masked_lm_ids_2
    self.instance_id = instance_id
    self.documents_match_labels = documents_match_labels

  def __str__(self):
    s = ""
    s += "instance_id: %s\n" % self.instance_id
    s += "documents_match_labels: %s\n" % (str(self.documents_match_labels))
    s += "tokens_1: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens_1]))
    s += "input_ids_1: %s\n" % (" ".join([str(x) for x in self.input_ids_1]))
    s += "segment_ids_1: %s\n" % (" ".join([str(x) for x in self.segment_ids_1]))
    s += "masked_lm_positions_1: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions_1]))
    s += "masked_lm_labels_1: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_lm_labels_1]))
    s += "input_mask_1: %s\n" % (" ".join([str(x) for x in self.input_mask_1]))
    s += "masked_lm_weights_1: %s\n" % (" ".join([str(x) for x in self.masked_lm_weights_1]))
    s += "masked_lm_ids_1: %s\n" % (" ".join([str(x) for x in self.masked_lm_ids_1]))
    s += "tokens_2: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens_2]))
    s += "input_ids_2: %s\n" % (" ".join([str(x) for x in self.input_ids_2]))
    s += "segment_ids_2: %s\n" % (" ".join([str(x) for x in self.segment_ids_2]))
    s += "masked_lm_positions_2: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions_2]))
    s += "masked_lm_labels_2: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_lm_labels_2]))
    s += "input_mask_2: %s\n" % (" ".join([str(x) for x in self.input_mask_2]))
    s += "masked_lm_weights_2: %s\n" % (" ".join([str(x) for x in self.masked_lm_weights_2]))
    s += "masked_lm_ids_2: %s\n" % (" ".join([str(x) for x in self.masked_lm_ids_2]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()

def create_instance_from_wiki_doc_pair(instance_id, 
                                       doc_match_label,
                                       doc_one_tokens, 
                                       doc_two_tokens,
                                       max_sent_length_by_word,
                                       max_doc_length_by_sentence,
                                       masked_lm_prob, 
                                       max_predictions_per_seq,
                                       vocab_words,
                                       rng,
                                       tokenizer):
  
  """Creates `TrainingInstance`s for a WikiDocPair input data."""
  (tokens_1, tokens_ids_1, segment_ids_1, masked_lm_positions_1, masked_lm_labels_1, input_mask_1, masked_lm_weights_1, masked_lm_ids_1) = \
  get_tokens_segment_ids_masks(max_sent_length_by_word, max_doc_length_by_sentence, doc_one_tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, tokenizer)
  
  (tokens_2, tokens_ids_2, segment_ids_2, masked_lm_positions_2, masked_lm_labels_2, input_mask_2, masked_lm_weights_2, masked_lm_ids_2) = \
  get_tokens_segment_ids_masks(max_sent_length_by_word, max_doc_length_by_sentence, doc_two_tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, tokenizer)

  instance = TrainingInstance(
      tokens_1=tokens_1,
      tokens_ids_1=tokens_ids_1,
      segment_ids_1=segment_ids_1,
      masked_lm_positions_1=masked_lm_positions_1,
      masked_lm_labels_1=masked_lm_labels_1,
      input_mask_1=input_mask_1,
      masked_lm_weights_1=masked_lm_weights_1,
      masked_lm_ids_1=masked_lm_ids_1,
      tokens_2=tokens_2,
      tokens_ids_2=tokens_ids_2,
      segment_ids_2=segment_ids_2,
      masked_lm_positions_2=masked_lm_positions_2,
      masked_lm_labels_2=masked_lm_labels_2,
      input_mask_2=input_mask_2,
      masked_lm_weights_2=masked_lm_weights_2,
      masked_lm_ids_2=masked_lm_ids_2,
      instance_id=instance_id,
      documents_match_labels=doc_match_label)
  return instance

def ingest_tfrecord_raw(input_file):
  """Ingest WikiDocPair proto data."""

  wiki_doc_pair = wiki_doc_pair_pb2.WikiDocPair()
  instances = []
  sent_token_counter = [0, 0]
  lis = []

  for example in tqdm.tqdm(tf.python_io.tf_record_iterator(input_file)):
      doc_pair = wiki_doc_pair.FromString(example)
      doc_one_text = " \n\n\n\n\n\n ".join([a.text for a in doc_pair.doc_one.section_contents])
      doc_two_text = " \n\n\n\n\n\n ".join([a.text for a in doc_pair.doc_two.section_contents])
      doc_one_text = tokenization.convert_to_unicode(doc_one_text).strip()
      doc_two_text = tokenization.convert_to_unicode(doc_two_text).strip()

      if doc_pair.human_label_for_classification:
        doc_match_label = doc_pair.human_label_for_classification
      else:
        # Set the label as 0.0 if there are no available labels.
        doc_match_label = 0.0

      lis.append({
        'pair_id': doc_pair.id, 
        'doc_one': doc_one_text,
        'doc_two': doc_two_text,
        'doc_label': doc_match_label,
      })

  return pd.DataFrame(lis)

def add_features_for_one_doc(features, 
                             input_ids, 
                             tokens, 
                             segment_ids, 
                             input_mask, 
                             masked_lm_positions, 
                             masked_lm_labels, 
                             masked_lm_weights, 
                             masked_lm_ids,
                             tokenizer, 
                             doc_index):
                             
  """Add features for one document in a WikiDocPair example."""
  # input_ids = tokenizer.convert_tokens_to_ids(tokens)
  features["input_ids_" + doc_index] = create_int_feature(input_ids)
  features["input_mask_" + doc_index] = create_int_feature(input_mask)
  features["segment_ids_" + doc_index] = create_int_feature(segment_ids)

  if masked_lm_labels:
    # masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    features["masked_lm_positions_" + doc_index] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids_" + doc_index] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights_" + doc_index] = create_float_feature(masked_lm_weights)

def write_instance_to_example_files(instances, tokenizer, output_files='data/temp/processed_output'):
  """Create tf record files from TrainingInstance"""
  
  writers = []
  writers.append(tf.python_io.TFRecordWriter(output_files))
  writer_index = 0
  total_written = 0
  
  feature_list = []
  for (inst_index, instance) in enumerate(instances):
    features = collections.OrderedDict()

    add_features_for_one_doc(
        features=features,
        tokens=instance.tokens_1,
        input_ids=instance.input_ids_1, 
        segment_ids=instance.segment_ids_1,
        input_mask=instance.input_mask_1,
        masked_lm_positions=instance.masked_lm_positions_1,
        masked_lm_labels=instance.masked_lm_labels_1,
        masked_lm_weights=instance.masked_lm_weights_1,
        masked_lm_ids=instance.masked_lm_ids_1,
        tokenizer=tokenizer,
        doc_index="1")
    
    add_features_for_one_doc(
        features=features,
        tokens=instance.tokens_2,
        input_ids=instance.input_ids_2, 
        segment_ids=instance.segment_ids_2,
        input_mask=instance.input_mask_2,
        masked_lm_positions=instance.masked_lm_positions_2,
        masked_lm_labels=instance.masked_lm_labels_2,
        masked_lm_weights=instance.masked_lm_weights_2,
        masked_lm_ids=instance.masked_lm_ids_2,
        tokenizer=tokenizer,
        doc_index="2")
    
    # Adds fields on more content/id information of the current example.
    features["instance_id"] = create_bytes_feature([bytes(instance.instance_id, "utf-8")])
    features["tokens_1"] = create_bytes_feature([bytes(t, "utf-8") for t in instance.tokens_1])
    features["tokens_2"] = create_bytes_feature([bytes(t, "utf-8") for t in instance.tokens_2])
    # Adds the documents matching labels.
    features["documents_match_labels"] = create_float_feature([float(instance.documents_match_labels)])
    
    feature_list.append(features)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)
    total_written += 1
  
  for writer in writers:
    writer.close()

def transform_features(instances, tokenizer):
    '''convert lsit into TF.constant'''
    
    ins_key = [x for x in dir(instances[0]) if "__" not in x]
    features = {}
    for key in ins_key:
      features[key] = [getattr(instance, key) for instance in instances]
    # features["input_ids_1"] = [tokenizer.convert_tokens_to_ids(instance.tokens_1) for instance in instances]
    # features["input_ids_2"] = [tokenizer.convert_tokens_to_ids(instance.tokens_2) for instance in instances]
    # features["masked_lm_ids_1"] = [tokenizer.convert_tokens_to_ids(instance.masked_lm_labels_1) for instance in instances]
    # features["masked_lm_ids_2"] = [tokenizer.convert_tokens_to_ids(instance.masked_lm_labels_2) for instance in instances]
    
    except_list = ['masked_lm_weights_1', 'masked_lm_weights_2', 'documents_match_labels', \
                  'instance_id', 'tokens_1', 'tokens_2', 'masked_lm_labels_1', 'masked_lm_labels_2']

    for key in features.keys():
      if key not in except_list:
        features[key] = tf.constant(features[key], dtype=tf.int32)
      elif key not in ['instance_id', 'tokens_1', 'tokens_2', 'masked_lm_labels_1', 'masked_lm_labels_2']:
        features[key] = tf.constant(features[key], dtype=tf.float32)
      else:
        features[key] = tf.constant(features[key], dtype=tf.string)

    return features

'''
# config
vocab_file = 'pretrained_model/configs/vocab.txt'
input_file = 'data/small_demo_data.external_wdp.filtered_contro_wiki_cc_team.tfrecord'
output_file = 'data/gwiki_output.tfrecord'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
vocab_words = list(tokenizer.vocab.keys())
rng = random.Random(12345)
max_sent_length_by_word = 32
max_doc_length_by_sentence = 64
greedy_sentence_filling = True
max_predictions_per_seq = 5
masked_lm_prob = 0.15

# preprocessing full text
df_record = ingest_tfrecord_raw(input_file='data/small_demo_data.external_wdp.filtered_contro_wiki_cc_team.tfrecord')
df_record.to_csv('data/preprocess_wiki_doc_pair.csv', index=False)
instances = []
for i, row in df_record.iterrows():
  doc_one_tokens, _ = get_smith_model_tokens(row['doc_one'], tokenizer, [0, 0])
  doc_two_tokens, _ = get_smith_model_tokens(row['doc_two'], tokenizer, [0, 0])

  if not doc_one_tokens or not doc_two_tokens:
    continue

  instances.append(create_instance_from_wiki_doc_pair(
      instance_id=row['pair_id'], 
      doc_match_label=row['doc_label'], 
      doc_one_tokens=doc_one_tokens, 
      doc_two_tokens=doc_two_tokens,
      max_sent_length_by_word=max_sent_length_by_word, 
      max_doc_length_by_sentence=max_doc_length_by_sentence, 
      masked_lm_prob=masked_lm_prob,
      max_predictions_per_seq=max_predictions_per_seq, 
      vocab_words=list(tokenizer.vocab.keys()), 
      rng=rng
      ))

rng.shuffle(instances)
write_instance_to_example_files(instances=instances, 
                                tokenizer=tokenizer, 
                                output_files=output_file,
                                )
'''

# text_file = open("data/temp/single_sentence_block.txt", "r")
# test_tokens = text_file.read().split('\n')[:-1]

# with open('data/temp/doc_tokens.json') as f:
#   test_doc_tokens = json.load(f)

