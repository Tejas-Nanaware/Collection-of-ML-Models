# Using Tensorflow v2.1.0

from os import path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

# Load Dataset
DATA_PATH = './'
df_train = pd.read_csv(path.join(DATA_PATH, 'train_cleaned.csv'), index_col='id')
df_test = pd.read_csv(path.join(DATA_PATH, 'test_cleaned.csv'), index_col='id')

#Setting up constants
MAX_LEN = 256

# Load BERT from Google Repository
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2', trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# Try tokenizing words and get token IDS
tokenizer.tokenize(df_train['text_cleaned'].iloc[0])
tokenizer.convert_tokens_to_ids(tokenizer.tokenize(df_train['text_cleaned'].iloc[0]))

# Encode function for the data
def encode_tweets(tweets, max_len=MAX_LEN):
  tokens = []
  masks = []
  segments = []

  for tweet in tweets:
    tweet = tokenizer.tokenize(tweet)
    tweet = tweet[:max_len - 2]
    input_seq = ['[CLS]'] + tweet + ['[SEP]']

    pad_seq = max_len - len(input_seq)

    token = tokenizer.convert_tokens_to_ids(input_seq)
    token += [0] * pad_seq
    
    mask = [1] * len(input_seq) + [0] * pad_seq

    segment = [0] * max_len

    tokens.append(token)
    masks.append(mask)
    segments.append(segment)

  return np.array(tokens), np.array(masks), np.array(segments)

# Encode Train and Test Data
train_tokens, train_masks, train_segments = encode_tweets(df_train['text_cleaned'])
test_encoded = encode_tweets(df_test['text_cleaned'])

# Build Neural Network using BERT
def build_bert_model(max_seq_length=MAX_LEN):
  input_word_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
  input_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
  segment_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
  pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

  bert_output = sequence_output[:, 0, :]

  additional_layers = layers.Reshape((1, 1024))(bert_output)
  additional_layers = layers.LSTM(300, return_sequences=True)(additional_layers)
  additional_layers = layers.LSTM(200, return_sequences=True)(additional_layers)
  additional_layers = layers.LSTM(100, return_sequences=True)(additional_layers)
  additional_layers = layers.LSTM(100)(additional_layers)
  additional_layers = layers.Dense(1, activation='sigmoid')(additional_layers)

  model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=additional_layers)

  model.compile(optimizer=Adam(lr=0.001), metrics=['accuracy'], loss=binary_crossentropy)

  return model

# Create the model and fit it
nn = build_bert_model()
nn.summary()
nn.fit([train_tokens, train_masks, train_segments], df_train['target'], validation_split=0.2, epochs=10, batch_size=32)


# Create submission ready test file
test_probs = nn.predict(test_encoded)
test_target = test_probs.argmax(axis=-1)
submission = pd.DataFrame({'id': df_test.index, 'target': test_target.flatten()})
submission.to_csv(path.join(DATA_PATH, 'submission_bert.csv'), index=False)