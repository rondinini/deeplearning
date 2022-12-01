import os
from pathlib import Path
from collections import Counter, defaultdict
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from architectures.rnn import create_agnews_model
from src.models.utils import callbacks


path = Path(os.path.dirname(os.path.realpath(__file__)))
model_save_fp = path / Path('trained_models')
train = True

# Loading data
datasets, info = tfds.load("ag_news_subset", as_supervised=True, with_info=True)
TRAIN_SIZE = int(120000 * 0.8)
train_dataset = datasets['train'].take(TRAIN_SIZE)
validation_dataset = datasets['train'].skip(TRAIN_SIZE)
test_dataset = datasets['test']

# Dataset elements are tuple of (x, y) elements
def regex_preprocess(x):
    x = tf.strings.regex_replace(x, b"<br\\s*/?>", b" ")
    x = tf.strings.regex_replace(x, b"[^a-zA-Z']", b" ")
    x = tf.strings.regex_replace(x, b" +", b" ")
    x = tf.strings.lower(x)
    x = tf.strings.split(x, sep=" ")
    return x

def bucketing(dataset: tf.data.Dataset, batch_size: int, drop_remainder: bool = True):
    bucket_boundaries = [(0, 20), (20, 30), (30, 40), (40, 60), (60, 90), (90, 120)]
    
    def filter_by_seq_length(tensor, timestep_low_bound, timestep_high_bound):
        return tf.cond(
            tf.shape(tensor)[0] >= timestep_low_bound and tf.shape(tensor)[0] < timestep_high_bound,
            lambda: True,
            lambda: False,
        )
    
    def padding(seq, desired_length):
        """
        Function to pad the sequence to the boundary length
        """
        seq_len = tf.shape(seq)[0]
        right_padding = desired_length - seq_len
        paddings = [[0, 0], [0, right_padding]]
        reshaped_seq = tf.reshape(seq, (1, seq_len))
        padded_seq = tf.pad(reshaped_seq, paddings, mode='CONSTANT', constant_values='<pad>')
        return tf.reshape(padded_seq, (desired_length, ))

    # Appending datasets with sequences of similar sizes in own bucket
    ds_buckets = []
    for bound in bucket_boundaries:
        low_bound = bound[0]
        high_bound = bound[1]
        ds = dataset.filter(lambda seq, label: filter_by_seq_length(seq, low_bound, high_bound))
        ds = ds.map(lambda seq, label: (padding(seq, high_bound), label), num_parallel_calls=tf.data.AUTOTUNE)
        ds_buckets.append(ds)
    
    # Padding to match sequence length in batch
    ds_buckets = [ds.batch(batch_size, drop_remainder=drop_remainder) for ds in ds_buckets]
    concat_dataset = ds_buckets[0]
    for dset in ds_buckets[1:]:
        concat_dataset = concat_dataset.concatenate(dset)
    return concat_dataset

def build_vocabulary(dataset: tf.data.Dataset, vocab_size: int, num_oov_buckets: int):
    vocabulary = Counter()
    for seq in dataset:
        vocabulary.update(seq.numpy())
    vocab_size = len(vocabulary)
    truncated_vocab = [word for word, count in vocabulary.most_common()][:vocab_size]
    words = tf.constant(truncated_vocab)
    word_ids = tf.range(len(truncated_vocab), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    return tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets=num_oov_buckets)

# Applying preprocess
train_dataset = train_dataset.map(lambda x, y: (regex_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(lambda x, y: (regex_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# Bucketing
batch_size = 32
train_dataset = bucketing(train_dataset, batch_size) # New shape of each element will be (batch_size, sequence_length)
validation_dataset = bucketing(validation_dataset, batch_size)

# Creating the vocabulary
train_text_ds = train_dataset.map(lambda seq, lab: seq, num_parallel_calls=tf.data.AUTOTUNE)
flattened_x_train_ds = train_text_ds.flat_map(tf.data.Dataset.from_tensor_slices)
vocab_size = 30000
num_oov_buckets = 8000
lookup_table = build_vocabulary(flattened_x_train_ds, vocab_size, num_oov_buckets)
train_dataset = train_dataset.map(lambda seq, lab: (lookup_table.lookup(seq), lab), num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(lambda seq, lab: (lookup_table.lookup(seq), lab), num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.shuffle(2000, reshuffle_each_iteration=True).repeat().prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.repeat().prefetch(tf.data.AUTOTUNE)

# Bulding and fitting the model
if train == True:
    model = create_agnews_model(input_shape=(None,), output_size=4, vocab_size=vocab_size+num_oov_buckets, embed_size=128)
    epochs = 50
    model.fit(x=train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks(), steps_per_epoch=3500, validation_steps=3500)
    model_name = model_save_fp / Path(model.name) 
    model.save(model_name)

else:
    model_name = model_save_fp / Path('agnews_cnn')
    model = tf.keras.models.load_model(model_name)

# Evaluating against test data
test_dataset = test_dataset.map(lambda x, y: (regex_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = bucketing(test_dataset, batch_size, False)
test_dataset = test_dataset.map(lambda seq, lab: (lookup_table.lookup(seq), lab), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

model_eval = model.evaluate(test_dataset)

pred_dataset = tf.data.Dataset.zip((test_dataset.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE), ))
y_true = np.array(list(test_dataset.map(lambda x, y: y, num_parallel_calls=tf.data.AUTOTUNE).unbatch().as_numpy_iterator()))

y_pred = model.predict(pred_dataset)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred)
print(cm)
cr = classification_report(y_true, y_pred)
print(cr) 




