from collections import Counter
import tensorflow as tf

def regex_preprocess(x: tf.Tensor):
    """ Function to remove undesired chars from strings

    Args:
        x (tf.Tensor): input string

    Returns:
        tf.Tensor: preprocessed string
    """    
    x = tf.strings.regex_replace(x, b"<br\\s*/?>", b" ")
    x = tf.strings.regex_replace(x, b"[^a-zA-Z']", b" ")
    x = tf.strings.regex_replace(x, b" +", b" ")
    x = tf.strings.lower(x)
    return tf.strings.split(x, sep=" ")

def filter_by_seq_length(tensor: tf.Tensor, timestep_low_bound: int, timestep_high_bound: int):
    """ Input function of tf.Dataset.filter, that returns True if the sequence length falls within
    the timesteps boundaries and False otherwise.

    Args:
        tensor (tf.Tensor): sequence
        timestep_low_bound (int): lower bound value for timesteps
        timestep_high_bound (int): higher bound value for timesteps

    Returns:
        _type_: _description_
    """    
    return tf.cond(
        tf.shape(tensor)[0] >= timestep_low_bound and tf.shape(tensor)[0] < timestep_high_bound,
        lambda: True,
        lambda: False,
    )

def padding(seq: tf.Tensor, desired_length: int):
    """ Function that pads a sequence to match the desired length.

    Eg. if seq has shape (30, ) and desired_length is 35, the output will be (35, )
    where the last 5 values will be <pad> values.

    Args:
        seq (tf.Tensor): sequence
        desired_length (int): desired sequence length after padding

    Returns:
        _type_: _description_
    """    
    seq_len = tf.shape(seq)[0]
    right_padding = desired_length - seq_len
    paddings = [[0, 0], [0, right_padding]]
    reshaped_seq = tf.reshape(seq, (1, seq_len))
    padded_seq = tf.pad(reshaped_seq, paddings, mode='CONSTANT', constant_values='<pad>')
    return tf.reshape(padded_seq, (desired_length, ))

def bucketing(dataset: tf.data.Dataset, bucket_boundaries: list, batch_size: int, drop_remainder: bool = True):
    """ Function to group sequences of variable lengths into buckets of different sizes.
    This is useful for training models with variable timesteps lengths, as the length of the sequence must
    be constant within the same batch (but not across different batches).

    The batch size must also be specified, and it will be applied to each bucket.

    Args:
        dataset (tf.data.Dataset): input tf Dataset
        bucket_boundaries (list): list of tuples (eg. [(0, 20), (20, 30)])
        batch_size (int): batch size of each bucket
        drop_remainder (bool, optional): True to drop remainder while batching

    Returns:
        tf.data.Dataset: batched tf Dataset, with shape (batch_size, bucket_sequence_length)
    """    
    # Appending datasets with sequences of similar sizes in own bucket
    ds_buckets = []
    for bound in bucket_boundaries:
        low_bound, high_bound = bound
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
    """ Function that builds a vocabulary for NLP models.

    Args:
        dataset (tf.data.Dataset): input tf Dataset with unique words
        vocab_size (int): maximum size of the vocabulary
        num_oov_buckets (int): size of the OOV buckets

    Returns:
        tf StaticVocabularyTable: Lookup table that maps words with word ids
    """    
    vocabulary = Counter()
    for seq in dataset:
        vocabulary.update(seq.numpy())
    vocab_size = len(vocabulary)
    truncated_vocab = [word for word, count in vocabulary.most_common()][:vocab_size]
    words = tf.constant(truncated_vocab)
    word_ids = tf.range(len(truncated_vocab), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    return tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets=num_oov_buckets)