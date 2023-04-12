"""
Speaker verification pipeline
"""
import tensorflow as tf
import numpy as np
import os
def make_tfrecord(  fname,
                    feature):
    """
    Make tfrecord
    """
    timesteps, dim_feat = feature.shape
    feature = feature.reshape([-1])
    with tf.io.TFRecordWriter(fname) as writer:
        timesteps   = tf.train.Feature(int64_list = tf.train.Int64List(value = [timesteps]))
        dim_feat    = tf.train.Feature(int64_list = tf.train.Int64List(value = [dim_feat]))
        feature     = tf.train.Feature(float_list = tf.train.FloatList(value = feature))

        context = tf.train.Features(feature = {
                "timesteps" : timesteps,
                "dim_feat"  : dim_feat,
            })

        feature_lists = tf.train.FeatureLists(feature_list={
                "feature": tf.train.FeatureList(feature = [feature]),
            })

        seq_example = tf.train.SequenceExample( # context and feature_lists
            context = context,
            feature_lists = feature_lists,
        )

        serialized = seq_example.SerializeToString()
        writer.write(serialized)

def parser(example_proto):
    """
    Create a description of the features.
    """
    context_features = {
        'timesteps' : tf.io.FixedLenFeature([], tf.int64),
        'dim_feat'  : tf.io.FixedLenFeature([], tf.int64),
    }

    sequence_features = {
        'feature'      : tf.io.VarLenFeature(tf.float32),
    }
    context_parsed, seq_parsed = tf.io.parse_single_sequence_example(
            example_proto,
            context_features  = context_features,
            sequence_features = sequence_features,
                                        )

    dim_feat = tf.cast(context_parsed['dim_feat'], tf.int32)

    feature = tf.sparse.to_dense(seq_parsed['feature'])
    feature = tf.reshape(feature, [-1, dim_feat])


    mask = feature[:,0] * 0 + 1
    mask = mask[..., tf.newaxis]
    mask = tf.cast(mask, tf.float32)

    return feature, mask

def tfrecords_pipeline(
        fnames,
        num_prons,
        num_noisetype,
        num_group_ppls):
    """_summary_

    Args:
        fnames (string): filenames
        num_prons (int): number of pronounciations for each person
        num_noisetype (int): number of noise types
        num_group_ppls (int): number of people for mini-batch
    """
    def decode_tfrecord(tfrecord):
        print(tfrecord)
        tfrecord = parser(tfrecord)
        return tfrecord
    def mapping_prons(dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.shuffle(num_noisetype)
        print(dataset)
        dataset = tf.data.TFRecordDataset(dataset)
        return dataset
    def mapping_ppl(dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.interleave(
            mapping_prons,
            cycle_length=num_prons,
            block_length=1,
            deterministic=True,
            num_parallel_calls = tf.data.AUTOTUNE)
        return dataset
    dataset = tf.data.Dataset.from_tensor_slices(fnames)
    dataset = dataset.shuffle(len(fnames), reshuffle_each_iteration=True)
    dataset = dataset.interleave(
        mapping_ppl,
        cycle_length=len(fnames),
        block_length=num_prons,
        deterministic=True,
        num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.map(
                decode_tfrecord,
                num_parallel_calls = tf.data.AUTOTUNE,
                deterministic = True)
    dataset = dataset.batch(num_group_ppls*num_prons)
    dataset = dataset.prefetch(1)
    iterator = iter(dataset)
    return iterator, dataset

if __name__ == "__main__":
    num_prons = 4 # pylint: disable=invalid-name
    num_noisetype = 3 # pylint: disable=invalid-name
    num_group_ppls = 2 # pylint: disable=invalid-name
    dim_feat=2
    timesteps = 6
    os.makedirs("fake_tfrecord", exist_ok=True)

    for ppl in range(1,5):
        for sent in range(1, num_prons+1):
            for ntype in range(1, num_noisetype+1):
                fname = f"fake_tfrecord/id{ppl}_sent{sent}_noise{ntype}.tfrecord"
                feat = np.ones((dim_feat, timesteps), dtype=np.float32) * (ppl+0.1*sent+0.01*ntype)
                make_tfrecord(fname, feat)
    fnames =  [[['fake_tfrecord/id1_sent1_noise1.tfrecord', 'fake_tfrecord/id1_sent1_noise2.tfrecord', 'fake_tfrecord/id1_sent1_noise3.tfrecord'],
                ['fake_tfrecord/id1_sent2_noise1.tfrecord', 'fake_tfrecord/id1_sent2_noise2.tfrecord', 'fake_tfrecord/id1_sent2_noise3.tfrecord'],
                ['fake_tfrecord/id1_sent3_noise1.tfrecord', 'fake_tfrecord/id1_sent3_noise2.tfrecord', 'fake_tfrecord/id1_sent3_noise3.tfrecord'],
                ['fake_tfrecord/id1_sent4_noise1.tfrecord', 'fake_tfrecord/id1_sent4_noise2.tfrecord', 'fake_tfrecord/id1_sent4_noise3.tfrecord']], # 1st person
               [['fake_tfrecord/id2_sent1_noise1.tfrecord', 'fake_tfrecord/id2_sent1_noise2.tfrecord', 'fake_tfrecord/id2_sent1_noise3.tfrecord'],
                ['fake_tfrecord/id2_sent2_noise1.tfrecord', 'fake_tfrecord/id2_sent2_noise2.tfrecord', 'fake_tfrecord/id2_sent2_noise3.tfrecord'],
                ['fake_tfrecord/id2_sent3_noise1.tfrecord', 'fake_tfrecord/id2_sent3_noise2.tfrecord', 'fake_tfrecord/id2_sent3_noise3.tfrecord'],
                ['fake_tfrecord/id2_sent4_noise1.tfrecord', 'fake_tfrecord/id2_sent4_noise2.tfrecord', 'fake_tfrecord/id2_sent4_noise3.tfrecord']], # 2st person
               [['fake_tfrecord/id3_sent1_noise1.tfrecord', 'fake_tfrecord/id3_sent1_noise2.tfrecord', 'fake_tfrecord/id3_sent1_noise3.tfrecord'],
                ['fake_tfrecord/id3_sent2_noise1.tfrecord', 'fake_tfrecord/id3_sent2_noise2.tfrecord', 'fake_tfrecord/id3_sent2_noise3.tfrecord'],
                ['fake_tfrecord/id3_sent3_noise1.tfrecord', 'fake_tfrecord/id3_sent3_noise2.tfrecord', 'fake_tfrecord/id3_sent3_noise3.tfrecord'],
                ['fake_tfrecord/id3_sent4_noise1.tfrecord', 'fake_tfrecord/id3_sent4_noise2.tfrecord', 'fake_tfrecord/id3_sent4_noise3.tfrecord']], # 3st person
               [['fake_tfrecord/id4_sent1_noise1.tfrecord', 'fake_tfrecord/id4_sent1_noise2.tfrecord', 'fake_tfrecord/id4_sent1_noise3.tfrecord'],
                ['fake_tfrecord/id4_sent2_noise1.tfrecord', 'fake_tfrecord/id4_sent2_noise2.tfrecord', 'fake_tfrecord/id4_sent2_noise3.tfrecord'],
                ['fake_tfrecord/id4_sent3_noise1.tfrecord', 'fake_tfrecord/id4_sent3_noise2.tfrecord', 'fake_tfrecord/id4_sent3_noise3.tfrecord'],
                ['fake_tfrecord/id4_sent4_noise1.tfrecord', 'fake_tfrecord/id4_sent4_noise2.tfrecord', 'fake_tfrecord/id4_sent4_noise3.tfrecord']]] # 4st person
    iter, dataset = tfrecords_pipeline(
        fnames,
        num_prons,
        num_noisetype,
        num_group_ppls)
    for d in dataset:
        print(d[0].numpy())
        break