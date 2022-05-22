import tensorflow as tf

class NSynthDataset(object):
    """Dataset object to help manage the TFRecord loading."""

    def __init__(self, tfrecord_path):
        self.record_path = tfrecord_path
        self.features = {
            "note": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
            "note_str": tf.io.FixedLenFeature([], dtype=tf.string, default_value=b"test"),
            "instrument": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
            "instrument_str": tf.io.FixedLenFeature([], dtype=tf.string, default_value=b"test"),
            "pitch": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
            "velocity": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
            "sample_rate": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
            "audio": tf.io.FixedLenFeature([64000], dtype=tf.float32, default_value=[0] * 64000),
            "qualities": tf.io.FixedLenFeature([10], dtype=tf.int64, default_value=[0] * 10),
            "qualities_str": tf.io.VarLenFeature(dtype=tf.string),
            "instrument_family": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
            "instrument_family_str": tf.io.FixedLenFeature([], dtype=tf.string, default_value=b"test"),
            "instrument_source": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=0),
            "instrument_source_str": tf.io.FixedLenFeature([], dtype=tf.string, default_value=b"test"),
        }

    def parse(self, example):
        return tf.io.parse_single_example(example, self.features)

    def get_dataset(self):
        raw_dataset = tf.data.TFRecordDataset([self.record_path])
        return raw_dataset.map(self.parse)
