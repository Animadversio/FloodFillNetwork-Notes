"""Builds a TFRecord file of coordinates for training.

Use ./compute_partitions.py to generate data for --partition_volumes.
Note that the volume names you provide in --partition_volumes will
have to match the volume labels you pass to the training script.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from absl import app
from absl import flags
from absl import logging

import h5py
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_list('partition_volumes', None,
                  'Partition volumes as '
                  '<volume_name>:<volume_path>:<dataset>, where volume_path '
                  'points to a HDF5 volume, and <volume_name> is an arbitrary '
                  'label that will have to also be used during training.')
flags.DEFINE_string('coordinate_output', None,
                    'Path to a TF Record file in which to save the '
                    'coordinates.')
flags.DEFINE_list('margin', None, '(z, y, x) tuple specifying the '
                  'number of voxels adjacent to the border of the volume to '
                  'exclude from sampling. This should normally be set to the '
                  'radius of the FFN training FoV (i.e. network FoV radius '
                  '+ deltas.')


IGNORE_PARTITION = 0


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
  del argv  # Unused.

  totals = defaultdict(int)  # partition -> voxel count (len of the indices of the same key )
  indices = defaultdict(list)  # partition -> [(vol_id, 1d index), ]
  # Note: shared index among different partition volumes are allowed and merged together
  vol_labels = []
  vol_shapes = []
  mz, my, mx = [int(x) for x in FLAGS.margin]  # marginal data is sliced off when importing data from file

  for i, partvol in enumerate(FLAGS.partition_volumes):
    name, path, dataset = partvol.split('::')
    with h5py.File(path, 'r') as f:
      partitions = f[dataset][mz:-mz, my:-my, mx:-mx]  # FIXME:input 0 induce error here! partitions will be none output, can use _sel() trick
      vol_shapes.append(partitions.shape)
      vol_labels.append(name)  # Name of the total volume

      uniques, counts = np.unique(partitions, return_counts=True)
      for val, cnt in zip(uniques, counts):  # val are the uint8 markers marked in compute_partition (not label)
        if val == IGNORE_PARTITION:
          continue

        totals[val] += cnt  #
        indices[val].extend(
            [(i, flat_index) for flat_index in  # indices here is tuple of volume #i and #flat_index in the volume
             np.flatnonzero(partitions == val)])  # FIXME: This line can induce MEMORYERROR
        # Last line is one-by-one searching for `val` labelled entry in the partitions tensor, and use `np.flatnonzero`
        # to get the flat index in the huge tensor. (Don't know if it is time costing)
  logging.info('Partition counts:')
  for k, v in totals.items():
    logging.info(' %d: %d', k, v)  # the voxel number for different labels

  logging.info('Resampling and shuffling coordinates.')

  max_count = max(totals.values())  # voxel number maximum in kinds
  indices = np.concatenate(
      [np.resize(np.random.permutation(v), (max_count, 2)) for
       v in indices.values()], axis=0)   # FIXME: This line can induce MEMORYERROR,
  np.random.shuffle(indices)  # NoteL This line is also the most time consumming
  # Note: Multi-dimensional arrays are only shuffled along the first axis
  logging.info('Saving coordinates.')  # Saving takes the most of time!!
  record_options = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)
  with tf.python_io.TFRecordWriter(FLAGS.coordinate_output,
                                   options=record_options) as writer:
    for i, coord_idx in indices:  # note `i` is the No. of volume
      z, y, x = np.unravel_index(coord_idx, vol_shapes[i])  # get z,y,x value back from the flat coordinate of 1d array

      coord = tf.train.Example(features=tf.train.Features(feature=dict(
          center=_int64_feature([mx + x, my + y, mz + z]),
          label_volume_name=_bytes_feature(vol_labels[i].encode('utf-8'))
      )))
      writer.write(coord.SerializeToString())  # Write the file serialized coord by coord.
  logging.info('Saving completed!')

if __name__ == '__main__':
  flags.mark_flag_as_required('margin')
  flags.mark_flag_as_required('coordinate_output')
  flags.mark_flag_as_required('partition_volumes')

  app.run(main)
