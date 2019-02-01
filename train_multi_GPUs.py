

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from ffn.training.import_util import import_symbol
import time
from train import prepare_ffn, define_data_input, train_eval_size, get_batch, EvalTracker


import platform
if platform.system() == 'Windows':
    DIVSTR = '::'
elif platform.system() in ['Linux', 'Darwin']:
    DIVSTR = ':'
else:
    DIVSTR = ':'

FLAGS = flags.FLAGS

# Options related to training data.
flags.DEFINE_string('train_coords', None,
                    'Glob for the TFRecord of training coordinates.') # What's the use of TFRecord
flags.DEFINE_string('data_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing uint8 '
                    'image data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('label_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing int64 '
                    'label data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('model_name', None,
                    'Name of the model to train. Format: '
                    '[<packages>.]<module_name>.<model_class>, if packages is '
                    'missing "ffn.training.models" is used as default.')
flags.DEFINE_string('model_args', None,
                    'JSON string with arguments to be passed to the model '
                    'constructor.')

# Training infra options.
flags.DEFINE_string('train_dir', '/tmp',
                    'Path where checkpoints and other data will be saved.')
flags.DEFINE_string('master', '', 'Network address of the master.')
flags.DEFINE_integer('batch_size', 4, 'Number of images in a batch.')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of tasks in the ps job.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train for.')
flags.DEFINE_integer('replica_step_delay', 300,
                     'Require the model to reach step number '
                     '<replica_step_delay> * '
                     '<replica_id> before starting training on a given '
                     'replica.')
flags.DEFINE_integer('summary_rate_secs', 120,
                     'How often to save summaries (in seconds).')

# FFN training options.
flags.DEFINE_float('seed_pad', 0.05,
                   'Value to use for the unknown area of the seed.')
flags.DEFINE_float('threshold', 0.9,
                   'Value to be reached or exceeded at the new center of the '
                   'field of view in order for the network to inspect it.')
flags.DEFINE_enum('fov_policy', 'fixed', ['fixed', 'max_pred_moves'],
                  'Policy to determine where to move the field of the '
                  'network. "fixed" tries predefined offsets specified by '
                  '"model.shifts". "max_pred_moves" moves to the voxel with '
                  'maximum mask activation within a plane perpendicular to '
                  'one of the 6 Cartesian directions, offset by +/- '
                  'model.deltas from the current FOV position.')
# TODO(mjanusz): Implement fov_moves > 1 for the 'fixed' policy.
flags.DEFINE_integer('fov_moves', 1,
                     'Number of FOV moves by "model.delta" voxels to execute '
                     'in every dimension. Currently only works with the '
                     '"max_pred_moves" policy.')
flags.DEFINE_boolean('shuffle_moves', True,
                     'Whether to randomize the order of the moves used by the '
                     'network with the "fixed" policy.')

flags.DEFINE_float('image_mean', None,
                   'Mean image intensity to use for input normalization.')
flags.DEFINE_float('image_stddev', None,
                   'Image intensity standard deviation to use for input '
                   'normalization.')
flags.DEFINE_list('image_offset_scale_map', None,
                  'Optional per-volume specification of mean and stddev. '
                  'Every entry in the list is a colon-separated tuple of: '
                  'volume_label, offset, scale.')

flags.DEFINE_list('permutable_axes', ['1', '2'],
                  'List of integers equal to a subset of [0, 1, 2] specifying '
                  'which of the [z, y, x] axes, respectively, may be permuted '
                  'in order to augment the training data.')

flags.DEFINE_list('reflectable_axes', ['0', '1', '2'],
                  'List of integers equal to a subset of [0, 1, 2] specifying '
                  'which of the [z, y, x] axes, respectively, may be reflected '
                  'in order to augment the training data.')

FLAGS = flags.FLAGS


def run_training_step(sess, model, fetch_summary, feed_dict):
  """Runs one training step for a single FFN FOV."""
  ops_to_run = [model.train_op, model.global_step, model.logits]  # train_op defined in model.set_up_optimizer()

  if fetch_summary is not None:
    ops_to_run.append(fetch_summary)

  results = sess.run(ops_to_run, feed_dict)  # get prediction for the operation
  step, prediction = results[1:3]

  if fetch_summary is not None:
    summ = results[-1]
  else:
    summ = None

  return prediction, step, summ

def train_ffn(model_cls, **model_kwargs):
  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)):
      # The constructor might define TF ops/placeholders, so it is important
      # that the FFN is instantiated within the current context.
      # model instantiation!
      model = model_cls(**model_kwargs)
      eval_shape_zyx = train_eval_size(model).tolist()[::-1]

      eval_tracker = EvalTracker(eval_shape_zyx)
      load_data_ops = define_data_input(model, queue_batch=1)
      prepare_ffn(model)
      merge_summaries_op = tf.summary.merge_all()

      if FLAGS.task == 0:
        save_flags()
      # Setup the Higher Order session enviroment!
      summary_writer = None
      saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.25)
      scaffold = tf.train.Scaffold(saver=saver)
      with tf.train.MonitoredTrainingSession(
          master=FLAGS.master,
          is_chief=(FLAGS.task == 0),
          save_summaries_steps=None,
          save_checkpoint_secs=300,
          config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True),
          checkpoint_dir=FLAGS.train_dir,
          scaffold=scaffold) as sess:

        eval_tracker.sess = sess
        step = int(sess.run(model.global_step))  # evaluate the step
        # global_step is the tf inner mechanism that keeps the batches seen by
        # thus when revive a model from checkpoint the `step` can be restored
        if FLAGS.task > 0:
          # To avoid early instabilities when using multiple replicas, we use
          # a launch schedule where new replicas are brought online gradually.
          logging.info('Delaying replica start.')
          while step < FLAGS.replica_step_delay * FLAGS.task:
            time.sleep(5.0)
            step = int(sess.run(model.global_step))
        else:
          summary_writer = tf.summary.FileWriterCache.get(FLAGS.train_dir)
          summary_writer.add_session_log(
              tf.summary.SessionLog(status=tf.summary.SessionLog.START), step)
        # Prepare the batch_it object by input shifts parameters into it
        fov_shifts = list(model.shifts)  # x, y, z  vector collection formed by input deltas in 3*3*3-1 directions
        if FLAGS.shuffle_moves:
          random.shuffle(fov_shifts)

        policy_map = {
            'fixed': partial(fixed_offsets, fov_shifts=fov_shifts),
            'max_pred_moves': max_pred_offsets
        }  # delta, fov_shift come into policy map to
        batch_it = get_batch(lambda: sess.run(load_data_ops),
                             eval_tracker, model, FLAGS.batch_size,
                             policy_map[FLAGS.fov_policy])

        t_last = time.time()
        # Start major loop
        while not sess.should_stop() and step < FLAGS.max_steps:
          # Run summaries periodically.
          t_curr = time.time()
          if t_curr - t_last > FLAGS.summary_rate_secs and FLAGS.task == 0:
            summ_op = merge_summaries_op
            t_last = t_curr  # update at summary_rate_secs
          else:
            summ_op = None

          # Core lines to be modified w.r.t. Multi GPU computing
          seed, patches, labels, weights = next(batch_it)

          updated_seed, step, summ = run_training_step(
              sess, model, summ_op,
              feed_dict={
                  model.loss_weights: weights,
                  model.labels: labels,
                  model.input_patches: patches,
                  model.input_seed: seed,
              })

          # Save prediction results in the original seed array so that
          # they can be used in subsequent steps.
          mask.update_at(seed, (0, 0, 0), updated_seed)

          # Record summaries.
          if summ is not None:
            logging.info('Saving summaries.')
            summ = tf.Summary.FromString(summ)

            # Compute a loss over the whole training patch (i.e. more than a
            # single-step field of view of the network). This quantifies the
            # quality of the final object mask.
            summ.value.extend(eval_tracker.get_summaries())
            eval_tracker.reset()

            assert summary_writer is not None
            summary_writer.add_summary(summ, step)

      if summary_writer is not None:
        summary_writer.flush()


def main(argv=()):
  del argv  # Unused.
  model_class = import_symbol(FLAGS.model_name)
  # Multiply the task number by a value large enough that tasks starting at a
  # similar time cannot end up with the same seed.
  seed = int(time.time() + FLAGS.task * 3600 * 24)
  logging.info('Random seed: %r', seed)
  random.seed(seed)

  train_ffn(model_class, batch_size=FLAGS.batch_size,
            **json.loads(FLAGS.model_args))