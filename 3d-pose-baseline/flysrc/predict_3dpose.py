
"""Predicting 3d poses from 2d joints"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import copy

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import viz
import cameras
import data_utils
import linear_model

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 1, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 200, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("camera_frame", False, "Convert 3d poses to camera coordinates")
tf.app.flags.DEFINE_boolean("max_norm", False, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", False, "Use batch_normalization")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", False, "Whether to add a residual connection every 2 layers")

# Preprocessing
tf.app.flags.DEFINE_boolean("superimpose", False, "Superimpose data from different files")
tf.app.flags.DEFINE_boolean("change_origin", False, "Change the origin of the system to ROOT_POSITION")
tf.app.flags.DEFINE_boolean("procrustes", False, "Number of the file to use as procrustes ground truth")
tf.app.flags.DEFINE_boolean("lowpass", False, "Whether to add low-pass filter to 3d data")

# Directories
tf.app.flags.DEFINE_string("train_dir", "tr12te3", "Training directory.")

# Train or load
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

# Misc
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

train_dir = FLAGS.train_dir
if FLAGS.camera_frame:
  train_dir += "_camproj"
if FLAGS.superimpose:
  train_dir += "_superimp"
if FLAGS.change_origin:
  train_dir += "_neworig"
if FLAGS.procrustes:
  train_dir += "_procrustes"
if FLAGS.lowpass:
  train_dir += "_lowpass"
train_dir += "_"+str(FLAGS.learning_rate)

print("\n\n[*] training directory: ", train_dir)
summaries_dir = os.path.join(train_dir, "log") # Directory for TB summaries

# To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
os.system('mkdir -p {}'.format(summaries_dir))

def create_model( session, batch_size ):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """
  model = linear_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      FLAGS.change_origin,
      summaries_dir,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

  if FLAGS.load <= 0:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
    return model

  # Load a previously saved model
  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific cpixels = pixels / pixels[2,:]heckpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model

def train():
  """Train a linear model for 3d pose estimation"""

  # Load camera parameters
  rcams = cameras.load_cameras()
  
  print("Cameras dic:")
  for k in rcams.keys():
    print(k)

  # Load 3d data and 2d projections
  full_train_set_3d, full_test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d =\
    data_utils.read_3d_data( FLAGS.camera_frame, rcams, FLAGS.superimpose, FLAGS.change_origin,
    FLAGS.procrustes, FLAGS.lowpass )
  
  # Read stacked hourglass 2D predictions
  full_train_set_2d, full_test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = \
    data_utils.read_2d_predictions( FLAGS.change_origin )
  
  print("\n[+] done reading and normalizing data")
  tr_subj = 0
  for v in full_train_set_3d.values():
    tr_subj += v.shape[0]
  te_subj = 0
  for v in full_test_set_3d.values():
    te_subj += v.shape[0]
  print("{0} training subjects, {1} test subjects".format(tr_subj, te_subj))
  
  unNorm_ftrs2d = data_utils.unNormalize_dic(full_train_set_2d, data_mean_2d, data_std_2d, dim_to_use_2d)
  unNorm_ftrs3d = data_utils.unNormalize_dic(full_train_set_3d, data_mean_3d, data_std_3d, dim_to_use_3d)
  unNorm_ftes3d = data_utils.unNormalize_dic(full_test_set_3d, data_mean_3d, data_std_3d, dim_to_use_3d)

  viz.visualize_train_sample(unNorm_ftrs2d, unNorm_ftrs3d, FLAGS.camera_frame)
  viz.visualize_files_animation(unNorm_ftrs3d, unNorm_ftes3d)
  
  train_set_2d = {}
  test_set_2d = {}
  train_set_3d = {}
  test_set_3d = {}
  for k in full_train_set_3d:
    (f, c) = k
    train_set_3d[k] = full_train_set_3d[k][:, dim_to_use_3d]
    train_set_2d[(f, data_utils.CAMERA_TO_USE)] =\
       full_train_set_2d[(f, data_utils.CAMERA_TO_USE)][:, dim_to_use_2d]
  for k in full_test_set_3d:
    (f, c) = k
    test_set_3d[k] = full_test_set_3d[k][:, dim_to_use_3d]
    test_set_2d[(f, data_utils.CAMERA_TO_USE)] =\
       full_test_set_2d[(f, data_utils.CAMERA_TO_USE)][:, dim_to_use_2d]
  
  print("3D data mean:")
  print(data_mean_3d)
  print("3D data std:")
  print(data_std_3d)

  print("2D data mean:")
  print(data_mean_2d)
  print("2D data std:")
  print(data_std_2d)
  
  input("Press Enter to continue...")

  # Avoid using the GPU if requested
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    # === Create the model ===
    print("[*] creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model( sess, FLAGS.batch_size )
    model.train_writer.add_graph( sess.graph )
    print("[+] model created")
    
    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    current_epoch = 0
    log_every_n_batches = 20

    for _ in range( FLAGS.epochs ):
      current_epoch = current_epoch + 1

      # === Load training batches for one epoch ===
      encoder_inputs, decoder_outputs =\
         model.get_all_batches( train_set_2d, train_set_3d, FLAGS.camera_frame, training=True )
      nbatches = len( encoder_inputs )
      print("[*] there are {0} train batches".format( nbatches ))
      start_time, loss = time.time(), 0.

      # === Loop through all the training batches ===
      for i in range( nbatches ):

        if (i+1) % log_every_n_batches == 0:
          # Print progress every log_every_n_batches batches
          print("Working on epoch {0}, batch {1} / {2}...".format( current_epoch, i+1, nbatches),end="" )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        step_loss, loss_summary, lr_summary, _ =\
          model.step( sess, enc_in, dec_out, FLAGS.dropout, isTraining=True )

        if (i+1) % log_every_n_batches == 0:
          # Log and print progress every log_every_n_batches batchespixels = pixels / pixels[2,:]
          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / log_every_n_batches ) )

        loss += step_loss
        current_step += 1
        # === end looping through training batches ===

      loss = loss / nbatches
      print("=============================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (model.global_step.eval(),
            model.learning_rate.eval(), loss) )
      # === End training for an epoch ===

      # === Testing after this epoch ===
      isTraining = False

      n_joints = len(data_utils.DIMENSIONS_TO_USE)
      encoder_inputs, decoder_outputs =\
         model.get_all_batches( test_set_2d, test_set_3d, FLAGS.camera_frame, training=False)

      total_err, joint_err, step_time, loss = evaluate_batches( sess, model,
        data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
        data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
        current_step, encoder_inputs, decoder_outputs, current_epoch )

      print("=============================\n"
            "Step-time (ms):      %.4f\n"
            "Val loss avg:        %.4f\n"
            "Val error avg (mm):  %.2f\n"
            "=============================" % ( 1000*step_time, loss, total_err ))

      for i in range(n_joints):
        # 6 spaces, right-aligned, 5 decimal places
        print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i+1, joint_err[i]))
      print("=============================")

      # Log the error to tensorboard
      summaries = sess.run( model.err_mm_summary, {model.err_mm: total_err} )
      model.test_writer.add_summary( summaries, current_step )

      # Save the model
      print( "Saving the model... ", end="" )
      start_time = time.time()
      model.saver.save(sess, os.path.join(train_dir, 'checkpoint'), global_step=current_step )
      print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) )

      # Reset global time and loss
      step_time, loss = 0, 0

      sys.stdout.flush()

def evaluate_batches( sess, model,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
  current_step, encoder_inputs, decoder_outputs, current_epoch=0 ):
  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.

  Args
    sess
    model
    data_mean_3d
    data_std_3d
    dim_to_use_3d
    dim_to_ignore_3d
    data_mean_2d
    data_std_2d
    dim_to_use_2d
    dim_to_ignore_2d
    current_step
    encoder_inputs
    decoder_outputs
    current_epoch
  Returns

    total_err
    joint_err
    step_time
    loss
  """

  n_joints = len( data_utils.DIMENSIONS_TO_USE )
  nbatches = len( encoder_inputs )

  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 20

  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )

    enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
    dp = 1.0 # dropout keep probability is always 1 at test time
    step_loss, loss_summary, poses3d = model.step( sess, enc_in, dec_out, dp, isTraining=False )
    loss += step_loss

    # denormalize
    enc_in  = data_utils.unNormalize_batch( enc_in,  data_mean_2d, data_std_2d, dim_to_use_2d )
    dec_out = data_utils.unNormalize_batch( dec_out, data_mean_3d, data_std_3d, dim_to_use_3d )
    poses3d = data_utils.unNormalize_batch( poses3d, data_mean_3d, data_std_3d, dim_to_use_3d )

    # Keep only the relevant dimensions
    dec_out = dec_out[:, dim_to_use_3d]
    poses3d = poses3d[:, dim_to_use_3d]

    assert dec_out.shape[0] == FLAGS.batch_size
    assert poses3d.shape[0] == FLAGS.batch_size

    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out)**2 # Squared error between prediction and expected output
    dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, n_joints*3, 3):
      # Sum across X,Y, and Z dimenstions to obtain L2 distance
      dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
      dist_idx += 1

    all_dists.append(dists)
    assert sqerr.shape[0] == FLAGS.batch_size

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )

  return total_err, joint_err, step_time, loss


def sample():
  """Get samples from a model and visualize them"""

  # Load camera parameters
  rcams = cameras.load_cameras()

  # Load 3d data and 2d projections
  full_train_set_3d, full_test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d =\
    data_utils.read_3d_data( FLAGS.camera_frame, rcams, FLAGS.superimpose, FLAGS.change_origin,
    FLAGS.procrustes, FLAGS.lowpass )

  # Read stacked hourglass 2D predictions
  full_train_set_2d, full_test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = \
    data_utils.read_2d_predictions( FLAGS.change_origin )

  print("[+] done reading and normalizing data")
  
  train_set_2d = {}
  test_set_2d = {}
  train_set_3d = {}
  test_set_3d = {}
  for k in full_train_set_3d:
    (f, c) = k
    train_set_3d[k] = full_train_set_3d[k][:, dim_to_use_3d]
    train_set_2d[(f, data_utils.CAMERA_TO_USE)] =\
       full_train_set_2d[(f, data_utils.CAMERA_TO_USE)][:, dim_to_use_2d]
  for k in full_test_set_3d:
    (f, c) = k
    test_set_3d[k] = full_test_set_3d[k][:, dim_to_use_3d]
    test_set_2d[(f, data_utils.CAMERA_TO_USE)] =\
       full_test_set_2d[(f, data_utils.CAMERA_TO_USE)][:, dim_to_use_2d]
  
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    device_count=device_count)) as sess:
    
    # === Create the model ===
    print("[*] creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    batch_size = FLAGS.batch_size #128 ???
    # Dropout probability 0 (keep probability 1) for sampling
    dp = 1.0

    model = create_model(sess, batch_size)
    print("[+] model loaded")
    encoder_inputs, decoder_outputs =\
      model.get_all_batches( test_set_2d, test_set_3d, FLAGS.camera_frame, training=False)
    nbatches = len( encoder_inputs )
    print("[*] there are {0} test batches".format( nbatches ))
    
    all_enc_in = []
    all_dec_out = []
    all_poses_3d = []
    for i in range( nbatches ):
      enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
      _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
      
      # denormalize
      enc_in  = data_utils.unNormalize_batch(enc_in, data_mean_2d, data_std_2d, dim_to_use_2d)
      dec_out = data_utils.unNormalize_batch(dec_out, data_mean_3d, data_std_3d, dim_to_use_3d)
      poses3d = data_utils.unNormalize_batch(poses3d, data_mean_3d, data_std_3d, dim_to_use_3d)
      all_enc_in.append( enc_in )
      all_dec_out.append( dec_out )
      all_poses_3d.append( poses3d )

    # Put all the poses together
    enc_in, dec_out, poses3d = map( np.vstack, [all_enc_in, all_dec_out, all_poses_3d] )
    
    # Convert back to world coordinates
    '''
    if FLAGS.camera_frame:
      R, T, f, ce, d, intr = rcams[data_utils.PROJECT_CAMERA]

      def cam2world_centered(data_3d_camframe):
        data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
        data_3d_worldframe = data_3d_worldframe.reshape((-1, data_utils.DIMENSIONS*3))
        return data_3d_worldframe

      # Apply inverse rotation and translation
      dec_out = cam2world_centered(dec_out)
      poses3d = cam2world_centered(poses3d)
    '''
    viz.visualize_test_sample(enc_in, dec_out, poses3d)
    viz.visualize_test_animation(enc_in, dec_out, poses3d)

def main(_):
  if FLAGS.sample:
    sample()
  else:
    train()

if __name__ == "__main__":
  tf.compat.v1.app.run()
