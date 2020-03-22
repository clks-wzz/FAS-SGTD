import numpy as pd
import tensorflow as tf
import FLAGS
from generate_data_train import input_fn_maker
from generate_network import generate_network as model_fn

flags=FLAGS.flags # setting paras
# log info setting
tf.logging.set_verbosity(tf.logging.INFO)
# data fn

train_data_list=[flags.path.train_file]
train_input_fn = input_fn_maker(train_data_list, shuffle=True, 
                                batch_size = flags.paras.batch_size_train,
                                epoch=flags.paras.epoch)

# model fn
model_fn_this=model_fn

#tensors_to_log = {"step": "global_ss"}
#logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, 
#                                        every_n_iter=flags.display.display_iter) # every_n_iter is important

# GPU config
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True 
# # create estimator
this_config=tf.estimator.RunConfig(
    save_summary_steps=flags.display.summary_iter,
    save_checkpoints_steps=flags.display.display_iter,
    keep_checkpoint_max=None,
    log_step_count_steps=flags.display.display_iter,
    session_config=tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=tf.GPUOptions(allow_growth=True))
)
fas_classifier = tf.estimator.Estimator(
    model_fn=model_fn_this, config=this_config, model_dir=flags.path.model)

''' only run train set '''
fas_classifier.train(input_fn=train_input_fn)

