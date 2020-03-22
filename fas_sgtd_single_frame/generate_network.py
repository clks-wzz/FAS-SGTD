# author: Zezheng Wang
import os
import numpy as np
import tensorflow as tf
from FLAGS import flags
import tensorflow.contrib.slim as slim
import util.util_network as util_network
from util.util_network import *

len_seq=flags.paras.len_seq
num_classes = flags.paras.num_classes

gpus_str = os.environ['CUDA_VISIBLE_DEVICES']
print('CUDA_VISIBLE_DEVICES:', gpus_str)
gpus_str_split = gpus_str.split(',')
len_gpus = len(gpus_str_split)
gpus_list = [int(x) for x in gpus_str_split]

Conv_act = None
BN_act = tf.nn.relu
multipler = 2

def FaceMapNet(image, is_training):	
    with slim.arg_scope([slim.conv2d],
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = residual_gradient_conv(image, 64, is_training=is_training, name='rgc_init')

        net = residual_gradient_conv(net, 64*multipler, is_training=is_training, name='rgc1_1')
        net = residual_gradient_conv(net, 96*multipler, is_training=is_training, name='rgc1_2')
        net = residual_gradient_conv(net, 64*multipler, is_training=is_training, name='rgc1_3')

        pool1=slim.max_pool2d(net,[2,2],stride=[2,2],scope='pool1')

        net = residual_gradient_conv(pool1, 64*multipler, is_training=is_training, name='rgc2_1')
        net = residual_gradient_conv(net, 96*multipler, is_training=is_training, name='rgc2_2')
        net = residual_gradient_conv(net, 64*multipler, is_training=is_training, name='rgc2_3')

        pool2=slim.max_pool2d(net,[2,2],stride=[2,2],scope='pool2')

        net = residual_gradient_conv(pool2, 64*multipler, is_training=is_training, name='rgc3_1')
        net = residual_gradient_conv(net, 96*multipler, is_training=is_training, name='rgc3_2')
        net = residual_gradient_conv(net, 64*multipler, is_training=is_training, name='rgc3_3')

        pool3=slim.max_pool2d(net,[2,2],stride=[2,2],scope='pool3')

        feature1 = tf.image.resize_bilinear(pool1, size = (32, 32))
        feature2 = tf.image.resize_bilinear(pool2, size = (32, 32))
        print('pool1 pool2', pool1.get_shape(), pool2.get_shape())
        pool_concat = tf.concat([feature1, feature2, pool3], axis = -1)

        net = residual_gradient_conv(pool_concat, 64*multipler, is_training=is_training, name='rgc4_1')
        net = residual_gradient_conv(net, 32*multipler, is_training=is_training, name='rgc4_2')
        net = slim.conv2d(net,len_seq,[3,3],stride=[1,1],activation_fn=tf.nn.relu,scope='conv4_3',padding='SAME')
        
        #net = tf.clip_by_value(net,0.0,1.0)

    return net


def images_to_tensor(images):
    images_list=[tf.expand_dims(x,axis=1) for x in images]
    input_image=tf.concat(images_list,axis=1)
    print('Input Shape:',input_image.get_shape())
    return input_image

def LstmCnnNet(images, mode):
    if(mode=='train'):
        isTraining=True
        keep_prob=0.5
    else:
        isTraining=False
        keep_prob=1.0
    # FaceMapNet
    with tf.variable_scope('FaceMapNet',reuse=tf.AUTO_REUSE):        
        input_image=images #images_to_tensor(images)
        print('input shape:', input_image.get_shape())
        logits_map = FaceMapNet(input_image, isTraining)

    return logits_map
    lucky='lucky!'

def build_loss(logits, maps, masks, label):
    logits_map = logits
    logits_cla = tf.reduce_mean(logits_map, axis=1)
    logits_cla = tf.reduce_mean(logits_cla, axis=1)
    logits_cla = tf.reduce_mean(logits_cla, axis=1, keep_dims=True)
    logits_cla = tf.concat([logits_cla, 1 - logits_cla], axis = 1)
    logits = logits_cla
    print('logits-shape:', logits.get_shape())

    label=tf.cast(label,tf.int32)
    print(label.get_shape())
    label=tf.one_hot(label,depth=num_classes,on_value=1.0,off_value=0.0,axis=-1)
    label=tf.squeeze(label, axis=1)
    print(label.get_shape())
    if(len(label.get_shape())==3):
        label=tf.squeeze(label,axis=0)
    
    # loss of classification
    #loss_cla=tf.nn.weighted_cross_entropy_with_logits(logits=logits,targets=label,
    #                                            pos_weight=tf.constant([1.0,1.0],dtype=tf.float32))
    if ( not flags.paras.isFocalLoss ) and flags.paras.isWeightedLoss:
        #loss_cla=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label)
        print('isWeightedLoss !')
        loss_cla=tf.nn.weighted_cross_entropy_with_logits(logits=logits,targets=label,
                                                pos_weight=tf.constant([4.0,1.0],dtype=tf.float32))
    elif( not flags.paras.isFocalLoss ) and (not flags.paras.isWeightedLoss):
        print('isSoftmaxLoss !')
        loss_cla=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label)
    else:
        print('isFocalLoss !')
        logits=tf.nn.softmax(logits)
        loss_cla=-label*tf.pow(1.0-logits,2.0)*tf.log(logits)-(1-label)*tf.pow(logits,2.0)*tf.log(1-logits)
    loss_cla=tf.reduce_mean(loss_cla, name='loss_cla')

    # tensorboard summary
    tf.summary.scalar('loss',loss_cla)
    # loss of depth map 
    
    maps_reg = maps / 255.0
    loss_depth_1= tf.pow(logits_map - maps_reg, 2)
    loss_depth_1 = tf.reduce_mean(loss_depth_1)
    # loss of contrast depth loss
    loss_depth_2 = util_network.contrast_depth_loss(logits_map, maps_reg)
    # total loss of depth 
    loss_depth = loss_depth_1 + 0.5 * loss_depth_2
    # loss of regularizer
    loss_reg=tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # total loss
    #loss = loss_cla + loss_reg + loss_depth
    loss = loss_reg + loss_depth

    correction=tf.equal(tf.cast(tf.argmax(label,axis=-1),dtype=tf.float32),\
                        tf.cast(tf.argmax(logits,axis=-1),dtype=tf.float32))
    accuracy=tf.reduce_mean(tf.cast(correction,dtype=tf.float32), name='accuracy')

    acc=tf.metrics.accuracy(labels=tf.cast(tf.argmax(label,axis=-1),dtype=tf.float32),
                             predictions=tf.cast(tf.argmax(logits,axis=-1),dtype=tf.float32) )
    eval_metric_ops = {"accuracy_metric": acc}
    tf.summary.scalar('accuracy', accuracy)

    return loss, loss_cla, loss_depth, accuracy, eval_metric_ops, logits_cla

def build_optimizer():
    optimizer = tf.train.AdamOptimizer(learning_rate=flags.paras.learning_rate)
    #optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    return optimizer

def show_trainable_vars():
    vars=tf.trainable_variables()
    for var in vars:
        print(var.name)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            print(g.name)
            if g is None:
                continue
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if grads==[]:
            continue
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def generate_network(features=[], mode=tf.estimator.ModeKeys.TRAIN):
    images=features['images']
    maps=features['maps']
    masks=features['masks']
    #face=features['faces']
    label=features['labels']
    #image=tf.placeholder(dtype=tf.float32, shape=(16, 6, 256, 256, 3))
    #face=tf.placeholder(dtype=tf.float32, shape=(16, 6, 128, 128, 3))
    #label=tf.placeholder(dtype=tf.float32, shape=(16, 1))

    #print(image.get_shape().as_list())

    mode_str='train' if mode==tf.estimator.ModeKeys.TRAIN else 'test'
    ### global step
    global_step_my = tf.train.get_global_step()

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits=LstmCnnNet(images, mode_str)
        loss, loss_cla, loss_depth, accuracy, eval_metric_ops, logits_cla = build_loss(logits, maps, masks, label)  
        show_trainable_vars()
        predictions={
            'depth_map': logits,
            'logits': logits_cla,
            'masks': masks,
            'labels': label,
            'names': features['names']
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    optimizer = build_optimizer()

    batch_size_this = tf.shape(images)[0]
    split_size = batch_size_this // len_gpus
    splits = [split_size, ] * (len_gpus - 1)
    splits.append(batch_size_this - split_size * (len_gpus - 1))
    images_split = tf.split(images, splits, axis=0)
    maps_split = tf.split(maps, splits, axis=0)
    masks_split = tf.split(masks, splits, axis=0)
    label_split = tf.split(label, splits, axis=0)
    tower_grads = []
    #eval_loss = []
    total_loss_depth = 0.0

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                    print('/gpu:%d' % gpus_list[i])
                    logits=LstmCnnNet(images_split[i], mode_str)
                    print('logits-shape:', logits.get_shape())
                
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    #update_ops.append(global_count_op)
                    updates_op = tf.group(*update_ops)
                    
                    with tf.control_dependencies([updates_op]):
                        loss, loss_cla, loss_depth, accuracy, eval_metric_ops, logits_cla = build_loss(logits, maps_split[i], masks_split[i], label_split[i])  
                    
                    train_vars = tf.trainable_variables()

                    grads = optimizer.compute_gradients(loss, var_list=train_vars)
                    tower_grads.append(grads)
                    #eval_loss.append(loss_depth)
                    total_loss_depth += loss_depth

    show_trainable_vars()
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step_my)
    train_op = apply_gradient_op

    total_loss_depth /= len_gpus
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss_depth, train_op=train_op)

if __name__=='__main__':
    generate_network()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver.save(sess,"./Model/model.ckpt")
    print('lucky!')

