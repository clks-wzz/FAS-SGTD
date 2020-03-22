# author: Zezheng Wang
import numpy as np
import tensorflow as tf
from FLAGS import flags
import tensorflow.contrib.slim as slim
import util.util_network as util_network
from util.util_network import *
import util.BasicConvLSTMCell as BasicConvLSTMCell
import os

BasicConvLSTMCell = BasicConvLSTMCell.ConvGRUCell
#from my_convolution3D import convolution3d
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
    pre_off_list = []
    with slim.arg_scope([slim.conv2d],
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = residual_gradient_conv(image, 64, is_training=is_training, name='rgc_init')

        net = residual_gradient_conv(net, 64*multipler, is_training=is_training, name='rgc1_1')
        net = residual_gradient_conv(net, 96*multipler, is_training=is_training, name='rgc1_2')
        net = residual_gradient_conv(net, 64*multipler, is_training=is_training, name='rgc1_3')
        pre_off_list.append(net)

        pool1=slim.max_pool2d(net,[2,2],stride=[2,2],scope='pool1')

        net = residual_gradient_conv(pool1, 64*multipler, is_training=is_training, name='rgc2_1')
        net = residual_gradient_conv(net, 96*multipler, is_training=is_training, name='rgc2_2')
        net = residual_gradient_conv(net, 64*multipler, is_training=is_training, name='rgc2_3')
        pre_off_list.append(net)

        pool2=slim.max_pool2d(net,[2,2],stride=[2,2],scope='pool2')

        net = residual_gradient_conv(pool2, 64*multipler, is_training=is_training, name='rgc3_1')
        net = residual_gradient_conv(net, 96*multipler, is_training=is_training, name='rgc3_2')
        net = residual_gradient_conv(net, 64*multipler, is_training=is_training, name='rgc3_3')
        pre_off_list.append(net)

        pool3=slim.max_pool2d(net,[2,2],stride=[2,2],scope='pool3')

        feature1 = tf.image.resize_bilinear(pool1, size = (32, 32))
        feature2 = tf.image.resize_bilinear(pool2, size = (32, 32))
        print('pool1 pool2', pool1.get_shape(), pool2.get_shape())
        pool_concat = tf.concat([feature1, feature2, pool3], axis = -1)

        net = residual_gradient_conv(pool_concat, 64*multipler, is_training=is_training, name='rgc4_1')
        net = residual_gradient_conv(net, 32*multipler, is_training=is_training, name='rgc4_2')
        net = slim.conv2d(net,1,[3,3],stride=[1,1],activation_fn=tf.nn.relu,scope='conv4_3',padding='SAME')

    return net, pre_off_list


def OFFNet(pre_off_list, isTraining):
    reduce_num = 32

    sobel_plane_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_plane_x = np.expand_dims(sobel_plane_x, axis=-1)
    sobel_plane_x = np.repeat(sobel_plane_x, reduce_num, axis=-1)
    sobel_plane_x = np.expand_dims(sobel_plane_x, axis=-1)
    sobel_kernel_x = tf.constant(sobel_plane_x, dtype=tf.float32)

    sobel_plane_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_plane_y = np.expand_dims(sobel_plane_y, axis=-1)
    sobel_plane_y = np.repeat(sobel_plane_y, reduce_num, axis=-1)
    sobel_plane_y = np.expand_dims(sobel_plane_y, axis=-1)
    sobel_kernel_y = tf.constant(sobel_plane_y, dtype=tf.float32)

    def OFFBlock(pre_off_feature, postfix = 'lucky'):
        feature_shape = pre_off_feature.get_shape()
        feature_dim = feature_shape[-1]
        #with tf.variable_scope(None, reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005)):

                net = slim.conv2d(pre_off_feature, reduce_num, \
                    [1,1],stride=[1,1],scope='OFFNet_conv_'+postfix,padding='SAME')
                net = slim.batch_norm(net, is_training=isTraining, \
                    activation_fn=None,scope='OFFNet_bn_'+postfix)
                net_reshape = tf.reshape(net, shape = [-1, len_seq, \
                    feature_shape[1], feature_shape[2], reduce_num])
                print('shape!!!!!!!!!:', net.get_shape(), net_reshape.get_shape())

                Spatial_Gradient_x = tf.nn.depthwise_conv2d(net, \
                    filter=sobel_kernel_x, strides=[1,1,1,1], padding='SAME')
                Spatial_Gradient_x = tf.reshape(Spatial_Gradient_x, shape = [-1, len_seq, \
                    feature_shape[1], feature_shape[2], reduce_num])
                Spatial_Gradient_y = tf.nn.depthwise_conv2d(net, \
                    filter=sobel_kernel_y, strides=[1,1,1,1], padding='SAME')
                Spatial_Gradient_y = tf.reshape(Spatial_Gradient_y, shape = [-1, len_seq, \
                    feature_shape[1], feature_shape[2], reduce_num])

                Temporal_Gradient = net_reshape[:, :-1, :, :, :] - net_reshape[:, 1:, :, :, :]

                pre_off_feature_squence = tf.reshape(pre_off_feature, shape = [-1, len_seq, \
                    feature_shape[1], feature_shape[2], feature_shape[3]])

                single_feature_scale = 1.0
                off_feature_squence = tf.concat([
                                                pre_off_feature_squence[:,:-1,:,:,:] * single_feature_scale,\
                                                Spatial_Gradient_x[:,:-1,:,:,:],\
                                                Spatial_Gradient_y[:,:-1,:,:,:],\
                                                Spatial_Gradient_x[:,1:,:,:,:],\
                                                Spatial_Gradient_y[:,1:,:,:,:],\
                                                Temporal_Gradient\
                                                ], axis = -1)
                off_feateure_batch = tf.reshape(off_feature_squence, shape = [-1, \
                    feature_shape[1], feature_shape[2], off_feature_squence.get_shape()[-1]])
                
                res_feature = slim.conv2d(off_feateure_batch, feature_dim, \
                    [3,3],stride=[1,1],scope='OFFNet_linear_'+postfix,padding='SAME')
                
                res_feature = slim.max_pool2d(res_feature,[2,2],stride=[2,2],scope='pool_'+postfix)
        return res_feature

    len_pre_off = len(pre_off_list)

    net1 = OFFBlock(pre_off_list[0], postfix = '1_off')
    net2 = OFFBlock(pre_off_list[1], postfix = '2_off')
    net3 = OFFBlock(pre_off_list[2], postfix = '3_off')

    with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net1 = tf.image.resize_bilinear(net1, size = (64, 64))
        net_concat = tf.concat([net1, net2], axis = -1)
        net2 = slim.conv2d(net_concat, 128, \
            [3,3],stride=[1,1],scope='cascade_conv_1',padding='SAME')
        net2 = slim.batch_norm(net2, is_training=isTraining, activation_fn=None,scope='cascade_bn_1')

        net2 = tf.image.resize_bilinear(net2, size = (32, 32))
        net_concat = tf.concat([net2, net3], axis = -1)
        net3 = slim.conv2d(net_concat, 128, \
            [3,3],stride=[1,1],scope='cascade_conv_2',padding='SAME')
        net3 = slim.batch_norm(net3, is_training=isTraining, activation_fn=None,scope='cascade_bn_2')

    '''
    feature1 = tf.image.resize_bilinear(net1, size = (32, 32))
    feature2 = tf.image.resize_bilinear(net2, size = (32, 32))
    feature3 = net3
    pool_concat = tf.concat([feature1, feature2, feature3], axis = -1)
    '''
    pool_concat = net3

    with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(pool_concat, 128, \
            [3,3],stride=[1,1],scope='OFFNet_conv_4_1',padding='SAME')
        net = slim.batch_norm(net, is_training=isTraining, activation_fn=None,scope='OFFNet_bn_4_1')
        net = tf.reshape(net, [-1, len_seq-1, net.get_shape()[1], net.get_shape()[2], net.get_shape()[3]])

    return net

def ConvLSTMNet(input_feature, isTraining):
    cell_1 = BasicConvLSTMCell([32,32], 64, [3,3], normalize=False, is_training = isTraining) 
    cell_2 = BasicConvLSTMCell([32,32],  1, [3,3], last_activation=None, normalize=False, is_training = isTraining) 
    cell_3 = BasicConvLSTMCell([32,32],  1, [3,3], last_activation=tf.nn.tanh, normalize=False, is_training = isTraining) 
    outputs1, state1 = tf.nn.dynamic_rnn(cell_1, input_feature, \
                                        initial_state=None, dtype=tf.float32, time_major=True, scope = 'cell_1')
    outputs2, state2 = tf.nn.dynamic_rnn(cell_2, outputs1, \
                                        initial_state=None, dtype=tf.float32, time_major=True, scope = 'cell_2')
    #outputs2 = tf.Print(outputs2, [outputs2])
    
    print('LSTM shape:', outputs2.shape)
    depth_split = tf.split(outputs2, num_or_size_splits=len_seq-1, axis=1)
    depth_split_list = [tf.squeeze(x, axis=1) for x in depth_split] 
    return depth_split_list
    
    
def SoftmaxNetSub(input):
    with slim.arg_scope([slim.fully_connected],
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        input = slim.flatten(input)
        fc1 = slim.fully_connected(input, 64, scope='fc/fc_1') 
        fc1 = tf.nn.relu(fc1)
        fc2 = slim.fully_connected(fc1, 2, scope='fc/fc_2')
        fc = tf.nn.softmax(fc2) 
    return fc

def images_to_tensor(images):
    images_list=[tf.expand_dims(x,axis=1) for x in images]
    input_image=tf.concat(images_list,axis=1)
    print('Input Shape:',input_image.get_shape())
    return input_image

def sequence_to_batch(input_image):
    input_image_split=tf.split(input_image, num_or_size_splits=len_seq, axis=-1)
    input_image=tf.concat(input_image_split,axis=0)
    return input_image
def batch_to_sequence(input_tensor):
    input_sequence = tf.reshape(input_tensor, \
                    shape=[-1, len_seq, \
                    input_tensor.get_shape()[1], \
                    input_tensor.get_shape()[2], \
                    input_tensor.get_shape()[3]\
                    ])
    return input_sequence

def get_logtis_cla_from_logits_list_bk(logits_list):
    logits_map_mean = tf.concat(logits_list, axis=-1)
    logits_map_mean = tf.reduce_mean(logits_map_mean, axis = -1, keep_dims=True)
    logits_cla = tf.pow(logits_map_mean, 2.0)
    logits_cla = tf.reduce_mean(logits_cla, axis=1)
    logits_cla = tf.reduce_mean(logits_cla, axis=1)
    logits_cla = tf.reduce_mean(logits_cla, axis=1, keep_dims=True)
    logits_cla = tf.sqrt(logits_cla)
    logits_cla = tf.concat([logits_cla, 1 - logits_cla], axis = 1)
    return logits_cla

def get_logtis_cla_from_logits_list(logits_list):
    logits_map_concat = tf.concat(logits_list, axis=-1)
    logits_map_concat = tf.reduce_mean(logits_map_concat, axis=-1)
    with tf.variable_scope('SoftmaxNet',reuse=tf.AUTO_REUSE):
        logits_cla = SoftmaxNetSub(logits_map_concat)

    return logits_cla

def get_train_vars_tune(train_vars):
    var_list = []
    for i in range(len(train_vars)):
        var_name  = train_vars[i].name
        if len(var_name.split('/'))>=1 and (var_name.split('/')[0] == 'ConvLSTMNet' \
                                            or var_name.split('/')[0] == 'OFFNet'
                                            or var_name.split('/')[0] == 'SoftmaxNet'):
            var_list.append(train_vars[i])
    print('len(var_list):', len(var_list))
    return var_list

def show_trainable_vars():
    vars=tf.trainable_variables()
    for var in vars:
        print(var.name)

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
        input_tensor = sequence_to_batch(input_image)
        print('input shape:', input_tensor.get_shape())
        features_map, pre_off_list = FaceMapNet(input_tensor, False)
    with tf.variable_scope('OFFNet',reuse=tf.AUTO_REUSE):
        off_feature = OFFNet(pre_off_list, isTraining)
    with tf.variable_scope('ConvLSTMNet',reuse=tf.AUTO_REUSE):
        input_feature = off_feature #batch_to_sequence(features_map)
        print('feature shape:', input_feature.get_shape())
        logits_map_split = ConvLSTMNet(input_feature, isTraining)
    
    beta = flags.paras.single_ratio
    alpha = 1 - beta
    single_maps = tf.split(features_map, num_or_size_splits=len_seq, axis=0)
    final_maps = []
    for i in range(len(logits_map_split)):
        final_maps.append( alpha * logits_map_split[i] +  beta * single_maps[i] )
        
    lucky='lucky!'
    return final_maps

def build_loss(logits_list, maps, masks, label):
    #logits_map = logits
    print('len-logits_list:', len(logits_list))
    logits_cla = get_logtis_cla_from_logits_list(logits_list)
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

    ### SUM DEPTH LOSS
    loss_depth = 0.0
    maps_list = tf.split(maps, num_or_size_splits=len_seq, axis=-1)
    #assert(len(logits_list) == len(maps_list) - 1)
    for i in range(len(maps_list)-1):
        logits_map = logits_list[i]
        # loss of depth map 
        maps_reg = maps_list[i] / 255.0
        loss_depth_1= tf.pow(logits_map - maps_reg, 2)
        loss_depth_1 = tf.reduce_mean(loss_depth_1)
        # loss of contrast depth loss
        loss_depth_2 = util_network.contrast_depth_loss(logits_map, maps_reg)
        # total loss of depth 
        loss_depth_this = loss_depth_1 + loss_depth_2
        loss_depth += loss_depth_this

    loss_depth = loss_depth/float(len_seq)

    # loss of regularizer
    loss_reg=tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # total loss
    #loss = loss_cla + loss_reg + loss_depth
    #loss = loss_reg + loss_depth
    cla_ratio = flags.paras.cla_ratio
    depth_ratio = 1 - cla_ratio
    loss = loss_reg + depth_ratio * loss_depth +  cla_ratio * loss_cla

    correction=tf.equal(tf.cast(tf.argmax(label,axis=-1),dtype=tf.float32),\
                        tf.cast(tf.argmax(logits,axis=-1),dtype=tf.float32))
    accuracy=tf.reduce_mean(tf.cast(correction,dtype=tf.float32), name='accuracy')

    acc=tf.metrics.accuracy(labels=tf.cast(tf.argmax(label,axis=-1),dtype=tf.float32),
                             predictions=tf.cast(tf.argmax(logits,axis=-1),dtype=tf.float32) )
    eval_metric_ops = {"accuracy_metric": acc}
    tf.summary.scalar('accuracy', accuracy)

    return loss, loss_cla, loss_depth, accuracy, eval_metric_ops, logits_cla

def build_optimizer():
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=flags.paras.learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate=flags.paras.learning_rate)
    #optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    return optimizer

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

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
    label=features['labels']

    mode_str='train' if mode==tf.estimator.ModeKeys.TRAIN else 'test'
    ### global step
    global_step_my = tf.train.get_global_step()

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits_list=LstmCnnNet(images, mode_str)
        logits = tf.reduce_mean(tf.concat(logits_list, axis=-1), axis=-1, keep_dims=True)
        logits_cla = get_logtis_cla_from_logits_list(logits_list)
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
                    logits_list=LstmCnnNet(images_split[i], mode_str)
                    logits = tf.reduce_mean(tf.concat(logits_list, axis=-1), axis=-1, keep_dims=True)
                    print('logits-shape:', logits.get_shape())
                
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    updates_op = tf.group(*update_ops)
                    with tf.control_dependencies([updates_op]):
                        loss, loss_cla, loss_depth, accuracy, eval_metric_ops, logits_cla \
                            = build_loss(logits_list, maps_split[i], masks_split[i], label_split[i])
                    
                    train_vars = tf.trainable_variables()
                    train_vars_tune = get_train_vars_tune(train_vars)

                    grads = optimizer.compute_gradients(loss, var_list=train_vars_tune)
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

    ### read params from ckpt
    exclude = ['OFFNet', 'ConvLSTMNet', 'SoftmaxNet']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    tf.train.init_from_checkpoint(os.path.join( \
                        'model_finetune', 'model.ckpt-9501'), 
                        {v.name.split(':')[0]: v for v in variables_to_restore})

    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss_depth, train_op=train_op)

if __name__=='__main__':
    generate_network()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver.save(sess,"./Model/model.ckpt")
    print('lucky!')

