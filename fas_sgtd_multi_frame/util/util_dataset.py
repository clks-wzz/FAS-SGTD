import os
import tensorflow as tf
from FLAGS import flags
import numpy as np
import copy

class IJCB:
    def __init__(self, protocol, mode):    

        protocol_dict = {}
        protocol_dict['ijcb_protocal_1']={'train': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 3, 4, 5] },
                                          'dev': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 3, 4, 5] },
                                          'test': { 'session': [3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 2, 3, 4, 5] }
                                        }
        protocol_dict['ijcb_protocal_2']={'train': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 4] },
                                          'dev': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 4] },
                                          'test': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 3, 5] }
                                        }        
        protocol_dict['ijcb_protocal_3']={'train': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 3, 4, 5] },
                                          'dev': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 3, 4, 5] },
                                          'test': { 'session': [1, 2, 3], 'phones': [6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 2, 3, 4, 5] }
                                        }
        for i in range(6):
            protocol_dict['ijcb_protocal_3_%d'%(i+1)] = copy.deepcopy(protocol_dict['ijcb_protocal_3'])
            protocol_dict['ijcb_protocal_3_%d'%(i+1)]['train']['phones'] = []
            protocol_dict['ijcb_protocal_3_%d'%(i+1)]['dev']['phones'] = []
            protocol_dict['ijcb_protocal_3_%d'%(i+1)]['test']['phones'] = []
            for j in range(6):
                if j==i:
                    protocol_dict['ijcb_protocal_3_%d'%(i+1)]['test']['phones'].append(j+1)
                else:
                    protocol_dict['ijcb_protocal_3_%d'%(i+1)]['train']['phones'].append(j+1)
                    protocol_dict['ijcb_protocal_3_%d'%(i+1)]['dev']['phones'].append(j+1)

        protocol_dict['ijcb_protocal_4']={'train': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 4] },
                                          'dev': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 4] },
                                          'test': { 'session': [3], 'phones': [6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 3, 5] }
                                        }
        for i in range(6):
            protocol_dict['ijcb_protocal_4_%d'%(i+1)] = copy.deepcopy(protocol_dict['ijcb_protocal_4'])
            protocol_dict['ijcb_protocal_4_%d'%(i+1)]['train']['phones'] = []
            protocol_dict['ijcb_protocal_4_%d'%(i+1)]['dev']['phones'] = []
            protocol_dict['ijcb_protocal_4_%d'%(i+1)]['test']['phones'] = []
            for j in range(6):
                if j==i:
                    protocol_dict['ijcb_protocal_4_%d'%(i+1)]['test']['phones'].append(j+1)
                else:
                    protocol_dict['ijcb_protocal_4_%d'%(i+1)]['train']['phones'].append(j+1)
                    protocol_dict['ijcb_protocal_4_%d'%(i+1)]['dev']['phones'].append(j+1)
        
        protocol_dict['ijcb_protocal_all']={'train': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 3, 4, 5] },
                                          'dev': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 3, 4, 5] },
                                          'test': { 'session': [3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 2, 3, 4, 5] }
                                        }

        self.protocol_dict = protocol_dict
        self.mode = mode      

        if not (protocol in self.protocol_dict.keys()):
            print('error: Protocal should be ', list(self.protocol_dict.keys()) )
            exit(1)
        self.protocol = protocol
        self.protocol_info = protocol_dict[protocol][mode]

    def isInPotocol(self, file_name_full):
        file_name = os.path.split(file_name_full)[-1]
        name_split = file_name.split('_')
        if not len(name_split)==4:
            return False
        
        [phones_, session_, users_, PAI_] = [int(x) for x in name_split]

        if (phones_ in self.protocol_info['phones']) and (session_ in self.protocol_info['session']) \
                and (users_ in self.protocol_info['users']) and (PAI_ in self.protocol_info['PAI']):
            return True
        else:
            return False

    def dataset_process(self, file_list):
        res_list = []
        for i in range(len(file_list)):
            file_name_full = file_list[i]
            if self.isInPotocol(file_name_full):
                res_list.append(file_name_full)
        print('Dataset Info:')
        print('----------------------------------------')
        print('IJCB', self.protocol, self.mode)
        print('File Counts:', len(res_list))
        print('----------------------------------------')

        return res_list

class Casia:
    def __init__(self, mode): 
        self.mode = mode

    def isInPotocol(self, file_name_full):
        file_name = os.path.split(file_name_full)[-1]
        name_split = file_name.split('_')
        if not len(name_split)==3:
            return False

        if name_split[0] == 'CASIA':
            return True
        else:
            return False

    def dataset_process(self, file_list):
        res_list = []
        for i in range(len(file_list)):
            file_name_full = file_list[i]
            if self.isInPotocol(file_name_full):
                res_list.append(file_name_full)
        print('Dataset Info:')
        print('----------------------------------------')
        print('CASIA', self.mode)
        print('File Counts:', len(res_list))
        print('----------------------------------------')

        return res_list

class ReplayAttack:
    def __init__(self, mode): 
        self.mode = mode

    def isInPotocol(self, file_name_full):
        file_name = os.path.split(file_name_full)[-1]
        name_split = file_name.split('_')
        if not len(name_split)==3:
            return False

        if name_split[0] == 'ReplayAttack':
            return True
        else:
            return False

    def dataset_process(self, file_list):
        res_list = []
        for i in range(len(file_list)):
            file_name_full = file_list[i]
            if self.isInPotocol(file_name_full):
                res_list.append(file_name_full)
        print('Dataset Info:')
        print('----------------------------------------')
        print('ReplayAttack', self.mode)
        print('File Counts:', len(res_list))
        print('----------------------------------------')

        return res_list

def pick_real_video(file_list):
    res_list = []
    for file_name in file_list:
        name_pure = os.path.split(file_name)[-1]
        if name_pure[-1] == '1':
            res_list.append(file_name)
    print('Pick real video :')
    print('----------------------------------------')
    print('Real Counts:', len(res_list))
    print('----------------------------------------')
    
    return res_list

color_odering_seed = 1
def distort_color(image, color_ordering=0):  
    if color_ordering == 0:  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)#????  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)#?????  
        image = tf.image.random_hue(image, max_delta=0.2)#???  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)#????  
    if color_ordering == 1:  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
        image = tf.image.random_hue(image, max_delta=0.2)  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
    if color_ordering == 2:  
        image = tf.image.random_hue(image, max_delta=0.2)  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
    if color_ordering == 3:  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
        image = tf.image.random_hue(image, max_delta=0.2)  
    return tf.clip_by_value(image, 0.0, 255.0) 

def distort_color_batch(image, args):
    image = tf.image.adjust_saturation(image, saturation_factor = args['saturation'])  
    #image = tf.image.adjust_hue(image, delta=args['hue'])  
    image = tf.image.adjust_contrast(image, contrast_factor=args['contrast'])  
    image = tf.image.adjust_brightness(image, delta=args['brightness']) 
    
    return tf.clip_by_value(image, 0.0, 255.0) 

def preprocess_for_train(images):
    images = images + 127.5
    if not len(images.get_shape()) == 4:
        print('Error dim: [T, H, W, C]')
        exit(1)
    ## random args of augment
    args={}
    args['saturation'] = tf.random_uniform([], 0.5, 1.5, dtype=tf.float32)
    args['hue'] = tf.random_uniform([], -0.2, 0.2, dtype=tf.float32)
    args['contrast'] = tf.random_uniform([], 0.5, 1.5, dtype=tf.float32)
    args['brightness'] = tf.random_uniform([], -32. / 255., 32. / 255., dtype=tf.float32)
    ##
    len_seq = flags.paras.len_seq
    input_image_split=tf.split(images, num_or_size_splits=len_seq, axis=0)
    input_image_list = []
    for i in range(len_seq):
        input_image_single=input_image_split[i]
        input_image_single=tf.squeeze(input_image_single, axis=0)
        distorted_image = distort_color_batch(input_image_single, args) 
        distorted_image = distorted_image - 127.5 #distorted_image*255.0-127.5
        distorted_image = tf.expand_dims(distorted_image, axis=0)
        input_image_list.append(distorted_image)
    data_augment = tf.concat(input_image_list, axis = 0)
    return data_augment

def preprocess_for_train_sequence(images):
    images = images + 127.5
    if not len(images.get_shape()) == 3:
        print('Error dim: [H, W, C]')
        exit(1)
    ## random args of augment
    args={}
    args['saturation'] = tf.random_uniform([], 0.5, 1.5, dtype=tf.float32)
    args['hue'] = tf.random_uniform([], -0.2, 0.2, dtype=tf.float32)
    args['contrast'] = tf.random_uniform([], 0.5, 1.5, dtype=tf.float32)
    args['brightness'] = tf.random_uniform([], -32. / 255., 32. / 255., dtype=tf.float32)
    ##
    len_seq = flags.paras.len_seq
    input_image_split=tf.split(images, num_or_size_splits=len_seq, axis=-1)
    input_image_list = []
    for i in range(len_seq):
        input_image_single=input_image_split[i]
        #input_image_single=tf.squeeze(input_image_single, axis=0)
        distorted_image = distort_color_batch(input_image_single, args) 
        distorted_image = distorted_image - 127.5 #distorted_image*255.0-127.5
        #distorted_image = tf.expand_dims(distorted_image, axis=0)
        input_image_list.append(distorted_image)
    data_augment = tf.concat(input_image_list, axis = -1)
    return data_augment

def preprocess_for_train_single(images):
    images = images + 127.5
    if not len(images.get_shape()) == 3:
        print('Error dim: [H, W, C]')
        exit(1)
    ## random args of augment
    args={}
    args['saturation'] = tf.random_uniform([], 0.5, 1.5, dtype=tf.float32)
    args['hue'] = tf.random_uniform([], -0.2, 0.2, dtype=tf.float32)
    args['contrast'] = tf.random_uniform([], 0.5, 1.5, dtype=tf.float32)
    args['brightness'] = tf.random_uniform([], -32. / 255., 32. / 255., dtype=tf.float32)
    ##
    data_augment = distort_color_batch(images, args) 
    data_augment = data_augment - 127.5
    ##
    return data_augment