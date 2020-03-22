import numpy as np
import tensorflow as tf
import FLAGS
import os
import pickle
import time
import sys
sys.path.append('./util/')

import util.util_test_OULU_Protocol_1 as util_OULU
from generate_data_test import input_fn_maker
from generate_network import generate_network as model_fn

flags=FLAGS.flags
our_checkpoint_path = None #os.path.join(flags.path.model, 'model.ckpt-697003')
isOfficialEval = True
isOnline = True
if isOnline:
    interval_iteration = 500
    interval_time = 16
else:
    start_iteration = 15001
    interval_iteration = 500

path_txt = './scores/norm_fusion_att_%.2f_%.2f/Protocol_1'%(flags.paras.single_ratio, flags.paras.cla_ratio)
if not os.path.exists(path_txt):
    os.makedirs(path_txt)

flags=FLAGS.flags # setting paras
num_classes = flags.paras.num_classes
# log info setting
tf.logging.set_verbosity(tf.logging.INFO)
# data fn

test_data_list=[flags.path.dev_file, flags.path.test_file]
test_input_fn = input_fn_maker(test_data_list, shuffle=False, 
                                batch_size = 20,
                                epoch=1)

# model fn
model_fn_this=model_fn

# GPU config
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True 
# # create estimator
this_config=tf.estimator.RunConfig(
    save_summary_steps=10000000000,
    save_checkpoints_steps=None,
    keep_checkpoint_max=1024,
    log_step_count_steps=None,
    session_config=config
)
mnist_classifier = tf.estimator.Estimator(
    model_fn=model_fn_this, config=this_config, model_dir=flags.path.model)

# savers
savers_name_lists=['evaluation_ijcb.pkl', 'evaluation_casia.pkl', 
                'evaluation_replayattack.pkl', 'evaluation_all.pkl']
savers_pkl_lists=[[], [], [], []]

# save by pcikle lib 
def save_contents(fea_ind, logits, labels, names):
    assert(len(names)==1)
    name_pure=names[0]
    name_tile=name_pure.split('_')[0]

    ## save contents in savers
    savers_pkl_lists[3].append([fea_ind, logits, labels, name_pure])
    if(name_tile=='CASIA'):
        savers_pkl_lists[1].append([fea_ind, logits, labels, name_pure])
    elif(name_tile=='ReplayAttack'):   
        savers_pkl_lists[2].append([fea_ind, logits, labels, name_pure])
    else:
        savers_pkl_lists[0].append([fea_ind, logits, labels, name_pure])

def save_pickles():
    for i in range(len(savers_name_lists)):
        savers_name=savers_name_lists[i]
        fid=open(savers_name,'wb')
        pickle.dump(savers_pkl_lists[i], fid)
        fid.close()

def ourEval():
    # # test
    features=mnist_classifier.predict(
            input_fn=test_input_fn,
            checkpoint_path= our_checkpoint_path )

    fea_ind=0
    acc_mean = 0.0
    for feature in features:
        logits=feature['logits']
        labels=feature['labels']
        names=feature['names']
        prob= logits[1]>logits[0]
        acc= int(prob==labels[0])
        acc_mean += float(acc)
        print(fea_ind, logits, labels, [name.decode() for name in names], acc )

        save_contents(fea_ind, logits, labels, [name.decode() for name in names])

        fea_ind+=1
    print('acc_mean:', acc_mean/float(fea_ind))

    save_pickles()

def officialEvalSub(txt_name, data_list, mode, path_model_now):
    def realProb(logits):
        #return np.exp(logits[1])/(np.exp(logits[0])+np.exp(logits[1]))
        x = np.array(logits)
        y = np.exp(x[0])/np.sum(np.exp(x))
        #y = x[0]
        return y
    def name_encode(name_):
        if mode == 'dev':
            return name_
        elif mode == 'test':
            name_split = name_.split('_')
            name_10 = name_split[0] + name_split[3] + name_split[1] + name_split[2]
            name_16 = hex(int(name_10))
            name_16 = name_16[0] + name_16[2:]
            return name_16
        else:
            print('Error mode: requires dev or test')
            exit(1)        

    eval_input_fn = input_fn_maker(data_list, shuffle=False, 
                                batch_size = 1,
                                epoch=1)
    features=mnist_classifier.predict(
            input_fn=eval_input_fn,
            checkpoint_path= path_model_now )
    fid = open(txt_name, 'w')
    fea_ind=0
    acc_mean = 0.0
    video_name = None
    video_score = 0.0
    video_frame_count = 0.0
    for feature in features:
        logits=feature['logits']
        '''
        logits_tmp = logits[1]
        logits[1] = logits[0]
        logits[0] = logits_tmp
        '''
        labels=feature['labels']
        names=feature['names']
        depth_map=feature['depth_map']   
        masks=feature['masks']  
        depth_map = depth_map[..., 0]*masks[..., 0]   
        #depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-6)
        depth_mean = np.sum(depth_map) / np.sum(masks[..., 0])

        cla_ratio = flags.paras.cla_ratio
        depth_ratio = 1 - cla_ratio
        logits[0] = depth_ratio * depth_mean + cla_ratio * logits[0]
        logits[1] = 1.0 - depth_mean

        out = np.argmax(np.array(logits))
        #out = 1 if out == 0 else 0
        acc = int(out == labels[0])
        acc_mean += float(acc)
        print(fea_ind, logits, out, labels, [name.decode() for name in names], acc )

        if (video_name == None):
            video_name = names[0].decode()
            video_score += realProb(logits)
            video_frame_count += 1.0
        elif (not names[0].decode() == video_name):
            video_score_mean = video_score/video_frame_count
            video_name_encode = name_encode(video_name)
            fid.write(video_name_encode + ',' + str(video_score_mean) + '\n')

            video_name = names[0].decode()
            video_score = 0.0
            video_frame_count = 0.0
            video_score += realProb(logits)
            video_frame_count += 1.0
        else:
            video_score += realProb(logits)
            video_frame_count += 1.0
        #save_contents(fea_ind, logits, labels, [name.decode() for name in names])
        fea_ind+=1
    if video_frame_count == 0:
        video_score_mean = 0.0
    else:
        video_score_mean = video_score/video_frame_count
    video_name_encode = name_encode(video_name)
    fid.write(video_name_encode + ',' + str(video_score_mean) + '\n')

    print('acc_mean:', acc_mean/float(fea_ind))
    fid.close()

def officialEval(path_model_now):
    path_txt_dev = os.path.join(path_txt, 'Dev_scores.txt')
    path_txt_test = os.path.join(path_txt, 'Test_scores.txt')
    
    officialEvalSub(path_txt_dev, [flags.path.dev_file], 'dev', path_model_now)
    officialEvalSub(path_txt_test, [flags.path.test_file], 'test', path_model_now)   

def getExel(path_scores, iter_now):
    Performances_this = util_OULU.get_scores_Protocol_1(os.path.split(path_scores)[0])
    scores_txt = os.path.join(path_scores, 'eval.txt')

    if not os.path.exists(scores_txt):
        lines = []
    else:
        fid = open(scores_txt, 'r')
        lines = fid.readlines()
        fid.close()

    str_list = [('%-7.4f'%(x)) for x in Performances_this]
    str_per = ''
    for str_ in str_list:
        str_per += str_ + ' '
    fid = open(scores_txt, 'w')
    line_new =  str_per + str(iter_now) + '\n'
    lines.append(line_new)
    fid.writelines(lines)
    fid.close()

def online_eval():
    all_path_ckpt = os.path.join(flags.path.model, 'checkpoint')
    iter_before = 0
    while True:
        time.sleep(interval_time)
        if not os.path.exists(all_path_ckpt): 
            continue
        fid = open(all_path_ckpt, 'r')
        lines = fid.readlines()
        fid.close()
        iter_now = int(lines[0].split('-')[-1][:-2])
        if iter_now - iter_before >= interval_iteration:
            officialEval(os.path.join(flags.path.model, 'model.ckpt-%d'%(iter_now)))
            getExel(path_txt, iter_now)
            iter_before = iter_now

def offline_eval():
    all_path_ckpt = os.path.join(flags.path.model, 'checkpoint')
    
    while True:
        if not os.path.exists(all_path_ckpt):
            continue
        else:
            print('No checkpoint')
            break        

    iter_before = start_iteration

    fid = open(all_path_ckpt, 'r')
    lines = fid.readlines()
    fid.close()
    
    if not our_checkpoint_path == None:
        officialEval(our_checkpoint_path)
        return 1
    
    if len(lines) < 2:
        print('Hasn\'t enough model!')
        return 0
    
    for i in range(1, len(lines)):
        iter_now = int(lines[i].split('-')[-1][:-2])
        if iter_now - iter_before >= interval_iteration:
            officialEval(os.path.join(flags.path.model, 'model.ckpt-%d'%(iter_now)))
            getExel(path_txt, iter_now)
            iter_before = iter_now

if __name__ == '__main__':
    if isOnline:
        online_eval()
    else:
        offline_eval()
    

