import numpy as np
import tensorflow as tf
import FLAGS
import os
import pickle
import time
import sys
sys.path.append('./util/')
from PIL import Image

import util.util_test_OULU_Protocol_1 as util_OULU
from generate_data_test import input_fn_maker
from generate_network import generate_network as model_fn

path_txt = './scores/Protocol_1'
path_map = './map'

flags=FLAGS.flags
our_checkpoint_path = os.path.join(flags.path.model, 'model.ckpt-9501')
isOfficialEval = True
isOnline = False#True
if isOnline:
    interval_iteration = 1000
    interval_time = 300
else:
    start_iteration = 6000
    interval_iteration = 1000

if not os.path.exists(path_txt):
    os.makedirs(path_txt)
if not os.path.exists(path_map):
    os.makedirs(path_map)

flags=FLAGS.flags # setting paras
num_classes = flags.paras.num_classes
# log info setting
tf.logging.set_verbosity(tf.logging.INFO)
# data fn

test_data_list=[flags.path.dev_file, flags.path.test_file]
test_input_fn = input_fn_maker(test_data_list, shuffle=False, 
                                batch_size = 1,
                                epoch=1)

# model fn
model_fn_this=model_fn

tensors_to_log = {"step": "global_ss", "loss": "loss_cla", "accuracy": "accuracy"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, 
                                        every_n_iter=1000) # every_n_iter is important

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
fas_classifier = tf.estimator.Estimator(
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
    features=fas_classifier.predict(
            input_fn=test_input_fn,
            hooks=[logging_hook],
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
    def save_depth_map(depth_map, label, names, fea_ind):
        video_name = names[0].decode()
        label_str = 'pos' if label[0]==0 else 'neg'
        #video_name_encode = name_encode(video_name)
        depth = np.squeeze(depth_map)
        depth = 255 * (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        depth = np.array(depth, np.uint8)
        depth_pil = Image.fromarray(depth)
        depth_name_full = os.path.join(path_map, mode + '_' + label_str + '_' + video_name + '_%d.jpg'%(fea_ind))
        depth_pil.save(depth_name_full)

    eval_input_fn = input_fn_maker(data_list, shuffle=False, 
                                batch_size = 1,
                                epoch=1)
    features=fas_classifier.predict(
            input_fn=eval_input_fn,
            hooks=[logging_hook],
            checkpoint_path= path_model_now )

    fea_ind=0
    acc_mean = 0.0
    video_name = None
    video_score = 0.0
    video_frame_count = 0.0
    for feature in features:
        logits=feature['logits']
        labels=feature['labels']
        names=feature['names']
        depth_map=feature['depth_map']
        '''save_depth_map'''
        save_depth_map(depth_map, labels, names, fea_ind)
        #prob= logits[1]>logits[0]
        #acc= int(prob==labels[0])
        out = np.argmax(np.array(logits))
        #out = 1 if out == 0 else 0
        acc = int(out == labels[0])
        acc_mean += float(acc)
        print(fea_ind, logits, out, labels, [name.decode() for name in names], acc )
        fea_ind += 1

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
    iter_before = start_iteration

    fid = open(all_path_ckpt, 'r')
    lines = fid.readlines()
    fid.close()
    
    if not our_checkpoint_path == None:
        officialEval(our_checkpoint_path)
        iter_now = int(our_checkpoint_path.split('-')[-1])
        print(iter_now, 'Has generated all maps')
        #getExel(path_txt, iter_now)
        return 1
    
    if len(lines) < 3:
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
    

