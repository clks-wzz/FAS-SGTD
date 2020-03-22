from easydict import EasyDict as edict

flags=edict()

flags.path=edict()

path_gen_save = './model_save/'

flags.path.train_file=['/workspace/data/train_images',
                      '/workspace/data/train_depth']
flags.path.dev_file=['/workspace/data/dev_images',
                    '/workspace/data/dev_depth']
flags.path.test_file=['/workspace/data/test_images',
                     '/workspace/data/test_images']

flags.path.model= path_gen_save #v10.1.1 for normal conv3d; v10.1.2 for 1.4 conv3d

flags.dataset=edict()
flags.dataset.protocal = 'ijcb_protocal_1'#'ijcb_protocal_1'

flags.paras=edict()
flags.paras.isFocalLoss= False
flags.paras.isWeightedLoss= False
flags.paras.isRealAttackPair= False #(real, print1/print2/replay1/replay2) or (real, print1, print2, replay1, replay2)
flags.paras.isAugment= False
flags.paras.num_classes = 2
flags.paras.interval_seq = 3  # interval stride between concesive frames
flags.paras.len_seq = 5   # length of video sequence
flags.paras.stride_seq = 10 # sample stride of each sample
flags.paras.stride_seq_dev=64
flags.paras.fix_len = 16
flags.paras.resize_size=[256,256]
flags.paras.resize_size_face=[128,128]
flags.paras.reshape_size=[256,256,3]
flags.paras.reshape_size_face=[128,128,3]

flags.paras.batch_size_train = 2
flags.paras.batch_size_test = 6
flags.paras.hidden_size=16
flags.paras.learning_rate= 0.01# 0.003#0.0001
flags.paras.padding_info = {'images':[256, 256, 3 * flags.paras.len_seq],
                            'maps': [32, 32, 1 * flags.paras.len_seq],
                            'masks': [32, 32, 1 * flags.paras.len_seq],
                            'labels':[1]}

flags.paras.single_ratio = 0.4# 0.9# 0.9 #0.5 #0.01
flags.paras.cla_ratio = 0.8#0.1#0.8 #0.01

flags.paras.epoch = 1000
flags.paras.epoch_eval = 2
flags.paras.shuffle_buffer=500
flags.paras.prefetch=flags.paras.batch_size_train*2

flags.display=edict()
flags.display.max_iter=300000
flags.display.display_iter=500
flags.display.log_iter=100
flags.display.summary_iter=100
flags.display.max_to_keeper=102400
