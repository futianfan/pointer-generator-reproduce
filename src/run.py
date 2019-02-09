"""

python src/run.py --data_path=./data/chunked/train_* --mode=train

python src/run.py --data_path=./data/chunked/val_* --mode=valid

python src/run.py --data_path=./data/chunked/test_* --mode=decode


"""

import numpy as np
import tensorflow as tf
from collections import namedtuple
from time import time
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import util 
from config import get_config, valid_config, decode_config
from data import Vocab 
from batcher import Batcher
################################################################################################################################################################################
################################################################################################################################################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or valid')

################################################################################################################################################################################
################################################################################################################################################################################

def train():
	config = get_config()
	### Create Vocabulary
	vocab = Vocab(config['vocab_file'], config['vocab_size'])
	### Create batch
	hps = namedtuple("HParams", config.keys())(**config)
	batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=config['single_pass'])
	### Model
	from model import SummarizeModel
	model = SummarizeModel(**config)
	setup_training(model, batcher, config)

def valid():
	config = valid_config()
	### Create Vocabulary
	vocab = Vocab(config['vocab_file'], config['vocab_size'])
	### Create batch
	hps = namedtuple("HParams", config.keys())(**config)
	batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=config['single_pass'])
	### Model
	from model import SummarizeModel
	model = SummarizeModel(**config)
	setup_valid(model, batcher, config)

def decode():
	config = decode_config()
	vocab = Vocab(config['vocab_file'], config['vocab_size'])
	hps = namedtuple("HParams", config.keys())(**config)
	batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=config['single_pass'])
	from model import SummarizeModel
	from decode import BeamSearchDecoder
	### Model
	model = SummarizeModel(**config)
	beamsearch_model = BeamSearchDecoder(model, batcher, vocab)
	beamsearch_model.decode()


'''
def setup_decode(model, batcher, config):
	decode_directory = config['decode_folder']
	valid_directory = config['valid_folder']
	train_directory = config['train_folder']

	if not os.path.exists(decode_directory):
		os.makedirs(decode_directory)

	saver = tf.train.Saver()  ### max_to_keep = 3 ? 
	sess = tf.Session(config = util.get_config())
	_ = util.load_checkpoint(saver = saver,
							 sess = sess, 
							 log_directory = config['decode_folder'],
							 checkpoint_directory = 'train'  ### valid
							)

	### batch
	batch = batcher.next_batch()

	from decode import run_beam_search

	### un-finished 
'''


def setup_valid(model, batcher, config):
	valid_directory = config['valid_folder']

	if not os.path.exists(valid_directory):
		os.makedirs(valid_directory)
	saver = tf.train.Saver(max_to_keep = 3)
	sess = tf.Session(config=util.get_config())
	#sess = tf.Session()
	_ = util.load_checkpoint(saver = saver, 
						 sess = sess, 
						 log_directory = config['log_folder'], 
						 checkpoint_directory = 'train')  ### valid or train ??? 
	best_valid_loss = None

	while True:
		valid_loss = 0 
		for i in range(config['valid_iter']):
			batch = batcher.next_batch()
			to_return =  model.model_valid(batch, sess)
			valid_loss += to_return['loss']

		### if blabla 
		if best_valid_loss == None or valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			saver.save(sess, config['bestmodel_save_path'], latest_filename='checkpoint_best')
			print('save checkpoint')
		else:
			print('not update checkpoint')



def setup_training(model, batcher, config):
	train_directory = config['train_folder']
	if not os.path.exists(train_directory):
		os.makedirs(train_directory)

	saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time
	sv = tf.train.Supervisor(logdir=train_directory,
				is_chief=True,
				saver=saver,
				summary_op=None,
				save_summaries_secs=60, # save summaries for tensorboard every 60 secs
				save_model_secs=60, # checkpoint every 60 secs
				#global_step=model.global_step
				)
				
	#sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
	sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())   ###   managed_session  
	with sess_context_manager as sess:
		loss_lst = []
		loss_long, loss_all = 0, 0
		for it in range(config['train_iter']):
			t1 = time()		
			batch = batcher.next_batch()
			to_return = model.model_train(batch, sess)
			loss = to_return['loss']
			loss_all += loss 
			loss_long += loss 
			t2 = time()
			#print(t2 - t1)
			if it > 0 and it % 100 == 0:
				print('iter: {}, loss: {}'.format(it, str(loss_all)[:6]))
				loss_all = 0 
			if it % 1000 == 0 and it > 0:
				loss_lst.append(loss_long)
				print(loss_lst)
				loss_long = 0




if __name__ == '__main__':
	import sys
	if FLAGS.mode == 'train':
		train()
	elif FLAGS.mode == 'valid':
		valid()
	elif FLAGS.mode == 'decode':
		decode()




"""
def main_old():
	config = get_config()
	### Create Vocabulary
	vocab = Vocab(config['vocab_file'], config['vocab_size'])
	### Create batch
	hps = namedtuple("HParams", config.keys())(**config)
	batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=config['single_pass'])
	### Model
	from model import SummarizeModel
	model = SummarizeModel(**config)
	### train 
	loss_lst = []
	loss_long, loss_all = 0, 0
	for it in range(config['train_iter']):
		t1 = time()		
		batch = batcher.next_batch()
		##print(batch.enc_batch)
		loss = model.train(batch)
		loss_all += loss 
		loss_long += loss 
		t2 = time()
		#print(int(t2 - t1), end = ' ')
		if it > 0 and it % 100 == 0:
			print('iter: {}, loss: {}'.format(it, str(loss_all)[:6]))
			loss_all = 0 
		if it % 1000 == 0 and it > 0:
			loss_lst.append(loss_long)
			print(loss_lst)
			loss_long = 0
	return 


def data_test():
	config = get_config()
	### Create Vocabulary
	vocab = Vocab(config['vocab_file'], config['vocab_size'])
	### Create batch
	hps = namedtuple("HParams", config.keys())(**config)
	batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=config['single_pass'])

	batch = batcher.next_batch()
	batch = batcher.next_batch()
	batch = batcher.next_batch()
	print('encoder batch ')
	print(batch.enc_batch)
	print(batch.enc_batch.shape)
	print(batch.enc_lens)
	#print(batch.enc_padding_mask)
	print(batch.enc_padding_mask.shape)
	print(np.sum(batch.enc_padding_mask,1))
	print('oov')
	print(batch.enc_batch_extend_vocab)
	print(batch.enc_batch_extend_vocab.shape)
	print(batch.max_art_oovs)
	print('decoder batch')
	print(batch.dec_batch.shape)
	#print(batch.dec_batch)
	print(batch.target_batch.shape)
	#print(batch.target_batch)
	print(batch.dec_padding_mask.shape)
	print((batch.enc_batch_extend_vocab > config['vocab_size']).any())
	print((batch.enc_batch > config['vocab_size']).any())


	'''
    batch.enc_batch   
    #### (16, 400)  each element is an word index.  最多400， 也有可能是序列中最长的。
    batch.enc_lens		
    ###  (16,)   each element is a integer.  长度从小到大排列 
    batch.enc_padding_mask  
    ###  (16, 400),   each element is 1 or 0.   1 mean exist words。最多400， 也有可能是序列中最长的。
    ### padding 只在最后 padding 
    if FLAGS.pointer_gen:
		batch.enc_batch_extend_vocab     ###  16，xxx,   和enc-batch一样大  OOV给了temporary_ID 
      	batch.max_art_oovs                  integer 
    if not just_enc:
      batch.dec_batch        input,  							<START_DECODER> x1, x2, x3, x_{k-1}
      batch.target_batch     output prediction,   				x1, x2, x3, ................ xk
      batch.dec_padding_mask
	'''


"""






