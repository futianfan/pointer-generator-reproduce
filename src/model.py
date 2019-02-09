

import os
from time import time
import numpy as np
import tensorflow as tf
from util import Hypothesis 
import data 

def mask_normalize_attention(padding_mask, attention_weight):
	""" 
		pkg:  tf 
		softmax + padding + re-normalize

		Input:
			padding_mask: B,T
			attention_weight: B,T

		Output:
			row sum is 1. attention weight 
			[[0.7, 0.1, 0.2, 0]
			[0.2, 0.3, 0.4, 0.1]
			[1, 0, 0, 0]]
	"""
	attention_weight = tf.nn.softmax(attention_weight, axis = 1)   		### 1. softmax
	padding_mask = tf.cast(padding_mask, dtype = tf.float32)
	attention_weight *= padding_mask 							   		### 2. mask
	attention_weight_sum = tf.reduce_sum(attention_weight, 1)
	return attention_weight / tf.reshape(attention_weight_sum, [-1,1])	### 3. normalize 

def test_mask_normalize_attention():
	a = [[1, 1, 1], [1, 1, 0]]
	b = np.random.random((2,3))
	padding_mask = tf.placeholder(tf.int32, shape = [2,3])
	attention_weight = tf.placeholder(tf.float32, shape = [2,3])
	new_attention_weight = mask_normalize_attention(padding_mask, attention_weight)
	with tf.Session() as sess:
		print(sess.run([attention_weight, new_attention_weight], feed_dict = {padding_mask:a, attention_weight:b}))

def linear(args, out_size, initializer, bias = True):
	if not isinstance(args, (list, tuple)):
		args = [args]
	with tf.variable_scope('linear', reuse = tf.AUTO_REUSE):
		if len(args) > 1:
			arg_mat = tf.concat(args, 1)
		else:
			arg_mat = args
		input_size = arg_mat.get_shape()[1].value 
		w_linear = tf.get_variable('linear-weight', shape = [input_size, out_size], initializer = initializer, dtype = tf.float32)
		Xw = tf.matmul(arg_mat, w_linear)
		if bias:
			b_linear = tf.get_variable('linear-bias', shape = [out_size], initializer = initializer, dtype = tf.float32)
			Xw += b_linear
	return  Xw 


def attention(encoder_output, decoder_hidden_state, padding_mask, initializer, coverage_features = None):
	"""
	Input
		encoder_output: B,T,D1,    h1,h2,...,h_D
		decoder_hidden_state:  Tuple (state.c, state.h) ([B,D2], [B,D2])  s_{t-1}
		padding_mask:   B,T   for encoder 
		coverage_features: B,T 

		paper: Get To The Point: Summarization with Pointer-Generator Networks   https://arxiv.org/pdf/1704.04368.pdf
	"""
	batch_size = encoder_output.get_shape()[0].value 
	T = encoder_output.get_shape()[1].value
	d1 = encoder_output.get_shape()[2].value
	d2 = decoder_hidden_state[0].get_shape()[1].value  

	attention_size = d1  ### D 

	with tf.variable_scope("attention", reuse = tf.AUTO_REUSE):
		v = tf.get_variable(name = 'attention-v', shape = [attention_size], dtype = tf.float32, initializer = initializer)   ### D
		W_h = tf.get_variable(name = 'attention-Wh', shape = [d1, attention_size], dtype = tf.float32, initializer = initializer)   ### D1, D
		Wh_hi = tf.tensordot(encoder_output, W_h, axes = (2,0))  ### B,T,D
		Ws_st = linear(decoder_hidden_state, attention_size, initializer)   ### B,D
		Ws_st_expand = tf.expand_dims(Ws_st, 1)    ### B,1,D

		if coverage_features != None: 
			W_c = tf.get_variable(name = "attention-Wc", shape = [attention_size])  ### !!!!!!   
			Wc_ci = tf.tensordot(tf.expand_dims(coverage_features,-1), tf.expand_dims(W_c,0), axes = 1) ### B,T,D 			
			summ = tf.nn.tanh(Wh_hi + Ws_st_expand + Wc_ci)
		else:
			summ = tf.nn.tanh(Wh_hi + Ws_st_expand)  ### B,T,D 
		e_summ = tf.tensordot(summ, v, axes = 1)  ####  B,T

	attention_weight = mask_normalize_attention(padding_mask, e_summ)   ### B,T
	attention_weight_expand = tf.expand_dims(attention_weight, -1)  ### B,T,1
	context_vectors = encoder_output * attention_weight_expand  #### B,T,D1
	context_vectors = tf.reduce_sum(context_vectors, 1)  ### B,D1
	if coverage_features != None:
		coverage_features += attention_weight
	return attention_weight, context_vectors, coverage_features



'''
def attention(encoder_output, decoder_hidden_state, padding_mask, initializer, coverage_features = None):
	"""
	Input
		encoder_output: B,T,D1,    h1,h2,...,h_D
		decoder_hidden_state:  Tuple (state.c, state.h) ([B,D2], [B,D2])  s_{t-1}
		padding_mask:   B,T   for encoder 
		coverage_features: B,T 

		paper: Get To The Point: Summarization with Pointer-Generator Networks   https://arxiv.org/pdf/1704.04368.pdf
	"""
	batch_size = encoder_output.get_shape()[0].value 
	T = encoder_output.get_shape()[1].value
	d1 = encoder_output.get_shape()[2].value
	d2 = decoder_hidden_state.get_shape()[1].value  

	attention_size = d1 

	with tf.variable_scope("attention", reuse = tf.AUTO_REUSE):
		v = tf.get_variable(name = 'attention-v', shape = [attention_size], dtype = tf.float32, initializer = initializer)   ### D
		W_h = tf.get_variable(name = 'attention-Wh', shape = [d1, attention_size], dtype = tf.float32, initializer = initializer)   ### D1, D
		W_s = tf.get_variable(name = 'attention-Ws', shape = [d2, attention_size], dtype = tf.float32, initializer = initializer)  ### D2, D
		b_attn = tf.get_variable(name = 'attention-bias', shape = [attention_size], dtype = tf.float32, initializer = initializer)  ### D

		Wh_hi = tf.tensordot(encoder_output, W_h, axes = (2,0))  ### B,T,D
		Ws_st = tf.matmul(decoder_hidden_state, W_s)   #### B,D
		Ws_st_expand = tf.expand_dims(Ws_st, 1)    ### B,1,D
		b_attn_expand = tf.expand_dims(tf.expand_dims(b_attn, 0), 0) ## 1,1,D

		if coverage_features != None: 
			W_c = tf.get_variable(name = "attention-Wc", shape = [attention_size])  ### !!!!!!   
			Wc_ci = tf.tensordot(tf.expand_dims(coverage_features,-1), tf.expand_dims(W_c,0), axes = 1) ### B,T,D 			
			summ = tf.nn.tanh(Wh_hi + Ws_st_expand + b_attn_expand + Wc_ci)
		else:
			summ = tf.nn.tanh(Wh_hi + Ws_st_expand + b_attn_expand)  ### B,T,D 
		e_summ = tf.tensordot(summ, v, axes = 1)  ####  B,T


	attention_weight = mask_normalize_attention(padding_mask, e_summ)   ### B,T
	attention_weight_expand = tf.expand_dims(attention_weight, -1)  ### B,T,1
	context_vectors = encoder_output * attention_weight_expand  #### B,T,D1
	context_vectors = tf.reduce_sum(context_vectors, 1)  ### B,D1
	if coverage_features != None:
		coverage_features += attention_weight
	return attention_weight, context_vectors, coverage_features


def decoder(decoder_input, 
			encoder_output, 
			decoder_init_state, 
			cell, 
			encoder_padding_mask, \
			initializer,
			pointer_gen = True, 
			coverage_features = None ):
	"""
		decoder_input:  B,T2,D2
		encoder_output: B,T1,D1
		decoder_init_state:  (c,h)
		cell: LSTM 
		encoder_padding_mask: B,T1
	"""
	D2 = decoder_input.get_shape()[2].value
	decoder_input = tf.unstack(decoder_input, axis = 1)  ### B,T2,D2 => list length T2 [(B,D2), (B,D2), ..., (B,D2)]
	T2 = len(decoder_input)
	state = decoder_init_state
	rnn_size = cell.state_size[0]

	output_all = []
	p_gens = []
	attention_weights = []
	context_vectors_all = []

	for i, dec_input in enumerate(decoder_input):

		### context vector 
		attention_weight, context_vectors, coverage_features = attention(
			encoder_output, 
			dec_input,  ### ?? 
			encoder_padding_mask, 
			initializer, 
			coverage_features)
		attention_weights.append(attention_weight)
		context_vectors_all.append(context_vectors)

		### rnn input
		with tf.variable_scope('rnn-input'):
			rnn_input = linear([dec_input] + [context_vectors], rnn_size, initializer)

		### rnn
		cell_output, state = cell(rnn_input, state)
		
		if pointer_gen:
			with tf.variable_scope('pointer_gen'):
				p_gen = linear([context_vectors] + [state.c] + [state.h] + [rnn_input] + [cell_output], 1, initializer)
				p_gen = tf.nn.sigmoid(p_gen)  ### (B,) 
				p_gens.append(p_gen)

		### rnn output
		with tf.variable_scope('rnn-output'):
			rnn_output = linear([cell_output] + [context_vectors], rnn_size, initializer)  ### B,D2

		output_all.append(rnn_output)  ### [(B,D2), (B,D2), ..., ]  length T2

	return output_all, p_gens, attention_weights, context_vectors_all, state 


'''




#### most important function 
def decoder(decoder_input, 
			encoder_output, 
			decoder_init_state, 
			cell, 
			encoder_padding_mask, \
			initializer,
			pointer_gen = True, 
			coverage_features = None, 
			mode = 'train'
			):
	"""
		decoder_input:  B,T2,D2
		encoder_output: B,T1,D1
		decoder_init_state:  (c,h)
		cell: LSTM 
		encoder_padding_mask: B,T1
	"""
	batch_size = decoder_input.get_shape()[0].value 
	D1 = encoder_output.get_shape()[2].value 
	D2 = decoder_input.get_shape()[2].value
	decoder_input = tf.unstack(decoder_input, axis = 1)  ### B,T2,D2 => list length T2 [(B,D2), (B,D2), ..., (B,D2)]
	T2 = len(decoder_input)
	state = decoder_init_state
	rnn_size = cell.state_size[0]

	output_all = []
	p_gens = []
	attention_weights = []

	#### init-state 
	state = decoder_init_state
	if mode in ['train', 'valid']:
		context_vectors = tf.zeros([batch_size, D1])
	elif mode == 'decode':
		_, context_vectors, coverage_features = attention(
			encoder_output, 
			state,
			encoder_padding_mask,
			initializer,
			coverage_features
			)
	#### init-state 

	for i, dec_input in enumerate(decoder_input):   ### 根据 time-length 分开  

		### 1 rnn input
		with tf.variable_scope('rnn-input'):
			rnn_input = linear([dec_input] + [context_vectors], rnn_size, initializer)

		### 2 rnn
		cell_output, state = cell(rnn_input, state)

		### 3 attention   ???????????
		if mode in ['train', 'valid']:
			attention_weight, context_vectors, coverage_features = attention(
				encoder_output, 
				state,  
				encoder_padding_mask, 
				initializer, 
				coverage_features)
		else:
			attention_weights, context_vectors, _ = attention(
				encoder_output, 
				state,  
				encoder_padding_mask, 
				initializer, 
				coverage_features)
		attention_weights.append(attention_weight)   #### [(B,T1), ....]   length is T2. 


		###### 4 post-processing 	
		### 4.1 pointer_gen 
		if pointer_gen:
			with tf.variable_scope('pointer_gen'):
				p_gen = linear([context_vectors] + [state.c] + [state.h] + [rnn_input] + [cell_output], 1, initializer)
				p_gen = tf.nn.sigmoid(p_gen)  ### (B,) 
				p_gens.append(p_gen)   ### list of [(B,), (B,), (B,) ..., ]

		### 4.2 rnn output
		with tf.variable_scope('rnn-output'):
			rnn_output = linear([cell_output] + [context_vectors], rnn_size, initializer)  ### B,D2
		output_all.append(rnn_output)  ### [(B,D2), (B,D2), ..., ]  length T2  

	return output_all, p_gens, attention_weights, coverage_features, state 



	
def AttentionWeight_2_VocabWeight(attention_weight, encoder_index, vocab_size):
	"""
		T is unknown 
		attention_weight:  B,T    each element is a weight. 
		encoder_index:  B,T     each element is an index
		vocab_size: integer.

		Return:
		vocab_weight:  B, vocab_size
	"""
	B = attention_weight.get_shape()[0].value
	T = attention_weight.get_shape()[1].value 
	shapes = [B, vocab_size]
	indices = tf.range(0,B)  ### (B,)
	idx = tf.zeros_like(attention_weight, dtype = tf.int32)  ## (B,T) T is unknown 
	batch_idx = tf.expand_dims(tf.range(0,B),1)  ## (B,1)
	idx += batch_idx
	#indices = tf.expand_dims(indices, 1)  #### (B,1)
	#indices = tf.tile(indices, [1,T])  ### (B,T)
	indices = tf.stack([idx, encoder_index], 2)
	vocab_weight = tf.scatter_nd(indices, attention_weight, shapes)  #### B, vocab_size
	return vocab_weight




class SummarizeModel(object):
	"""
		seq2seq  
			pointer-generator
			coverage
	"""

	def __init__(self, **config):
		self.__dict__.update(**config)
		with tf.device("/gpu:0"):
			self._build_graph()


	def _build_graph(self):
		t1 = time()
		self._placeholder()
		self._random_initializer()
		self._encode_embedding()
		self._encode_rnn()		
		self._state_transformer()
		self._decoder()
		self._decoder_projection_vocab()   ### max oov 
		if self.mode in ['train', 'valid']:
			self._compute_loss()
			if self.mode == 'train':
				self._train_op()
		elif self.mode == 'decode':
			self._beam_search()

		#self._session()
		t2 = time()
		print('build graph cost {} seconds'.format(int(t2-t1)))

	def _placeholder(self):
		self._encode_batch = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = 'encode_batch')
		self._encode_lens = tf.placeholder(tf.int32, shape = [self.batch_size], name = 'encode_lens')
		self._encode_padding_mask = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = 'encode_padding_mask')

		if self.pointer_gen:
			self._encode_batch_extend_vocab = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = 'encode_batch_extend_vocab')
			self._max_num_oovs = tf.placeholder(tf.int32, shape = [], name = 'max_num_oovs')

		self._decode_batch = tf.placeholder(tf.int32, shape = [self.batch_size, self.max_dec_steps], name = 'decode_batch')
		self._target_batch = tf.placeholder(tf.int32, shape = [self.batch_size, self.max_dec_steps], name = 'target_batch')
		self._decode_padding_mask = tf.placeholder(tf.int32, shape = [self.batch_size, self.max_dec_steps], name = 'decode_padding_mask')
		if self.coverage and self.mode == 'decode':
			self._prev_coverage = tf.placeholder(tf.float32, shape = [self.batch_size, None], name = 'previous_coverage')

	def _random_initializer(self):
		self.rand_unif_init = tf.random_uniform_initializer(-self.rand_unif_init_mag, self.rand_unif_init_mag, seed=123)
		self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self.trunc_norm_init_std)

	def _encode_embedding(self):
		with tf.variable_scope('encode-embedding'):
			self.embed_mat = tf.get_variable(
				name = 'embedding-matrix', 
				shape = [self.vocab_size, self.emb_dim], 
				dtype = tf.float32, 
				initializer = self.trunc_norm_init)
			self._encode_embedded = tf.nn.embedding_lookup(
											params = self.embed_mat,
											ids = self._encode_batch)
			### B, T, D batch-major 
			self._decode_embedded = tf.nn.embedding_lookup(
											params = self.embed_mat, 
											ids = self._decode_batch)

	def _encode_rnn(self):
		with tf.variable_scope('encode-rnn'):
			self.cell_fw = tf.contrib.rnn.LSTMCell(
				num_units = self.hidden_dim, 
				initializer = self.rand_unif_init,
				state_is_tuple = True)
			self.cell_bw = tf.contrib.rnn.LSTMCell(
				num_units = self.hidden_dim,
				initializer = self.rand_unif_init,
				state_is_tuple = True)
			self.encode_output, (self.encode_final_state_fw, self.encode_final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw = self.cell_fw,
				cell_bw = self.cell_bw,
				inputs = self._encode_embedded,
				sequence_length = self._encode_lens,
				dtype = tf.float32,
				time_major = False,
				swap_memory = True
				)
			self.encode_output = tf.concat(self.encode_output, 2)  ### B, T, 2D 

	def _state_transformer(self):
		with tf.variable_scope("state-transformer"):
			w_state_h = tf.get_variable(
				name='state-transformer-h',
				shape = [self.hidden_dim * 2, self.hidden_dim],
				dtype = tf.float32,
				initializer = self.trunc_norm_init)
			w_state_c = tf.get_variable(
				name = 'state-transformer-c',
				shape = [self.hidden_dim * 2, self.hidden_dim],
				dtype = tf.float32,
				initializer = self.trunc_norm_init)
			b_state_h = tf.get_variable(
				name = 'state-transformer-h2',
				shape = [self.hidden_dim],
				dtype = tf.float32,
				initializer = self.trunc_norm_init
				)
			b_state_c = tf.get_variable(
				name = 'state-transformer-c2',
				shape = [self.hidden_dim],
				dtype = tf.float32,
				initializer = self.trunc_norm_init
				)
			self.encode_final_state_c = tf.concat([self.encode_final_state_fw.c, self.encode_final_state_bw.c], 1)  #### B,D => B,2D
			self.encode_final_state_h = tf.concat([self.encode_final_state_fw.h, self.encode_final_state_bw.h], 1)
			self.decode_init_state_c = tf.nn.relu(tf.matmul(self.encode_final_state_c, w_state_c) + b_state_c)  ### B,2D => B,D
			self.decode_init_state_h = tf.nn.relu(tf.matmul(self.encode_final_state_h, w_state_h) + b_state_h)
			self.decode_init_state = tf.contrib.rnn.LSTMStateTuple(self.decode_init_state_c, self.decode_init_state_h)

	def _decoder(self):
		'''
			mode = 'train' or 'eval'
			mode = 'decode'

			if self.coverage ?  
			if self.pointer_gen ?

		'''
		self.decode_rnn = tf.contrib.rnn.LSTMCell( 
			num_units = self.hidden_dim,
			initializer = self.rand_unif_init,
			state_is_tuple = True
			)
		##leng = self._encode_batch.get_shape()[1].value
		leng = tf.shape(self._encode_padding_mask)[1]  ##  _encode_padding_mask, _encode_batch  have equal size 
		if not self.coverage:
			coverage_features = None
		elif self.coverage and self.mode in ['train', 'valid']:
			coverage_features = tf.Variable(tf.zeros(shape = [self.batch_size, 1]))
			coverage_features = tf.tile(coverage_features, [1, leng])
		elif self.coverage and self.mode == 'decode':
			coverage_features = self._prev_coverage
		self.output_all, self.p_gens, self.attention_weights, self.coverage_features, self.decoder_final_state = decoder(
														decoder_input = self._decode_embedded,   ### B,T2,D2
														encoder_output = self.encode_output, 	### B,T1,D1
														decoder_init_state = self.decode_init_state, 
														cell = self.decode_rnn, 
														encoder_padding_mask = self._encode_padding_mask, ## B,T1
														initializer = self.rand_unif_init, 
														pointer_gen = self.pointer_gen,    ### True False
														coverage_features = coverage_features)  ## tf.zeros or None 

	def _decoder_projection_vocab(self):
		with tf.variable_scope('projection'):
			w_proj = tf.get_variable(
						'projection-w', 
						shape = [self.hidden_dim, self.vocab_size], 
						initializer = self.rand_unif_init)
			b_proj = tf.get_variable(
						'projection-b',
						shape = [self.vocab_size],
						initializer = self.rand_unif_init)
			self.projection_all = []
			self.projection_logits_all = []
			for i, dec_output in enumerate(self.output_all):
				projection_logits = tf.matmul(dec_output, w_proj) + b_proj
				self.projection_logits_all.append(projection_logits)
				projection = tf.nn.softmax(projection_logits, axis = 1)
				
				if self.pointer_gen:
					attention_weight = self.attention_weights[i]
					p_gen = self.p_gens[i]   #### (B,) 
					vocab_size_extend = self.vocab_size + self._max_num_oovs
					#print(type(projection))
					extend_projection = tf.zeros([self.batch_size, self._max_num_oovs])
					extend_projection = tf.concat([projection, extend_projection], 1)
					#print(type(extend_projection))
					pointer_gen_weight = AttentionWeight_2_VocabWeight(
											attention_weight, 
											self._encode_batch_extend_vocab, 
											vocab_size_extend)  	#### _encode_batch_extend_vocab, _encode_batch !!! ???
					projection = p_gen * extend_projection + (1 - p_gen) * pointer_gen_weight

				self.projection_all.append(projection)

	"""
	AttentionWeight_2_VocabWeight(attention_weight, encoder_index, vocab_size):
		attention_weight:  B,T    each element is a weight. 
		encoder_index:  B,T     each element is an index
		vocab_size: integer.

		Return:
		vocab_weight:  B, vocab_size
	"""

	def _compute_loss(self):
		if self.pointer_gen:
			## consider oov      ***gather_nd****
			## input:  
			##   prediction		projection_all: list of length T2, [(B,vocabsize),(B,vocabsize),(B,vocabsize)]
			##   target 		self._target_batch   B,T2,    each element is a integer index of word
			loss_all = []
			for i, projection in enumerate(self.projection_all):
				idx = self._target_batch[:,i]  ### (B,)
				batch_index = tf.range(0, self.batch_size)
				indices = tf.stack([batch_index, idx], 1)
				values = tf.gather_nd(params = projection, indices = indices)
				values = tf.maximum(values, self.minimum_log_value)
				values = -tf.log(values)
				loss_all.append(values)
			loss_mat = tf.stack(loss_all, 1)  ### B,T2 
			_decode_padding_mask = tf.cast(self._decode_padding_mask, dtype = tf.float32)
			loss_mat *= _decode_padding_mask   ### (B,T2)
			loss_mat = tf.reduce_sum(loss_mat, 1)  ### (B,)
			decode_leng = tf.reduce_sum(_decode_padding_mask, 1) ### (B,)
			self._loss = tf.reduce_mean(loss_mat / decode_leng)


		else:
			_decode_padding_mask = tf.cast(self._decode_padding_mask, dtype = tf.float32)			
			self.projection_logits_all = tf.stack(self.projection_logits_all, 1)
			self._loss = tf.contrib.seq2seq.sequence_loss(
				logits = self.projection_logits_all,
				targets = self._target_batch,
				weights = _decode_padding_mask
				)

		if self.coverage:
			## compute coverage loss
			pass 

	def _train_op(self):
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self._loss)

	def _beam_search(self):
		assert len(self.projection_all) == 1
		projection = self.projection_all[0]
		topk_prob, self.topk_id = tf.nn.top_k(projection, self.batch_size * 2)  ## batch_size * 2?
		self.log_prob = tf.log(topk_prob)


	
	def _make_feed_dict(self, batch, only_enc = False):
		"""
			only_enc: 
				True  
				False
		"""
		feed_dict = {}
		feed_dict[self._encode_batch] = batch.enc_batch
		feed_dict[self._encode_lens] = batch.enc_lens
		feed_dict[self._encode_padding_mask] = batch.enc_padding_mask

		if self.pointer_gen:
			feed_dict[self._encode_batch_extend_vocab] = batch.enc_batch_extend_vocab
			feed_dict[self._max_num_oovs] = batch.max_art_oovs

		if not only_enc:
			feed_dict[self._decode_batch] = batch.dec_batch
			feed_dict[self._target_batch] = batch.target_batch
			feed_dict[self._decode_padding_mask] = batch.dec_padding_mask

		return feed_dict

	'''
	def train(self, batch):
		feed_dict = self._make_feed_dict(batch, only_enc = False)
		_, loss = self.sess.run([self.train_fn, self._loss], feed_dict) 
		return loss 
	'''

	### for 'train'
	def model_train(self, batch, sess):
		feed_dict = self._make_feed_dict(batch, only_enc = False)
		to_return = dict()
		to_return['train_op'] = self.train_fn 
		to_return['loss'] = self._loss
		return sess.run(to_return, feed_dict) 


	### for 'valid'
	def model_valid(self, batch, sess):
		feed_dict = self._make_feed_dict(batch, only_enc = False)
		to_return = dict()
		to_return['loss'] = self._loss
		return sess.run(to_return, feed_dict)

	############################################################################################################
	############################################################################################################
	### for 'decode'
	def model_encode(self, batch, sess):
		feed_dict = self._make_feed_dict(batch, only_enc = True)
		encoder_output, decode_init_state = sess.run([self.encode_output, self.decode_init_state], feed_dict)
		#dec_in_state = tf.contrib.rnn.LSTMStateTuple(decode_init_state.c[0], decode_init_state.h[0])  
		dec_in_state = tf.contrib.rnn.LSTMStateTuple(decode_init_state.c[0], decode_init_state.h[0])  
		return encoder_output, dec_in_state

	def model_decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_states, prev_coverage):
		"""
		Args
			sess
			batch  batch.enc_padding_mask, ... , 
			latest_tokens: [B,1] decoder's input 
			enc_states:  B,T1,D1
			dec_states  decoder's hidden state for previous step.   
			prev_coverage 

		Returns
		"""
		### 1. to_return
		to_return = dict()
		to_return['topk_id'] = self.topk_id
		to_return['log_prob'] = self.log_prob
		to_return['dec_state'] = self.decoder_final_state
		to_return['attn_dist'] = self.attention_weights 

		### 2. feed_dict
		feed_dict = {}
		feed_dict[self._decode_batch] = latest_tokens    #### => self._decode_embedded
		feed_dict[self.encode_output] = enc_states
		feed_dict[self.decode_init_state] = dec_states  ### LSTMStateTuple(c, h)
		feed_dict[self._encode_padding_mask] = batch.enc_padding_mask

		if self.coverage:  
			to_return['coverage_features'] = self.coverage_features
			feed_dict[self._prev_coverage] = prev_coverage

		if self.pointer_gen:  
			to_return['p_gens'] = self.p_gens		
			feed_dict[self._encode_batch_extend_vocab] = batch.enc_batch_extend_vocab
			feed_dict[self._max_num_oovs] = batch.max_art_oovs 

		'''
			decoder(
				decoder_input = self._decode_embedded,   ### B,T2,D2
				encoder_output = self.encode_output, 	### B,T1,D1
				decoder_init_state = self.decode_init_state, 
				cell = self.decode_rnn, 														
				encoder_padding_mask = self._encode_padding_mask, ## B,T1														
				initializer = self.rand_unif_init, 														
				pointer_gen = self.pointer_gen,    ### True False	
				coverage_features = coverage_features) 
		'''

		### 3. sess.run 
		return_dict = sess.run(to_return, feed_dict)
		if not self.coverage:
			return_dict['coverage_features'] = [None for _ in range(self.batch_size)]
		if not self.pointer_gen:
			return_dict['p_gens'] = [None for _ in range(self.batch_size)]
		### 4. post-processing 

		return  return_dict['topk_id'], \
				return_dict['log_prob'], \
				return_dict['dec_state'], \
				return_dict['attn_dist'], \
				return_dict['p_gens'], \
				return_dict['coverage_features']


	def model_decode(self, batch, sess, vocab):
		'''
			self.model_encode + multiple * "self.model_decode_onestep"
		'''
		encoder_output, dec_in_state = self.model_encode(batch, sess)

		### initial state 
		coverage_init = np.zeros([batch.enc_batch.shape[1]])   	
		hypothesis_lst = [Hypothesis(tokens = [vocab.word2id(data.START_DECODING)],
									 log_probs = [0.0],
									 state = dec_in_state, 
									 attn_dists = [], 
									 p_gens = [], 
									 coverage = coverage_init) 
							for _ in range(self.batch_size)]
		results = []


		step = 0
		while step < self.max_dec_steps2 and len(results) < self.batch_size:
			### 1. preprocessing:  change format? 
			### 1.1 lastest token
			latest_tokens = [h.latest_token for h in hypothesis_lst]
			latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens]
			latest_tokens = np.array(latest_tokens).reshape(-1, 1)
			latest_tokens.astype(np.int32)
			### 1.2 decoder hidden state
			states = [h.state for h in hypothesis_lst]
			states_c = [h.state[0].reshape(1,-1) for h in hypothesis_lst]
			states_c = np.concatenate(states_c, 0)
			states_h = [h.state[1].reshape(1,-1) for h in hypothesis_lst]
			states_h = np.concatenate(states_h, 0) 
			states = (states_c, states_h)

			### 1.3 prev coverage 
			prev_coverage = [h.coverage for h in hypothesis_lst]
			prev_coverage = [i.reshape(1,-1) for i in prev_coverage]
			prev_coverage = np.concatenate(prev_coverage, 0)

			(topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = self.model_decode_onestep(
						sess=sess,
						batch=batch,
						latest_tokens=latest_tokens,
						enc_states=encoder_output,
						dec_states=states,
						prev_coverage=prev_coverage)

			all_hypos = []
			num_orig_hyps = 1 if step == 0 else len(hypothesis_lst)
			# On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
			for i in range(num_orig_hyps):
				h, new_state, attn_dist, p_gen, new_coverage_i = hypothesis_lst[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i]  
				# take the ith hypothesis and new decoder state info
				for j in range(self.batch_size * 2):  # for each of the top 2*beam_size hyps:
					# Extend the ith hypothesis with the jth option
					new_hyp = h.extend(token=topk_ids[i, j],
                           			   log_prob=topk_log_probs[i, j],
                           			   state=new_state,
                          			   attn_dist=attn_dist,
                          			   p_gen=p_gen,
                          			   coverage=new_coverage_i)
					all_hypos.append(new_hyp)



			# Filter and collect any hypotheses that have produced the end token.
			hyps = [] # will contain hypotheses for the next step
			for h in sort_hyps(all_hypos): # in order of most likely h
				if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
				# If this hypothesis is sufficiently long, put in results. Otherwise discard.
					if step >= self.min_dec_steps:
						results.append(h)
				else: # hasn't reached stop token, so continue to extend this hypothesis
					hyps.append(h)
				if len(hyps) == self.batch_size or len(results) == self.batch_size:
					# Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
					break

			step += 1

		# At this point, either we've got beam_size results, or we've reached maximum decoder steps

		if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
			results = hyps

		# Sort hypotheses by average log probability
		hyps_sorted = sort_hyps(results)

		# Return the hypothesis with highest average log prob
		return hyps_sorted[0]


def sort_hyps(hyps):
	"""Return a list of Hypothesis objects, sorted by descending average log probability"""
	return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


if __name__ == '__main__':
	
	from config import get_config, valid_config, decode_config
	#config = get_config()
	#config = valid_config()
	config = decode_config()
	model = SummarizeModel(**config)
	

	#test_mask_normalize_attention()

	pass







