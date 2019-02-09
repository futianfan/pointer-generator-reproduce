
import tensorflow as tf
import time
import os


def get_config():
	"""Returns config for tf.session"""
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	return config

def load_checkpoint(saver, sess, log_directory = 'log', checkpoint_directory = 'train'):

	while True:
		try:
			latest_filename = "checkpoint_best" if checkpoint_directory == 'valid' else None
			checkpoint_directory = os.path.join(log_directory, checkpoint_directory)
			checkpoint_state = tf.train.get_checkpoint_state(checkpoint_directory, latest_filename = latest_filename)
			saver.restore(sess, checkpoint_state.model_checkpoint_path)
			return checkpoint_state.model_checkpoint_path
			### sess is modified. 
		except:
			print('fail to load model')
			time.sleep(10)




class Hypothesis:
	'''
		single hypothesis 
		used for 'decode'

	'''
	def __init__(self, 	
				tokens, 	### [] 
				log_probs,  ### [0.0]
				state, 		### init: tuple?
				attn_dists, ### init: []
				p_gens, 	### init: []
				coverage    ### init: [0, 0, 0, 0, ....]  和enc等长
			):
		self.__dict__.update(locals().items())

	'''
		Example: 
			[ Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                     log_probs=[0.0],
                     state=dec_in_state,
                     attn_dists=[],
                     p_gens=[],
                     coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length
                     ) for _ in range(FLAGS.beam_size)]
	'''



	def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
		return Hypothesis(tokens = self.tokens + [token],
                          log_probs = self.log_probs + [log_prob],
                          state = state,   ####
                          attn_dists = self.attn_dists + [attn_dist],
                          p_gens = self.p_gens + [p_gen],
                          coverage = coverage)  ###

	@property
	def latest_token(self):
		return self.tokens[-1]

	@property
	def log_prob(self):
		return sum(self.log_probs)

	@property
	def avg_log_prob(self):
		return self.log_prob / len(self.tokens)



