
import os 

def get_config():
	'''
		train 

	'''
	config = {}

	config['data_folder'] = 'data'
	config['vocab_file'] = os.path.join(config['data_folder'], 'vocab')
	config['vocab_size'] = 20000  ### 50000 

	config['log_folder'] = 'log'
	config['train_folder'] = os.path.join(config['log_folder'], 'train')

	config['mode'] = 'train'
	config['hidden_dim'] = 100  ### 256
	config['emb_dim'] = 100   ### 128 
	config['batch_size'] = 16  ## 16 
	config['max_enc_steps'] = 400   ## 400
	config['max_dec_steps'] = 100  ### 100 
	config['lr'] = 0.1
	config['adagrad_init_acc'] = 0.1
	config['rand_unif_init_mag'] = 0.01  ### random initialization  ***0.02***
	config['trunc_norm_init_std'] = 0.001  ### random initialization ***0.0001*** 
	config['max_grad_norm'] = 2.0
	config['pointer_gen'] = True  
	config['coverage'] = True  ###
	config['cov_loss_wt'] = 1.0 


	config['single_pass'] = False
	config['minimum_log_value'] = 1e-12  ### ???     

	config['train_iter'] = int(1e5)

	return config


def valid_config():
	'''
		validation 
	'''
	config = get_config()


	### change 
	config['mode'] = 'valid'


	### new 
	config['valid_folder'] = os.path.join(config['log_folder'], 'valid')
	config['bestmodel_save_path'] = os.path.join(config['valid_folder'], 'bestmodel')	
	config['valid_iter'] = 30 

	return config



def decode_config():
	
	valid_cfg = valid_config()
	config = valid_cfg


	### modified 
	config['batch_size'] = valid_cfg['batch_size']  ### beam-size 
	config['mode'] = 'decode'
	config['max_dec_steps2'] = valid_cfg['max_dec_steps']  ### 100 	
	config['max_dec_steps'] = 1   ### 1 ?
	config['single_pass'] = True


	### new
	config['decode_folder'] = os.path.join(config['log_folder'], 'decode')
	config['min_dec_steps'] = 10 ##  35 


	return config


if __name__ == '__main__':
	config = decode_config()
	print(config['max_dec_steps2'])







