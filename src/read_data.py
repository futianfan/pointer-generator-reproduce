
import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
import subprocess




def example_generator(filename):
	reader = open(filename, 'rb')
	while True:
		len_bytes = reader.read(8)
		if not len_bytes: break # finished reading this file
		str_len = struct.unpack('q', len_bytes)[0]
		example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
		yield example_pb2.Example.FromString(example_str)


if __name__ == '__main__':
	#child1 = subprocess.Popen(["ls","-data/chunked/train_*.bin"], stdout=subprocess.PIPE)
	#child1 = subprocess.Popen(["ls", "data/chunked"])
	#print(child1)

	
	filename = 'data/chunked/train_204.bin'
	iterator = example_generator(filename)
	leng = 0 
	for it in iterator:
		print(it)
		leng += 1
	print(leng)
	