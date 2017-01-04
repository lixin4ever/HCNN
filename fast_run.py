import os
import sys

data_sets = ['bbc', 'digg', 'MySpace', 'rw', 'Twitter', 'YouTube']

if __name__ == '__main__':
	mode = sys.argv[1] 
	total = []
	for ds1 in data_sets:
		total.append(ds1)
		for ds2 in data_sets:
			if ds2 == ds1:
				continue
			total.append('%s_%s' % (ds1, ds2))
	
	if mode == 'baseline':
		for ds in total:
			res = os.system('python baseline.py CNN %s' % ds)

		for ds in total:
			res = os.system('python baseline.py LSTM %s' % ds)
	elif mode == 'HCNN':
		for ds in total:
			res = os.system('THEANO_FLAGS="device=gpu3" python model.py HCNN %s' % ds)
	elif mode == 'HCNN_ablation':
		for ds in data_sets:
			res = os.system('python model.py HCNN_no_glove %s' % ds)

		for ds in data_sets:
			res = os.system('python model.py HCNN_no_onehot %s' % ds)

		for ds in data_sets:
			res = os.system('python model.py HCNN_no_POS %s' % ds)
