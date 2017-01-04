import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def corrcoef(a, b):
	# a, b-->numpy.ndarray
	sum_a = a.sum()
	sum_b = b.sum()

	dot_prod = np.dot(a, b)
	sum_a2 = (a ** 2).sum()
	sum_b2 = (b ** 2).sum()

	num = dot_prod - (float(sum_a) * float(sum_b) / len(a))

	den = np.sqrt((sum_a2 - float(sum_a ** 2) / len(a)) * (sum_b2 - float(sum_b ** 2) / len(a)))

	return num / den

def accuAT1(strength_gold, strength_pred):
	# a, b-->numpy.ndarray
	true_labels = []
	assert len(strength_pred) == len(strength_gold)
	n_pred = len(strength_pred)
	good_pred = 0.0
	for strength in strength_gold:
		if strength[0] == strength[1]:
			true_labels.append(2)
		else:
			true_labels.append(np.argmax(strength))
	for i in xrange(n_pred):
		if true_labels[i] == 2:
			good_pred += 1.0
		else:
			if np.argmax(strength_pred[i]) == true_labels[i]:
				good_pred += 1.0
	return good_pred / n_pred

def RMSE(gold, pred):
	# root mean square error
	assert gold.shape == pred.shape
	return mean_squared_error(gold, pred) ** 0.5

def MAE(gold, pred):
	assert gold.shape == pred.shape
	return mean_absolute_error(gold, pred)

def evaluate(strength_gold, strength_pred):
	# strength_gold, strength_pred-->numpy array
	accuracy = accuAT1(strength_gold, strength_pred)
	n_test = len(strength_gold)

	pos_gold = strength_gold[:, 0]
	pos_pred = strength_pred[:, 0]
	# averaged pearson correlation coefficient
	#ap = corrcoef(a=pos_gold, b=pos_pred)
	mae = MAE(gold=strength_gold, pred=strength_pred)
	rmse = RMSE(gold=strength_gold, pred=strength_pred)
	return accuracy, mae, rmse



