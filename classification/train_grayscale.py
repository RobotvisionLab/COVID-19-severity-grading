import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os
import xlrd
import random
import numpy as np
import tqdm
import logging
import time
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd

IMAGE_SIZE = [672, 752]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

excel_name = 'grading.xlsx'
class_info = {}

train_x = []
train_y = []
test_x = []
test_y = []

def get_class_info(excel_name):
	Descriptor = xlrd.open_workbook(excel_name)
	Descriptor_Sheet = Descriptor.sheets()[0]
	sheet_header = Descriptor_Sheet.row_values(0)

	nrows = Descriptor_Sheet.nrows #行数
	ncols = Descriptor_Sheet.ncols #列数

	print('sheet_header: ', sheet_header, 'nrows: ', nrows, 'ncols: ', ncols)

	for i in range(nrows):
		if i == 0:
			continue

		rowValues = Descriptor_Sheet.row_values(i) #某一行数据
		img_name = rowValues[0]
		img_grade = int(rowValues[1])

		img_names = img_name.split('_')
		img_name = img_names[1] + '_' + img_names[2]
		# print(img_name, '---> ', img_grade)
		class_info[img_name] = img_grade

get_class_info(excel_name)

def getTrain_data(img_path, lab_path, lung_lab_path):
	l = os.listdir(img_path)
	random.shuffle(l)

	midvalue = [42.5, 127.5, 170]
	for filename in l:
		subdir, img_name = filename.split('_')

		ct_name = img_path + '/' + subdir + '_' + img_name
		lung_name = lung_lab_path + '/' + subdir + '_' + img_name
		lesion_name = lab_path + '/' + subdir + '_2_' + img_name

		ct = cv2.imread(ct_name, 0)
		lung_lab = cv2.imread(lung_name, 0)
		lesion_lab = cv2.imread(lesion_name, 0)

		lung_area = np.sum(lung_lab == 255)
		lesion_area = np.sum(lesion_lab > 0)

		lesion = ct[lesion_lab>0]

		l1 = np.sum(lesion <= 85)
		l2 = np.count_nonzero((85 < lesion) & (lesion <= 170))
		l3 = np.sum(lesion > 170)

		p1 = l1/lesion_area
		p2 = l2/lesion_area
		p3 = l3/lesion_area

		g = p1*midvalue[0] + p2*midvalue[1] + p3*midvalue[2]
		grade = class_info[filename]
		# print(lung_area, lesion_area, p1, p2, p3, g, grade)

		x = [lung_area, lesion_area, g]

		train_x.append(x)
		train_y.append(grade)

def getTrain_data_neg(img_path, lab_path, lung_lab_path):
	l = os.listdir(img_path)
	random.shuffle(l)

	midvalue = [42.5, 127.5, 170]
	for filename in l:
		subdir, img_name = filename.split('_')

		ct_name = img_path + '/' + subdir + '_' + img_name
		lung_name = lung_lab_path + '/' + subdir + '_' + img_name
		lesion_name = lab_path + '/' + subdir + '_2_' + img_name

		ct = cv2.imread(ct_name, 0)
		lung_lab = cv2.imread(lung_name, 0)
		lung_area = np.sum(lung_lab == 255)

		x = [lung_area, 0, 0]

		train_x.append(x)
		train_y.append(0)

def getTest_data(img_path, lab_path, lung_lab_path):
	l = os.listdir(img_path)
	random.shuffle(l)

	midvalue = [42.5, 127.5, 170]
	for filename in l:
		subdir, img_name = filename.split('_')

		ct_name = img_path + '/' + subdir + '_' + img_name
		lung_name = lung_lab_path + '/' + subdir + '_' + img_name
		lesion_name = lab_path + '/' + subdir + '_2_' + img_name

		# print(ct_name)
		# print(lung_name)
		# print(lesion_name)

		ct = cv2.imread(ct_name, 0)
		lung_lab = cv2.imread(lung_name, 0)
		lung_area = np.sum(lung_lab == 255)

		if os.path.exists(lesion_name):
			lesion_lab = cv2.imread(lesion_name, 0)
			lesion_area = np.sum(lesion_lab > 0)

			lesion = ct[lesion_lab>0]

			l1 = np.sum(lesion <= 85)
			l2 = np.count_nonzero((85 < lesion) & (lesion <= 170))
			l3 = np.sum(lesion > 170)

			p1 = l1/lesion_area
			p2 = l2/lesion_area
			p3 = l3/lesion_area

			g = p1*midvalue[0] + p2*midvalue[1] + p3*midvalue[2]
			grade = class_info[filename]
			# print(lung_area, lesion_area, p1, p2, p3, g, grade)

			x = [lung_area, lesion_area, g]

			test_x.append(x)
			test_y.append(grade)
		else:
			x = [lung_area, 0, 0]

			test_x.append(x)
			test_y.append(0)


def get_logger(log_path='log_path'):
	if not os.path.exists(log_path):
		os.mkdir(log_path)
	timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
	txthandle = logging.FileHandler((log_path+'/'+timer+'log.txt'))
	txthandle.setFormatter(formatter)
	logger.addHandler(txthandle)
	return logger

def del_models(file_path, count=5):
	dir_list = os.listdir(file_path)
	if not dir_list:
		print('file_path is empty: ', file_path)
		return
	else:
		dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
		print('dir_list: ', dir_list)
		if len(dir_list) > 5:
			os.remove(file_path + '/' + dir_list[0])

		return dir_list


class BP(nn.Module):
	def __init__(self):
		super(BP, self).__init__()
 
		self.input_layer = nn.Linear(3, 5)
		self.output_layer = nn.Linear(5, 5)
		self.relu = nn.ReLU(True)

	def forward(self, x):
		x = self.input_layer(x)
		x = self.relu(x)
		x = self.output_layer(x)
		return x


class myDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.len = x.shape[0]

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

def train(net, epoch, iterations, loss_stop, train_path):
	net.train()
	epoch_loss = 0.0
	print('train...')

	fault = 0
	for iters in tqdm.tqdm(range(iterations//10)):

		inputs1, labels1 = getNormal(train_path)
		inputs2, labels2 = getMild(train_path)
		inputs3, labels3 = getModerate(train_path)
		inputs4, labels4 = getSevere(train_path)
		inputs5, labels5 = getCritical(train_path)

		inputs6, labels6 = getNormal(train_path)
		inputs7, labels7 = getMild(train_path)
		inputs8, labels8 = getModerate(train_path)
		inputs9, labels9 = getSevere(train_path)
		inputs10, labels10 = getCritical(train_path)


		inputs = torch.cat([inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10], dim=0)
		labels = torch.cat([labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10], dim=0)

		inputs = torch.as_tensor(inputs, dtype=torch.float32)

		# print(inputs.dtype, inputs.shape)
		# print(labels.dtype, labels.shape)

		optimizer.zero_grad()
		outputs = net(inputs)

		s = nn.Softmax(dim = 1)
		out = s(outputs)
		out = torch.argmax(out, dim=1).numpy()
		lab = torch.squeeze(labels).numpy()

		# print(out)
		# print(lab)

		not_equ = np.sum(out != lab)
		# print(not_equ)
		fault += not_equ

		loss = criterion(outputs, torch.squeeze(labels).long())
		loss.backward()
		optimizer.step()
		epoch_loss += loss

	epoch_loss_mean = epoch_loss / iterations
	print('Train Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}'.format(epoch, epoch_loss.item(), epoch_loss_mean.item()))
	logger.info('Train Epoch:[{}] , loss: {:.6f}'.format(epoch, epoch_loss.item()))

	print('fault: ', fault)
	print('fault ratio: ', fault/iterations)

	if epoch_loss < loss_stop:
		return True, epoch_loss
	else:
		return False, epoch_loss

if __name__ == '__main__': 

	train_img_path = '/Users/zhang/Desktop/Paper/Lesion_segmentation/ds_ct_lesion_g2/train/positive/img' 
	train_lab_path = '/Users/zhang/Desktop/Paper/Lesion_segmentation/ds_ct_lesion_g2/train/positive/lab' 
	train_lung_lab = '/Users/zhang/Desktop/Paper/train/lung_lab'
	getTrain_data(train_img_path, train_lab_path, train_lung_lab)

	train_img_path1 = '/Users/zhang/Desktop/Paper/Lesion_segmentation/ds_ct_lesion_g2/train/negative/img' 
	train_lab_path1 = '/Users/zhang/Desktop/Paper/Lesion_segmentation/ds_ct_lesion_g2/train/negative/lab' 
	train_lung_lab1 = '/Users/zhang/Desktop/Paper/train/lung_lab'
	getTrain_data_neg(train_img_path, train_lab_path, train_lung_lab)

	test_img_path = '/Users/zhang/Desktop/Paper/Lesion_segmentation/ds_ct_lesion_g2/test/img' 
	test_lab_path = '/Users/zhang/Desktop/Paper/Lesion_segmentation/ds_ct_lesion_g2/test/lab' 
	test_lung_lab = '/Users/zhang/Desktop/Paper/test/lung_lab'
	getTest_data(test_img_path, test_lab_path, test_lung_lab)

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)

	print(train_x.shape)
	print(train_y.shape)

	print(test_x.shape)
	print(test_y.shape)

	for i in range(len(train_x)):
		print(train_x[i], '-->', train_y[i])


	exit(0)
	scaler = MinMaxScaler(feature_range=(0,1))
	train_x_scale = scaler.fit_transform(train_x)
	test_x_scale = scaler.fit_transform(test_x)


	my_train_ds = myDataset(x=train_x_scale, y=train_y)
	tensor_dataloader_train = DataLoader(dataset=my_train_ds,batch_size=32,shuffle=True)  

	my_test_ds = myDataset(x=test_x_scale, y=test_y)
	tensor_dataloader_test = DataLoader(dataset=my_test_ds,batch_size=1,shuffle=True)  

	net = BP()
	criterion = nn.CrossEntropyLoss()  
	optimizer = optim.Adam(net.parameters(), lr = 0.001)
	model_path = "./checkpoint"
	log_path = "./log"

	for epoch in range(20000):

		net.train()
		epoch_loss = 0.0
		for i, data in enumerate(tensor_dataloader_train):
			inputs, labels = data
			inputs = torch.as_tensor(inputs, dtype=torch.float32)
			labels = torch.as_tensor(labels, dtype=torch.float32)

			if inputs.shape[0] != 32:
				continue

			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, torch.squeeze(labels).long())
			loss.backward()
			optimizer.step()
			epoch_loss += loss

		net.eval()
		valid_loss = 0.0

		grade = [0, 0, 0, 0, 0]
		grade_d = [0, 0, 0, 0, 0]
		grades = {'0': [0, 0, 0, 0, 0], '1': [0, 0, 0, 0, 0], '2': [0, 0, 0, 0, 0], '3': [0, 0, 0, 0, 0], '4': [0, 0, 0, 0, 0]}

		with torch.no_grad():
			for i, data in enumerate(tensor_dataloader_test):
				inputs, labels = data
				inputs = torch.as_tensor(inputs, dtype=torch.float32)
				labels = torch.as_tensor(labels, dtype=torch.float32)

				optimizer.zero_grad()
				outputs = net(inputs)

				s = nn.Softmax(dim = 1)
				out = s(outputs)
				out = torch.argmax(out, dim=1).numpy()
				lab = int(torch.squeeze(labels).numpy())

				not_equ = np.sum(out[0] != lab)
				grade[lab] += 1

				grades[str(lab)][out[0]] += 1

				if not_equ > 0:
					grade_d[lab] += 1

				loss = criterion(outputs, labels.long())
				valid_loss += loss

		if epoch % 100 == 0:
			print('epoch: ', epoch, 'epoch_loss: ', epoch_loss.item(), 'valid_loss: ', valid_loss.item())
			print(grade)
			print(grade_d)
			print('fault ratio: ', np.array(grade_d)/np.array(grade))

			print(grades['0'])
			print(grades['1'])
			print(grades['2'])
			print(grades['3'])
			print(grades['4'])

# 											#         		P   	SEN    	SPE       TP    TN 	  FP    FN         
# 											#  norm: 		                          336   369   0     0              1        1       1
# 											#  mild: 		                          0     698   0     0              0        0       1
# 											#  moderate:	                          312   374   7     3              97.80    99.05   98.16
# 											#  severe:			                      26    663   3     13             89.66    66.67   99.10
# 											#  critial:		                          8     693   4     0              66.67    1       99.43


# 											#         		P   	SEN    	SPE       TP    TN 	  FP    FN         
# 											#  norm: 		                          336   369   0     0               1       1       1
# 											#  mild: 		                          7     698   0     0               1       1       1
# 											#  moderate:	                          314   390   0     1               1       99.68   1
# 											#  severe:			                      39    665   1     0               97.50   1       99.85
# 											#  critial:		                          8     697   0     0               1       1       1	
# ####	
# #####    336,  0,	0,	0,	0        
# #####	 0,    0,	7,	0,	0        	
# #####	 0,	   0,  312, 3,	0
# #####    0,    0,   9,  26, 4
# #####    0,    0,   0,  0,  8

# ####	
# #####    336,  0,	0,	0,	0 
# #####	 0,    7,	0,	0,	0
# #####	 0,	   0,  314, 1,	0
# #####    0,    0,   0,  39, 0
# #####    0,    0,   0,  0,  8

# 真实预测      0      1      2
# 0            2      0      0       TP:对角相交的那个  TN：除了对角相交的  FP:列除了自己  FN:行除了自己
# 1            1      0      1
# 2            0      2      0

# 对于类别0的 FP=1 TP=2 FN=0 TN=3
# 对于类别1的 FP=2 TP=0 FN=2 TN=2
# 对于类别2的 FP=1 TP=0 FN=2 TN=3





