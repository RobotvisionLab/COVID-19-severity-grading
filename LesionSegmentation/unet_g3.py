import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
from tqdm import tqdm

from torchvision import models
from torchvision.models.vgg import VGG
from evaluation_new_m import *
import logging
import sys
import urllib
from thop import profile

import time
from PIL import Image
from os import listdir
import shutil
from torchstat import stat
from collections import OrderedDict
import copy

IMAGE_SIZE = [672, 752]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def show(tensor, strIndex):
	img = tensor[0][0]
	lab = tensor[1][0]
	out = tensor[2][0]
	
	img = img.detach().cpu().squeeze().numpy()
	lab = lab.detach().cpu().squeeze().numpy()
	out = out.detach().cpu().squeeze().numpy()
	
	img_name = "./savepng/" + strIndex + '_img.jpg'
	lab_name = "./savepng/" + strIndex + '_lab.jpg'
	out_name = "./savepng/" + strIndex + '_out.jpg'

	cv2.imwrite(img_name, img)
	cv2.imwrite(lab_name, lab*255)
	cv2.imwrite(out_name, out*255)


	plt.figure()
	ax1 = plt.subplot(1,3,1)
	ax1.set_title('Input')
	plt.imshow(cv2.split(img)[0], cmap="gray")
	ax2 = plt.subplot(1,3,2)
	ax2.set_title('Label')
	plt.imshow(lab, cmap="gray")
	ax3 = plt.subplot(1,3,3)
	ax3.set_title('Output')
	plt.imshow(out, cmap="gray")

	picName = './visualization/' + strIndex + '.jpg'
	plt.savefig(picName)
	plt.cla()
	plt.close("all")

def caculate(output, label, clear=False):
	cal = Bin_classification_cal(output, label, 0.5, clear)
	return cal.caculate_total()

class DoubleConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(DoubleConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)
 
	def forward(self, input):
		return self.conv(input)
 
class Unet(nn.Module):
	def __init__(self,in_ch, out_ch):
		super(Unet, self).__init__()
 
		self.conv1 = DoubleConv(in_ch, 64)
		self.pool1 = nn.MaxPool2d(2)
		self.conv2 = DoubleConv(64, 128)
		self.pool2 = nn.MaxPool2d(2)
		self.conv3 = DoubleConv(128, 256)
		self.pool3 = nn.MaxPool2d(2)
		self.conv4 = DoubleConv(256, 512)
		self.pool4 = nn.MaxPool2d(2)
		self.conv5 = DoubleConv(512, 1024)
		self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
		self.conv6 = DoubleConv(1024, 512)
		self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
		self.conv7 = DoubleConv(512, 256)
		self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
		self.conv8 = DoubleConv(256, 128)
		self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
		self.conv9 = DoubleConv(128, 64)
		self.conv10 = nn.Conv2d(64, out_ch, 1)

	def forward(self, x):
		c1 = self.conv1(x)
		p1 = self.pool1(c1)
		c2 = self.conv2(p1)
		p2 = self.pool2(c2)
		c3 = self.conv3(p2)
		p3 = self.pool3(c3)
		c4 = self.conv4(p3)
		p4 = self.pool4(c4)
		c5 = self.conv5(p4)
		up_6 = self.up6(c5)
		merge6 = torch.cat([up_6, c4], dim=1)
		c6=self.conv6(merge6)
		up_7 = self.up7(c6)
		merge7 = torch.cat([up_7, c3], dim=1)
		c7 = self.conv7(merge7)
		up_8 = self.up8(c7)
		merge8 = torch.cat([up_8, c2], dim=1)
		c8 = self.conv8(merge8)
		up_9 = self.up9(c8)
		merge9 = torch.cat([up_9, c1], dim=1)
		c9 = self.conv9(merge9)
		c10 = self.conv10(c9)
		return c10

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

def ImageBinarization(img, threshold=1):
	img = np.array(img)
	image = np.where(img > threshold, 1, 0)
	return image

def label_preprocess(label):
	label_pixel = ImageBinarization(label)
	return  label_pixel

def cvTotensor(img):
	img = (np.array(img[:, :, np.newaxis]))
	img = np.transpose(img,(2,0,1))
	img = (np.array(img[np.newaxis, :,:, :]))    
	tensor = torch.from_numpy(img)
	tensor = torch.as_tensor(tensor, dtype=torch.float32)
	return tensor

def cvTotensor_img(img):
	img = np.transpose(img,(2,0,1))
	img = (np.array(img[np.newaxis, :,:, :]))    
	tensor = torch.from_numpy(img)
	tensor = torch.as_tensor(tensor, dtype=torch.float32)
	return tensor

g = 3
iterations = 0
net = Unet(1, g+1)
# print(net)
net.cuda()

def getInput_and_Label_generator(data_path):
	img_Path = data_path + "/img"
	l = os.listdir(img_Path)
	random.shuffle(l)
	for filename in l:
		sub_dir = filename.split('_')[0]
		img_dir = filename.split('_')[1]

		img_name = img_Path + '/' + filename
		label_name = data_path + '/lab/' + sub_dir + '_' + str(g) + '_' + img_dir

		img = cv2.imread(img_name, 0)
		img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
		img = cvTotensor(img)

		if os.path.exists(label_name):
			lab = cv2.imread(label_name, 0)
			lab = cv2.resize(lab, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
		else:
			lab = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]))

		lab = cvTotensor(lab)
		yield img, lab

def getInput_and_Label_generator_valid(data_path):
	img_Path = data_path + "/img"
	l = os.listdir(img_Path)

	for filename in l:
		sub_dir = filename.split('_')[0]
		img_dir = filename.split('_')[1]

		img_name = img_Path + '/' + filename
		label_name = data_path + '/lab/' + sub_dir + '_' + str(g) + '_' + img_dir

		img = cv2.imread(img_name, 0)
		img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
		img = cvTotensor(img)

		# print(label_name)
		if os.path.exists(label_name):
			lab = cv2.imread(label_name, 0)
			lab = cv2.resize(lab, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
		else:
			lab = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]))

		lab = cvTotensor(lab)
		yield img, lab

def train(net, epoch, iterations, loss_stop, positive_path, negative_path):
	net.train()
	epoch_loss = 0.0
	print('train...')
	g_postive = getInput_and_Label_generator(positive_path)
	g_negative = getInput_and_Label_generator(negative_path)

	for iters in tqdm(range(iterations//3)):
		for index in range(2):
			if index == 0:
				inputs1, labels1 = next(g_postive)
				inputs2, labels2 = next(g_postive)
				inputs3, labels3 = next(g_postive)
			else:
				inputs1, labels1 = next(g_negative)
				inputs2, labels2 = next(g_negative)
				inputs3, labels3 = next(g_negative)

			inputs = torch.cat([inputs1, inputs2, inputs3], dim=0)
			labels = torch.cat([labels1, labels2, labels3], dim=0)

			inputs = inputs.cuda()
			labels = labels.cuda()

			optimizer.zero_grad()
			outputs = net(inputs)

			loss = criterion(outputs, torch.squeeze(labels).long())
			loss.backward()
			optimizer.step()
			epoch_loss += loss

	epoch_loss_mean = epoch_loss / iterations
	print('Train Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}'.format(epoch, epoch_loss.item(), epoch_loss_mean.item()))
	logger.info('Train Epoch:[{}] , loss: {:.6f}'.format(epoch, epoch_loss.item()))
	if epoch_loss < loss_stop:
		return True, epoch_loss
	else:
		return False, epoch_loss


mark_bgr_g3 = [[0,   0,   0], # 背景    	   黑色
			[201, 230,  253], # 类别 1      粉色
			[0, 128, 255], # 类别 2         橘色
			[0, 0, 255], # 类别 3           红色
			[0, 0, 0] # 类别 4              黑色
			]

def colored(img):
	img = img.astype(dtype="uint8")
	output = np.zeros([img.shape[0], img.shape[1], 3], dtype="uint8")
	for r in range(img.shape[0]):
		for c in range(img.shape[1]):
			output[r][c] = mark_bgr_g3[img[r][c]]

	return output



# def getMetrixs(out, lab):

# 	s = nn.Softmax(dim = 1)
# 	out = s(out)
# 	out = torch.argmax(out, dim=1)

# 	ious = get_ious(lab, out)

# 	lab = lab.detach().cpu().squeeze().numpy()
# 	out = out.detach().cpu().squeeze().numpy()

# 	matrix = cal_confu_matrix(lab, out, g+1)
# 	mious = get_mious(matrix)
# 	mpa = get_mPa(matrix)
# 	pa = get_pa(matrix)
# 	cpa = get_cpa(matrix)[g]

# 	return ious, mious, mpa, pa, cpa

def getMetrixs_pro(out, lab):

	s = nn.Softmax(dim = 1)
	out = s(out)
	out = torch.argmax(out, dim=1)

	ious_total = get_ious_total(lab, out)
	dices_total = get_dices_total(lab, out)
	sens_total = get_sens_total(lab, out)
	spes_total = get_spes_total(lab, out)

	iou_1 = get_ious(lab, out, g=1)
	iou_2 = get_ious(lab, out, g=2)
	iou_3 = get_ious(lab, out, g=3)

	dice_1 = get_dices(lab, out, g=1)
	dice_2 = get_dices(lab, out, g=2)
	dice_3 = get_dices(lab, out, g=3)

	sen_1 = get_sens(lab, out, g=1)
	sen_2 = get_sens(lab, out, g=2)
	sen_3 = get_sens(lab, out, g=3)

	spe_1 = get_spes(lab, out, g=1)
	spe_2 = get_spes(lab, out, g=2)
	spe_3 = get_spes(lab, out, g=3)

	lab = lab.detach().cpu().squeeze().numpy()
	out = out.detach().cpu().squeeze().numpy()

	matrix = cal_confu_matrix(lab, out, g+1)
	mious = get_mious(matrix)
	mious = get_mious(matrix)
	cpa = get_cpa(matrix)
	mpa = get_mPa(matrix)

	return ious_total, dices_total, sens_total, spes_total, [iou_1,iou_2,iou_3], [dice_1,dice_2,dice_3], [sen_1,sen_2,sen_3], [spe_1,spe_2,spe_3], mious, cpa, mpa


def valid(net, epoch, img_path):
	#net.eval()
	valid_loss = 0.0
	img_Path = img_path + "/img"
	l = os.listdir(img_Path)
	iterations = len(l)
	print('img_Path: ', img_Path, 'len: ', iterations)

	index = 0

	ious_totals = []
	dices_totals = []
	sens_totals = []
	spes_totals = []

	iou_singles = np.array([0, 0, 0])
	dice_singles = np.array([0, 0, 0])
	sen_singles = np.array([0, 0, 0])
	spe_singles = np.array([0, 0, 0])

	miouss = []
	mpas = []
	cpass = np.array([0, 0, 0])

	g_data = getInput_and_Label_generator_valid(img_path)
	with torch.no_grad():
		for iters in tqdm(range(iterations)):
			inputs, labels = next(g_data)

			inputs = inputs.cuda()
			labels = labels.cuda()

			optimizer.zero_grad()
			outputs = net(inputs)
			if torch.sum(labels) > 0:
				index = index + 1
				ious_total, dices_total, sens_total, spes_total, iou_single, dice_single, sen_single, spe_single, mious, cpa, mpa = getMetrixs_pro(outputs, labels)

				ious_totals.append(ious_total)
				dices_totals.append(dices_total)
				sens_totals.append(sens_total)
				spes_totals.append(spes_total)

				iou_singles = iou_singles + iou_single
				dice_singles = dice_singles + dice_single
				sen_singles = sen_singles + sen_single
				spe_singles = spe_singles + spe_single

				miouss.append(mious)
				mpas.append(mpa)
				cpass = cpass + cpa


				# s = nn.Softmax(dim = 1)
				# out = s(outputs)
				# out = torch.argmax(out, dim=1)

				# lab_s = labels.detach().cpu().squeeze().numpy()
				# out_s = out.detach().cpu().squeeze().numpy()

				# out_save = colored(out_s)
				# lab_save = colored(lab_s)

				# final_img = np.zeros((out_s.shape[0], out_s.shape[1]*2, 3), np.uint8)
				# final_img[0:out_s.shape[0], 0:out_s.shape[1]] = lab_save
				# final_img[0:out_s.shape[0], out_s.shape[1]:out_s.shape[1]*2] = out_save

				# name = './visualization/' + '_iter_' + str(iters) + '.png'
				# cv2.imwrite(name, final_img)

			labels = torch.squeeze(labels, dim=1).long()
			valid_loss += criterion(outputs, labels)

		iou = np.mean(ious_totals)
		dice = np.mean(dices_totals)
		sen = np.mean(sens_totals)
		spe = np.mean(spes_totals)

		iou_single = iou_singles / index
		dice_single = dice_singles / index
		sen_single = sen_singles / index
		spe_single = spe_singles / index

		miou = np.mean(miouss)
		mpa = np.mean(mpas)

		cpa = cpass / index

		valid_loss_mean = valid_loss / iterations
		print('Valid Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}\t IoUs_total: {:.6f}\t DICE_total: {:.6f}\t SEN_total: {:.6f}\t SPE_total: {:.6f} MIoU_total: {:.6f}'.format(epoch, valid_loss.item(), valid_loss_mean.item(), iou, dice, sen, spe, miou))

		print('iou_single: ', iou_single)
		print('dice_single: ', dice_single)
		print('sen_single: ', sen_single)
		print('spe_single: ', spe_single)

		print('cap: ', cpa)
		print('mpa: ', mpa)
		
		logger.info('Valid Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}\t IoUs_total: {:.6f}\t DICE_total: {:.6f}\t SEN_total: {:.6f}\t SPE_total: {:.6f} MIoU_total: {:.6f}'.format(epoch, valid_loss.item(), valid_loss_mean.item(), iou, dice, sen, spe, miou))

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(net.parameters(), lr = 0.0001)
valid_path = "../ds_ct_lesion_g2/test"
positive_path = "../ds_ct_lesion_g2/train/positive"
negative_path = "../ds_ct_lesion_g2/train/negative"
model_path = "./checkpoint"
log_path = "./log"

def caculate_FLOPs_and_Params():
	net = Unet(1, 1)
	input = torch.randn(1, 1, 672, 752)
	flops, params = profile(net, inputs=(input, ))
	print('flops: ', flops, ' params: ', params)
	return flops, params

def calFlop(path):
	net = myNet(use_dilation=True)
	checkpoint = torch.load(path, map_location='cpu' )
	net.load_state_dict(checkpoint['model'])
	stat(net, (3, 1408, 256))

def main(epochs = 101):
	img_Path = negative_path + "/img"
	l = os.listdir(img_Path)
	iterations = len(l)
	print('img_Path: ', img_Path, 'iterations: ', iterations)
	if os.path.exists(model_path):
		dir_list = os.listdir(model_path)
		if len(dir_list) > 0:
			dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
			print('dir_list: ', dir_list)
			last_model_name = model_path + '/' + dir_list[-1]

			checkpoint = torch.load(last_model_name)
			net.load_state_dict(checkpoint['model'])
			last_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			print('load epoch {} succeed! loss: {:.6f} '.format(last_epoch, loss))
		else:
			last_epoch = 0
			print('no saved model, start a new train.')

	else:
		last_epoch = 0
		print('no saved model, start a new train.')


	for epoch in range(last_epoch+1, epochs+1):
		ret, loss = train(net = net, epoch=epoch, iterations=iterations, loss_stop=0.01, positive_path=positive_path, negative_path=negative_path)
		state = {'model':net.state_dict(),'epoch':epoch, 'loss':loss}
		model_name = model_path + '/model_epoch_' + str(epoch) + '.pth'
		torch.save(state, model_name)
		if epoch%2 == 0:
			valid(net, epoch, valid_path)

		del_models(model_path)
		if ret:
			break
	print("train.....done.")


if __name__ == '__main__':  
	# calFlop('./checkpoint/model_epoch_3.pth')
	# caculate_FLOPs_and_Params()
	logger = get_logger(log_path)
	# #flops, params = caculate_FLOPs_and_Params()
	# #logger.info('----> flops: {:.6f}, params: {:.6f}'.format(flops, params))
	main()
	# checkpoint = torch.load('./checkpoint/model_epoch_15.pth')
	# net.load_state_dict(checkpoint['model'])
	# model = repvgg_model_convert(net)
	#calFlop('model_epoch_deploy.pth')
