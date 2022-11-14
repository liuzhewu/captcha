import torch
import os
from util.one_hot import *
import time
import tqdm

captcha_array = list("0123456789abcdefghijklmnopqrstuvwxyz")
captcha_size = 4

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print(device)

# 先证明确实可以提高正确率，可以优化，再来增大这个次数
num_epochs = 100
batch_size = 512
# batch_size = 256
# batch_size = 128
# batch_size = 64
# batch_size = 2
#配置为cpu实际核心数，不是逻辑核心数
num_workers=4

DIR_DATA = os.path.dirname(os.path.abspath(__file__)) + r'\..\..\captcha_data'
# 模型保存，名字自己期
filename = 'best.pt'



captcha_type=['train','valid','test','predict']
def get_captcha_dir(captcha_type):
	dir=os.path.dirname(__file__)+'/../../captcha_data/'
	return dir+captcha_type,dir+'/'+captcha_type+'.txt'

def accuracy(predictions, labels):

	correct = 0
	length = labels.shape[0]
	for idx in range(length):
		prediction = predictions[idx]
		prediction = prediction.view(-1, captcha_array.__len__())
		prediction_label = vectotext(prediction)
		label = labels[idx].view(-1, captcha_array.__len__())

		true_label = vectotext(label)
		# print("真实值:{},预测值:{}".format(true_label, prediction_label))

		if prediction_label == true_label:
			# print("真实值:{},预测值:{}".format(true_label, prediction_label))
			correct += 1

	return correct, length

since = time.time()

def printTime(tag):
	time_elapsed = time.time() - since
	print(tag,',Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	pass