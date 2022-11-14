
from torch.utils.data import DataLoader

from models import captcha
from util.my_dataset import *
from util.one_hot import *

transforms = transforms.Compose([
			transforms.Grayscale(),
			transforms.ToTensor()
		])

def getImgTensor(img_path):
	img = Image.open(img_path)
	img_tensor = transforms(img).float()
	img_tensor = img_tensor.to(device)
	return img_tensor

# 加载之前训练好的权重参数
checkpoint = torch.load(filename)
model = captcha.MyModel()
model.to(device)
model.eval()  # 有这个会导致正确率为0
model.load_state_dict(checkpoint['state_dict'])

def getResult(img_tensor):
	# img_tensor= img_tensor.reshape([1, 1,60,160])
	img_tensor = img_tensor.unsqueeze(0) #升维
	# print(img_tensor.shape)
	target= model(img_tensor)
	# print(target.shape)
	prediction = target.view(-1, captcha_array.__len__())
	prediction_label = vectotext(prediction)
	return prediction_label

if __name__=='__main__':
	file_path=DIR_DATA+r'/predict/0a1x_1667082356.png'
	print(getResult(getImgTensor(file_path)))