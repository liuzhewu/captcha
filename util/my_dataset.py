import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import util.one_hot as one_hot
from util.common import *


class MyDataSet(Dataset):

	def __init__(self, captcha_type):
		super(MyDataSet, self).__init__()
		dir,file= get_captcha_dir(captcha_type)
		with open(file,'r',encoding='utf-8') as f:
			self.list_img_path=[dir+'/'+x.strip() for x in f.readlines()]

		self.labels=[one_hot.text2vec( x.split('/')[-1].split('_')[0]).view(1, -1)[0] for x in self.list_img_path]
		self.transforms = transforms.Compose([
			transforms.Grayscale(),
			transforms.ToTensor()
		])

	def __getitem__(self, index):
		img_path = self.list_img_path[index]
		img = Image.open(img_path)

		img_tensor = self.transforms(img).float()
		# img_tensor = img_tensor.to(device)

		# label = img_path.split('/')[-1].split('_')[0]
		# label = one_hot.text2vec(label)
		# label = label.view(1, -1)[0]
		# label = torch.FloatTensor(label).to(device)

		return img_tensor, self.labels[index]

	def __len__(self):
		return self.list_img_path.__len__()
