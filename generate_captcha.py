import os.path
import random
import time

from util.common import *
from captcha.image import ImageCaptcha
import shutil

captcha_num=100000
captcha_dir,captcha_file=get_captcha_dir(captcha_type[0])
# captcha_num=1000
# captcha_dir,captcha_file=get_captcha_dir(captcha_type[1])
# captcha_num=500
# captcha_dir,captcha_file=get_captcha_dir(captcha_type[3])


shutil.rmtree(captcha_dir,ignore_errors=True)
if os.path.exists(captcha_file):
	os.remove(captcha_file)
os.makedirs(captcha_dir,exist_ok=True)


if __name__=='__main__':
	# print(captcha_dir)
	img_list=[]
	image=ImageCaptcha()
	for i in range(captcha_num):
		img_val=''.join(random.sample(captcha_array,4))
		img_name='{}_{}.png'.format(img_val, int(time.time()))
		image.write(img_val,captcha_dir+'/'+img_name)
		img_list.append(img_name)
	with open(captcha_file,'w',encoding='utf-8') as f:
		for img in img_list:
			f.write(img+'\n')