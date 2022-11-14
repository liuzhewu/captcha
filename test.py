from torch.utils.data import DataLoader

from models import captcha
from util.my_dataset import *
from util.one_hot import *

# test_dir = DIR_DATA + r'\valid'
# test_dir = DIR_PROJECT + r'\train'
test_data = MyDataSet(captcha_type[2])
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 加载之前训练好的权重参数
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model = captcha.MyModel()
model.to(device)
model.eval()  # 有这个会导致正确率为0
model.load_state_dict(checkpoint['state_dict'])

test_rights = []

for img, target in test_dataloader:
	img = img.cuda()
	target = target.cuda()
	output = model(img)
	right = accuracy(output, target)
	test_rights.append(right)

val_r = (sum([tup[0] for tup in test_rights]), sum([tup[1] for tup in test_rights]))
print(' [{}] \t测试集正确率:{}'.format(
	len(test_dataloader.dataset),
	100 * val_r[0] / val_r[1]
))
