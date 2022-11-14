import copy
import time

from torch import nn
from torch.utils.data import DataLoader

from models import captcha
from util.my_dataset import *
from util.one_hot import *
from torch import optim

if __name__=='__main__':
	train_data = MyDataSet(captcha_type[0])
	valid_data = MyDataSet(captcha_type[1])

	# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
	# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)

	# valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
	# valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True,  pin_memory=True)
	valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)

	model = captcha.MyModel()
	model.to(device)
	model.train()  # 启用 batch normalization 和 dropout

	loss_fn = nn.MultiLabelSoftMarginLoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
	scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=8,gamma=0.1)#学习率每8个epoch衰减为原来的1/10

	for epoch in range(num_epochs):
		# 当前epoch训练结果保存下来
		train_rights = []
		# 记录最好的一次
		best_acc = 0

		# tqdm会使第一次迭代特别慢,这个也会记录时间。暂时比较丑陋，将来从其他好的项目借鉴下怎么写比较好
		for batch_idx, (img, target) in enumerate(tqdm.tqdm( train_dataloader,mininterval=10,ncols=20)):
		# for batch_idx, (img, target) in enumerate(train_dataloader):

			img=img.cuda()
			target=target.cuda()

			output = model(img)
			loss = loss_fn(output, target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			right = accuracy(output, target)
			train_rights.append(right)

			if batch_idx % 100 == 0:
				val_rights = []
				torch.set_grad_enabled(mode=False)
				for (data, target) in valid_dataloader:
					output = model(data.cuda())
					target = target.cuda()
					right = accuracy(output, target)
					val_rights.append(right)

				torch.set_grad_enabled(mode=True)
				# 准确率计算
				train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
				val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
				epoch_acc = val_r[0] / val_r[1]
				if epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())
					state = {
						'num_epochs':num_epochs,
						'batch_size':batch_size,
						'best_acc': best_acc,
						'state_dict': model.state_dict(),  # key就是各层的名字，值就是训练好的权重
					}


				printTime('')
				print('当前epoch:{} [{}/{}({:.2f}%)] \t损失:{}\t训练集准确率:{}\t测试集正确率:{}'.format(
					epoch, batch_idx * batch_size, len(train_dataloader.dataset),
						   100 * batch_idx * batch_size / len(train_dataloader.dataset),
					loss.data,
						   100 * train_r[0] / train_r[1],
						   100 * val_r[0] / val_r[1]
				))




		print('optimizer learning rate: {:.7f}'.format(optimizer.param_groups[0]['lr']))
		scheduler.step()#记录步骤数

	torch.save(state, filename)


# num_epochs = 10 batch_size = 64 218588，85m9s 96.3 batch_size = 64
# num_epochs = 10 batch_size = 128 218588，56m34s 95.3   gpu使用率13%  最后电脑卡住了
# num_epochs = 30 batch_size = 128 218588，182m 8s 95.6  学习率衰减 gpu使用率13%
# num_epochs = 2 batch_size = 128 218588，182m 8s 83.6  学习率衰减 gpu使用率13%

#模型修改之后的训练
# num_epochs = 10 batch_size = 128 218588，56m34s 100

#2022 10:31
#大概8小时，60%正确率
