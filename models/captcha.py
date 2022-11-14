from torch import nn

from util.common import *

# bn和dropout一般不同时使用，bn放在cnn中，dropout放在nn中，bn在conv后，relu之前。dropout在relu之后
class MyModel(nn.Module):
	def __init__(self):
		# x #[64, 1, 60, 160]
		super(MyModel, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)  # [64, 32, 30, 80]
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)  # [64, 64, 15, 40]
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)  # [64, 64, 7, 20]
		)
		self.fc = nn.Sequential(
			nn.Linear((IMAGE_WIDTH // 8) * (IMAGE_HEIGHT // 8) * 64, 1024),  # 这里//8是因为前面3层每次都做了2*2的池化，每次长宽都变为原来的一半。
			nn.ReLU(),
			nn.Dropout(0.5),
		)
		self.rfc = nn.Sequential(
			nn.Linear(1024, out_features=captcha_size * captcha_array.__len__())
		)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		x = self.rfc(x)

		return x;
