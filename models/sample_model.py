import torch.nn as nn

# Model described in Table 7 of:
#   "A SAR dataset for ATR development: the Synthetic and Measured Paired Labeled Experiment (SAMPLE)"

# Input size is 1x64x64
class sample_model(nn.Module):
	def __init__(self,input_channels=1,num_classes=10, drop_prob=0.):
		super(sample_model,self).__init__()
		self.features = nn.Sequential(
			# Size = 1x64x64
			nn.Conv2d(input_channels,16,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=False),
			nn.MaxPool2d(kernel_size=2,stride=2),
			# Size = 16x32x32
			nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=False),
			nn.MaxPool2d(kernel_size=2,stride=2),
			# Size = 32x16x16
			nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=False),
			nn.MaxPool2d(kernel_size=2,stride=2),
			# Size = 64x8x8
			nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
			nn.ReLU(inplace=False),
			nn.MaxPool2d(kernel_size=2,stride=2),
			# Size = 128x4x4
		)
		self.classifier = nn.Sequential(
			# Size = 128*4*4 = 2048
			nn.Dropout(p=drop_prob, inplace=True),
			nn.Linear(128*4*4, 1000),
			nn.ReLU(inplace=False),
			nn.Dropout(p=drop_prob, inplace=True),
			nn.Linear(1000, 500),
			nn.ReLU(inplace=False),
			nn.Dropout(p=drop_prob, inplace=True),
			nn.Linear(500, 250),
			nn.ReLU(inplace=False),
			nn.Dropout(p=drop_prob, inplace=True),
			nn.Linear(250, num_classes),
		)
	def forward(self,x):
		x = self.features(x)
		x = x.view(x.size(0),-1)
		x = self.classifier(x)
		return x
			

