import torch.nn as nn

# S. Chen et al. "Target Classification Using the Deep Convolutional Networks for SAR Images"

# Input size is 1x128x128
class a_convnet(nn.Module):
	def __init__(self,input_channels=1,num_classes=10):
		super(a_convnet,self).__init__()
		self.classes = num_classes
		self.features = nn.Sequential(
			# Size = Input = 1x128x128
			nn.Conv2d(input_channels,16,kernel_size=5,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 16x124x124
			nn.MaxPool2d(kernel_size=2,stride=2),
			# Size = 16x62x62
			nn.Conv2d(16,32,kernel_size=5,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 32x58x58
			nn.MaxPool2d(kernel_size=2,stride=2),
			# Size = 32x29x29
			nn.Conv2d(32,64,kernel_size=6,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 64x24x24
			nn.MaxPool2d(kernel_size=2,stride=2),
			# Size = 64x12x12
			nn.Conv2d(64,128,kernel_size=5,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 128x8x8
			nn.Dropout2d(p=0.5),
			nn.Conv2d(128,self.classes,kernel_size=8,stride=1,padding=0), # [NAI] Ksize 3->8 for 128x128
			# Size = 10x1x1
		)


	def forward(self,x):
		# Forward pass through conv layers
		x = self.features(x)
		# Reshape for use with softmax
		x = x.view(x.size(0),self.classes)
		return x
			

