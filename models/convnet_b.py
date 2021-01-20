import torch.nn as nn

# S. Chen et al. "Target Classification Using the Deep Convolutional Networks for SAR Images"
# From table 10, this is the "B" network

# Input size is 1x128x128
class convnet_b(nn.Module):
	def __init__(self,input_channels=1,num_classes=10):
		super(convnet_b,self).__init__()
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
			nn.Conv2d(32,64,kernel_size=5,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 64x25x25
			nn.MaxPool2d(kernel_size=2,stride=2),
			# Size = 64x12x12
		)
		self.classifier = nn.Sequential(
			# Size = 64x12x12
			nn.Dropout(p=0.5),
			nn.Linear(64*12*12, 1024),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(1024, num_classes),
			# Size = num_classes
		)
	def forward(self,x):
		# Forward pass through conv layers
		x = self.features(x)
		# Flatten features to vector
		x = x.view(x.size(0),64*12*12)
		# Forward pass through FC layers
		x = self.classifier(x)
		return x
			

