import torch.nn as nn

# Heiligers and Huizing. On the Importance of Visual Explanation and Segmentation for SAR ATR using Deep Learning

# Input size is 1x128x128
class heiligers_model(nn.Module):
	def __init__(self,num_classes=10):
		super(heiligers_model,self).__init__()
		self.features = nn.Sequential(
			# Size = Input = 1x128x128
			nn.Conv2d(1,18,kernel_size=5,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 18x124x124
			nn.Conv2d(18,18,kernel_size=5,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 18x120x120
			nn.MaxPool2d(kernel_size=6,stride=6),
			# Size = 18x20x20
			nn.Conv2d(18,36,kernel_size=5,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 36x16x16
			nn.MaxPool2d(kernel_size=4,stride=4),
			# Size = 36x4x4
			nn.Conv2d(36,120,kernel_size=4,stride=1,padding=0),
			nn.ReLU(inplace=True),
			# Size = 120x1x1
		)
		self.classifier = nn.Sequential(
			# Size=120x1x1
			nn.Linear(120*1*1, num_classes),
			#nn.ReLU(inplace=True),
			# Size = num_classes
		)
	def forward(self,x):
		# Forward pass through conv layers
		x = self.features(x)
		# Flatten features to vector
		x = x.view(x.size(0),120*1*1)
		# Forward pass through FC layers
		x = self.classifier(x)
		return x
			

