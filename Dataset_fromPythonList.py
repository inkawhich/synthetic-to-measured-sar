# NAI

# Custom PyTorch dataset that creates a PyTorch dataset from a python list
#  input list = [ ["/pth/to/file.png", class#] , ... ]

import torch
from PIL import Image


class Dataset_fromPythonList(torch.utils.data.Dataset):
	def __init__(self, dset_list, transform=None):
		self.dset_list = dset_list
		self.transform = transform
	def __len__(self):
		return len(self.dset_list)
	def __getitem__(self,idx):
		path, target = self.dset_list[idx]
		with open(path, 'rb') as f:
			img = Image.open(f)
			img = img.convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		return img, target, path



