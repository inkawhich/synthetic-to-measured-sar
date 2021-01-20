# NAI

import numpy as np
import random
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients


############################################################################################################
#### Random boilerplate functions

def save_checkpoint(state, is_best, checkpoint_prefix):
	filepath = checkpoint_prefix+"_checkpoint.pth.tar"
	torch.save(state, filepath)
	if is_best:
		print("New best file! Saving.")
		shutil.copyfile(filepath, checkpoint_prefix+'_checkpoint_best.pth.tar')

def create_learning_rate_table(initial_lr, decay_schedule, gamma, epochs):
	lr_table = np.zeros((epochs))
	prev_lr = initial_lr
	for i in range(epochs):
		if i in decay_schedule:
			prev_lr *= gamma
		lr_table[i] = prev_lr
	return lr_table

def adjust_learning_rate(optimizer, curr_epoch, learning_rate_table):
	for param_group in optimizer.param_groups:
		param_group['lr'] = learning_rate_table[curr_epoch]


############################################################################################################
#### Perturb the input image with Uniform noise

def UniformNoise_attack(device, dat, eps):
	x_adv = dat + torch.FloatTensor(dat.shape,device=device).uniform_(-eps, eps).to(device)
	x_adv = torch.clamp(x_adv,0.,1.) # respect image bounds
	return x_adv

############################################################################################################
#### Perturb the input image with Uniform noise

def SimClutter_attack(device, dat, p):
	# Generate a mask that covers p% of pixels. These pixels will be perturbed
	noise = torch.rand_like(dat);
	noise_mask = torch.zeros_like(dat).to(device)
	noise_mask[noise<=(p/100.)] = 1.
	# Generate the noise that will be applied at the masked pixel locations
	final_noise = torch.rand_like(dat);
	# Create the data mask for the pixel locations that will not be perturbed
	data_mask = torch.zeros_like(dat).to(device)
	data_mask[noise>(p/100.)] = 1.
	# Create the perturbed image
	x_adv = dat*data_mask + final_noise*noise_mask
	x_adv = torch.clamp(x_adv,0.,1.) # respect image bounds
	return x_adv

############################################################################################################
#### Perturb the input image with Gaussian noise

def GaussianNoise_attack(device, dat, eps):
	x_nat = dat.clone()
	# Generate noise from standard normal
	noise = torch.FloatTensor(dat.shape,device=device).normal_(0., 1.).to(device)
	# Scale values to range [-eps,eps]
	scaled_noise = (2*eps)*(noise-noise.min())/(noise.max()-noise.min()) - eps
	#print(scaled_noise.max())
	#print(scaled_noise.min())
	# Apply the noise
	x_adv = x_nat + scaled_noise	
	x_adv = torch.clamp(x_adv,0.,1.) # respect image bounds
	#diff = torch.abs(x_adv-dat)
	#print(diff.max())	
	#print(diff.min())	
	return x_adv


############################################################################################################
#### Test the input model on data from the loader. Used in training script

def test_model(net,device,loader,mean,std):
	net.eval()
	# Stat keepers
	running_clean_correct = 0.
	running_clean_loss = 0.
	running_total = 0.
	with torch.no_grad():
		for batch_idx,(data,labels,_) in enumerate(loader):
			data = data.to(device); labels = labels.to(device)
			clean_outputs = net((data-mean)/std)
			clean_loss = F.cross_entropy(clean_outputs, labels)
			_,clean_preds = clean_outputs.max(1)
			running_clean_correct += clean_preds.eq(labels).sum().item()
			running_clean_loss += clean_loss.item()
			running_total += labels.size(0)
		clean_acc = running_clean_correct/running_total
		clean_loss = running_clean_loss/running_total
	net.train()
	return clean_acc,clean_loss


############################################################################################################
#### ADVERSARIAL TRAINING STUFF

def gradient_wrt_data(model,device,data,lbl):
	# Manually Normalize
	mean = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
	std = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
	dat = (data-mean)/std
	# Forward pass through the model
	dat.requires_grad = True
	out = model(dat)
	# Calculate loss
	loss = F.cross_entropy(out,lbl)
	# zero all old gradients in the model
	model.zero_grad()
	# Back prop the loss to calculate gradients
	loss.backward()
	# Extract gradient of loss w.r.t data
	data_grad = dat.grad.data
	# Unnorm gradients back into [0,1] space
	grad = data_grad / std
	return grad.data.detach()

def PGD_Linf_attack(model, device, dat, lbl, eps, alpha, iters):
	x_nat = dat.clone().detach()
	# Randomly perturb within small eps-norm ball
	x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
	x_adv = torch.clamp(x_adv,0.,1.) # respect image bounds
	# Iteratively Perturb data	
	for i in range(iters):
		zero_gradients(x_adv)
		model.zero_grad()		
		# Calculate gradient w.r.t. data
		grad = gradient_wrt_data(model,device,x_adv.clone().detach(),lbl)
		# Perturb by the small amount a
		x_adv = x_adv + alpha*grad.sign()
		# Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
		x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
		# Make sure we are still in bounds
		x_adv = torch.clamp(x_adv, 0., 1.)
	return x_adv.data.clone().detach()

############################################################################################################
#### LABEL SMOOTHING STUFF
# Transform the true "Long" labels to softlabels. The confidence of the gt class is 
#  1-smoothing, and the rest of the probability (i.e. smoothing) is uniformly distributed
#  across the non-gt classes. Note, this is slightly different than standard smoothing
#  notation.  

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
	"""
	if smoothing == 0, it's one-hot method
	if 0 < smoothing < 1, it's smooth method
	"""
	assert 0 <= smoothing < 1
	confidence = 1.0 - smoothing
	label_shape = torch.Size((true_labels.size(0), classes))
	with torch.no_grad():
		true_dist = torch.empty(size=label_shape, device=true_labels.device)
		true_dist.fill_(smoothing / (classes - 1))
		true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
	return true_dist.float()

def xent_with_soft_targets(logit_preds, targets):
	logsmax = F.log_softmax(logit_preds, dim=1)
	batch_loss = targets * logsmax
	batch_loss =  -1*batch_loss.sum(dim=1)
	return batch_loss.mean()

############################################################################################################
#### MIXUP STUFF -- https://github.com/facebookresearch/mixup-cifar10

def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1
	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)
	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

