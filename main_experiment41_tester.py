# NAI

# Main training file for SAMPLE Experiment 4.1, where we vary the percentage of synthetic vs measured
#	data in the training set

from __future__ import print_function
import numpy as np
import sys
import os
import random
import torch
import torch.nn as nn
import torch.utils.data as utilsdata
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats as st
from torch.autograd import Variable

# Custom
import models
import create_split
import Dataset_fromPythonList as custom_dset
import helpers

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#################################################################################################################
# Inputs
#################################################################################################################

# Experiment 4.1 specifics 
K = float(sys.argv[1])
dataset_root = "./SAMPLE_Public_Dist_A/png_images/qpm"
REPEAT_ITERS = 100
DSIZE = 64

# Learning Params
num_epochs = 60
batch_size = 128
learning_rate_decay_schedule = [61]
learning_rate = 0.001
gamma = 0.1
weight_decay = 0.
dropout = 0.4
gaussian_std = 0.4
uniform_range = 0.
simClutter = 0.
flipProb = 0.
degrees = 0
LBLSMOOTHING_PARAM = 0.1
MIXUP_ALPHA = 0.1

#AT_EPS = 2./255.; AT_ALPHA = 0.5/255.; AT_ITERS = 7
#AT_EPS = 4./255.; AT_ALPHA = 1./255. ; AT_ITERS = 7
#AT_EPS = 8./255.; AT_ALPHA = 2./255. ; AT_ITERS = 7

# Normalization Constants for range [-1,+1]
MEAN = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
STD  = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)

ACCUMULATED_ACCURACIES = []

# Loop over training runs to get average accuracies
for ITER in range(REPEAT_ITERS):

	print("**********************************************************")
	print("Starting Iter: {} / {} for K = {}".format(ITER,REPEAT_ITERS,K))
	print("**********************************************************")

	#################################################################################################################
	# Load Model
	#################################################################################################################
	net = None
	#net = models.sample_model(num_classes=10, drop_prob=dropout).to(device)
	net = models.resnet18(num_classes=10, drop_prob=dropout).to(device);
	#net = models.wide_resnet18(num_classes=10, drop_prob=dropout).to(device);
	net.train()
	print(net)

	# Optimizer
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

	# Build the checkpoint prefix for this run
	#checkpoint_prefix = "tmp-checkpoints/{}_{}_seed{}_{}".format(model_name,dataset_name,seed,perturbation_method)

	# Define the learning rate schedule
	learning_rate_table = helpers.create_learning_rate_table(learning_rate,learning_rate_decay_schedule,gamma,num_epochs)


	#################################################################################################################
	# Create datasets
	#################################################################################################################

	transform_train = transforms.Compose([
		transforms.Grayscale(),
		transforms.CenterCrop(DSIZE),
		transforms.RandomRotation(degrees),
		transforms.RandomHorizontalFlip(flipProb),
		transforms.ToTensor(),
	]) 
	transform_test = transforms.Compose([
		transforms.Grayscale(),
		transforms.CenterCrop(DSIZE),
		transforms.ToTensor(),
	])	
	
	# Create the measured/synthetic split training and test data
	full_train_list,test_list = create_split.create_mixed_dataset_exp41(dataset_root, K)

	# Create validation set split
	val_set_size = 0 #int(0.15 * len(full_train_list))
	val_sample_inds = random.sample(list(range(len(full_train_list))), val_set_size)
	train_list = []; val_list = []
	for ind in range(len(full_train_list)):
		if ind in val_sample_inds:
			val_list.append(full_train_list[ind])
		else:
			train_list.append(full_train_list[ind])
	print("# Train: ",len(train_list))
	print("# Val:	",len(val_list))
	print("# Test:	",len(test_list))

	# Construct datasets and dataloaders
	trainset = custom_dset.Dataset_fromPythonList(train_list, transform=transform_train)
	trainloader = utilsdata.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2,timeout=1000)
	#valset = custom_dset.Dataset_fromPythonList(val_list, transform=transform_test)
	#valloader = utilsdata.DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=2,timeout=1000)
	testset = custom_dset.Dataset_fromPythonList(test_list, transform=transform_test)
	testloader = utilsdata.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2,timeout=1000)


	#################################################################################################################
	# Training Loop
	#################################################################################################################

	global_training_iteration = 0.
	best_test_acc = 0.
	final_test_acc = 0.
	final_train_acc = 0.

	for epoch in range(num_epochs):

		# Decay learning rate according to decay schedule 
		helpers.adjust_learning_rate(optimizer, epoch, learning_rate_table)
		net.train()
		print("Starting Epoch {}/{}. lr = {}".format(epoch,num_epochs,learning_rate_table[epoch]))
	
		running_correct = 0.
		running_total = 0.
		running_loss_sum = 0.
		running_real_cnt = 0.

		for batch_idx,(data,labels,pth) in enumerate(trainloader):
			data = data.to(device); labels = labels.to(device)

			# MIXUP
			#mixed_data, targets_a, targets_b, lam = helpers.mixup_data(data, labels, MIXUP_ALPHA, use_cuda=True)
			#mixed_data, targets_a, targets_b      = map(Variable, (mixed_data, targets_a, targets_b))

			# Gaussian/Uniform/SimClutter Noise
			if(uniform_range != 0):
				noise = (torch.rand_like(data)-.5)*2*uniform_range;
				data += noise;
				data = torch.clamp(data, 0, 1);
			if(gaussian_std != 0):
				data += torch.randn_like(data)*gaussian_std;
				data = torch.clamp(data, 0, 1);
				#mixed_data += torch.randn_like(mixed_data)*gaussian_std;
				#mixed_data = torch.clamp(mixed_data, 0, 1);
			if(simClutter != 0):
				data = helpers.SimClutter_attack(device, data, simClutter);

			# ADVERSARIALLY PERTURB DATA
			#data = helpers.PGD_Linf_attack(net, device, data.clone().detach(), labels, eps=AT_EPS, alpha=AT_ALPHA, iters=AT_ITERS)

			### Optional: Plot some training samples	
			#plt.figure(figsize=(10,3))
			#plt.subplot(1,6,1);plt.imshow(data[0].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[0].split("/")[-1].split("_")[:2])
			#plt.subplot(1,6,2);plt.imshow(data[1].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[1].split("/")[-1].split("_")[:2])
			#plt.subplot(1,6,3);plt.imshow(data[2].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[2].split("/")[-1].split("_")[:2])
			#plt.subplot(1,6,4);plt.imshow(unshifted[0].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[0].split("/")[-1].split("_")[:2])
			#plt.subplot(1,6,5);plt.imshow(unshifted[1].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[1].split("/")[-1].split("_")[:2])
			#plt.subplot(1,6,6);plt.imshow(unshifted[2].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[2].split("/")[-1].split("_")[:2])
			#plt.show()
			#exit()

			# MIXUP
			#outputs = net((mixed_data-MEAN)/STD)
			#loss = helpers.mixup_criterion(nn.CrossEntropyLoss(), outputs, targets_a, targets_b, lam)	

			# Forward pass data through model. Normalize before forward pass
			outputs = net((data-MEAN)/STD)

			# VANILLA CROSS-ENTROPY
			loss = F.cross_entropy(outputs, labels);
			
			# LABEL SMOOTHING LOSS
			#sl = helpers.smooth_one_hot(labels,10,smoothing=LBLSMOOTHING_PARAM)
			#loss =  helpers.xent_with_soft_targets(outputs, sl)

			# COSINE LOSS
			#one_hots = smooth_one_hot(labels,10,smoothing=0.)
			#loss = (1. - (one_hots * F.normalize(outputs,p=2,dim=1)).sum(1)).mean()

			# Calculate gradient and update parameters
			optimizer.zero_grad()
			net.zero_grad()
			loss.backward()
			#torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=10., norm_type=2)	# For cosine loss
			optimizer.step()

			# Measure accuracy and loss for this batch
			_,preds = outputs.max(1)
			running_total += labels.size(0)
			running_correct += preds.eq(labels).sum().item()
			#running_correct += (lam * preds.eq(targets_a.data).cpu().sum().float() + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) # For Mixup
			running_loss_sum += loss.item()		

			# Global training iteration count across epochs
			global_training_iteration += 1

			# Compute measured/synthetic split for the batch
			for tp in pth:
				if "/real/" in tp:
					running_real_cnt += 1.

		### End of epoch - print stats
		print("[{}] Epoch [ {} / {} ]; lr: {} TrainAccuracy: {} TrainLoss: {} %-Real: {}".format(ITER,epoch,num_epochs,learning_rate_table[epoch],running_correct/running_total,running_loss_sum/running_total,running_real_cnt/running_total))
		#val_acc,val_loss = helpers.test_model(net,device,valloader,MEAN,STD)
		#print("\t[{}] Epoch [ {} / {} ]; ValAccuracy: {} ValLoss: {}".format(ITER,epoch,num_epochs,val_acc,val_loss))

		test_acc,test_loss = helpers.test_model(net,device,testloader,MEAN,STD)

		print("\t[{}] Epoch [ {} / {} ]; TestAccuracy: {} TestLoss: {}".format(ITER,epoch,num_epochs,test_acc,test_loss))
		if test_acc > best_test_acc:
			print("\tNew best test accuracy!")
			best_test_acc = test_acc
		final_test_acc = test_acc
		final_train_acc = running_correct/running_total

	if final_train_acc > .5:
		print("BREAK. FINAL RECORDED TEST ACC = ",final_test_acc)
		ACCUMULATED_ACCURACIES.append(final_test_acc)
	else:
		print("MODEL NEVER LEARNED ANYTHING. NOT RECORDING")


	## Optional: Save a model checkpoint here
	#if final_test_acc > SAVE_THRESH:
	#	helpers.save_checkpoint({'test_acc': final_test_acc,'state_dict': net.state_dict()}, False, SAVE_CKPT)	
	#	exit("Found above average model to save. Exit now!")
	#helpers.save_checkpoint({'test_acc': final_test_acc,'state_dict': net.state_dict()}, False, "{}_K{}_ITER{}".format(SAVE_CKPT,int(100*K),ITER))	

print("\n\nEND OF TRAINING!")
print("ACCUMULATED ACCURACIES: ",ACCUMULATED_ACCURACIES)
print("\tMin = ",np.array(ACCUMULATED_ACCURACIES).min())
print("\tMax = ",np.array(ACCUMULATED_ACCURACIES).max())
print("\tAvg = ",np.array(ACCUMULATED_ACCURACIES).mean())
print("\tStd = ",np.array(ACCUMULATED_ACCURACIES).std())
print("\tlen = ",len(ACCUMULATED_ACCURACIES))

exit("Done!")






