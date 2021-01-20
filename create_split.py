# NAI

# This file creates a train/test split of the SAMPLE dataset according to the notation
#   in the SAMPLE paper

import glob
import os
import random

def create_mixed_dataset_exp41(root, k):

	CLASSES = ["2s1", "bmp2", "btr70", "m1", "m2", "m35", "m548", "m60", "t72", "zsu23"]

	# Lists of tuples that make up the datasets ("/pth/to/f.png", class#)
	dataset_list_train = []
	dataset_list_test  = []

	# Create the splits for each of the classes individually
	for cls in CLASSES:
		all_measured = glob.glob("{}/{}/{}/*.png".format(root,"real",cls))
		Nmj = len(all_measured)
		test_data = []; train_data = []
		for fname in all_measured:
			if "elevDeg_017" in fname:
				test_data.append([fname,CLASSES.index(cls)])
			else:
				train_data.append([fname,CLASSES.index(cls)])
		Smj = len(test_data)
		Tmj = round(k*(Nmj - Smj))  # How many "real" samples to use for this class
		Tsj = (Nmj - Smj) - Tmj     # How many "synth" samples to use for this class
		assert((Nmj-Smj)==len(train_data))
		# For each measured sample we dont use replace it with its synthetic version
		synth_inds = random.sample( list(range(len(train_data))), Tsj)
		for ind in synth_inds:
			train_data[ind][0] = train_data[ind][0].replace("/real/","/synth/")
			train_data[ind][0] = train_data[ind][0].replace("_real_","_synth_")
			assert(os.path.isfile(train_data[ind][0]))
		dataset_list_train.extend(train_data)
		dataset_list_test.extend(test_data)
		print("Class: {}\tNmj: {} \tSmj: {}\tTmj: {}\tTsj: {}".format(cls,Nmj,Smj,Tmj,Tsj))
	print("len( train ): ",len(dataset_list_train))
	print("len( test ):  ",len(dataset_list_test))

	return dataset_list_train, dataset_list_test



