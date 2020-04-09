import os, sys
import datetime
import numpy as np
import random
import timeit
import argparse
import getpass

sys.path.append('..')
from defacing.training.training import trainer
from defacing.helpers.utils import get_available_gpus
from distutils.dir_util import copy_tree
import tensorflow as tf

list_gpu = get_available_gpus()
n_gpu = len(list_gpu)
print("Available GPUs: ", list_gpu)

parser = argparse.ArgumentParser(description='Training DefacingNet')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use')
parser.add_argument('-jn', '--job_name', required=True, type=str, help="The job name is required. All the training will be saved here.")

args = parser.parse_args()

t0 = timeit.default_timer()
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
print ("GPU Availability: ", tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))


Kfolds = -1
nfolds = list(range(0, Kfolds + 1))

for fold in nfolds:
	root_dir = './Logs/' + args.job_name + '/train_test_fold_{}'.format(fold)
	dir_path = './Logs/' + args.job_name + '/train_test_fold_{}/csv/'.format(fold)

	# currently a very hacky way of doing this -- will need to fix later
	from_dir = os.path.abspath('./csv/faced_defaced/train_test_fold_{}/csv/'.format(fold))
	to_dir = dir_path
	copy_tree(from_dir, to_dir)

	train_csv_path = os.path.join(dir_path, 'training.csv')
	valid_csv_path = os.path.join(dir_path, 'validation.csv')

	# Model Path
	model_path = root_dir + '/' + args.job_name


	# create a path to where the model will be saved
	if not os.path.exists(root_dir):
	    os.makedirs(root_dir)

	if not os.path.exists(dir_path):
	    os.makedirs(dir_path)

	if not os.path.exists(model_path):
	    os.makedirs(model_path)

	# basic job info text file to identify jobs
	basic_job_info = os.path.join(os.path.abspath(root_dir), 'job_info.txt')
	with open(basic_job_info, 'w') as f:
	    f.write("Jobname: %s\n" % args.job_name)
	    f.write("Created on: %s\n" % str(datetime.datetime.now()))
	    f.write("Created by: %s\n" % str(getpass.getuser()))
	    f.write("Model store path: %s\n" % os.path.abspath(model_path))
	    f.write("GPU Availability: %s\n" % str(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)))
	    f.write("Available GPUs: %s\n" % (','.join(list_gpu)))


	train = trainer(train_csv_path,
			valid_csv_path,
			basic_job_info,
			model_path,
			image_size = 32,
			batch_size = 32,
			initial_epoch = 0,
			nepochs = 15,
			dropout = 0.2,
			nclasses = 2,
			nchannels = 1,
			gpus = n_gpu)
	train.train()

	elapsed = timeit.default_timer() - t0
	print('Time: {:.3f} min'.format(elapsed / 60))
	del train

############################################ final model training #################################################
print ("Training final model")
root_dir = './Logs/' + args.job_name + '/train_test'
dir_path = './Logs/' + args.job_name + '/train_test/csv/'

# currently a very hacky way of doing this -- will need to fix later
from_dir = os.path.abspath('./csv/faced_defaced')

train_csv_path = os.path.join(from_dir, 'all.csv')
valid_csv_path = os.path.join(from_dir, 'all.csv')

# Model Path
model_path = root_dir + '/' + args.job_name


# create a path to where the model will be saved
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

# basic job info text file to identify jobs
basic_job_info = os.path.join(os.path.abspath(root_dir), 'job_info.txt')
with open(basic_job_info, 'w') as f:
    f.write("Jobname: %s\n" % args.job_name)
    f.write("Created on: %s\n" % str(datetime.datetime.now()))
    f.write("Created by: %s\n" % str(getpass.getuser()))
    f.write("Model store path: %s\n" % os.path.abspath(model_path))
    f.write("GPU Availability: %s\n" % str(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)))
    f.write("Available GPUs: %s\n" % (','.join(list_gpu)))


train = trainer(train_csv_path,
		valid_csv_path,
		basic_job_info,
		model_path,
		image_size = 32,
		batch_size = 32,
		initial_epoch = 0,
		nepochs = 15,
		dropout = 0.2,
		nclasses = 2,
		nchannels = 1,
		gpus = n_gpu)
train.train()

elapsed = timeit.default_timer() - t0
print('Time: {:.3f} min'.format(elapsed / 60))
del train
