import matplotlib
matplotlib.use('Agg')

import os, sys
import glob
import random
import time
import imgaug
from imgaug import augmenters as iaa
import nibabel as nib
import SimpleITK as sitk
import tensorflow as tf
from tqdm import tqdm
import pandas as pd 
import numpy as np 
from six.moves import range
# sys.path.append('..')
from ..helpers.utils import *
from skimage.restoration import denoise_wavelet


class DataGeneratoronFly(tf.keras.utils.Sequence):
	"""
	"""
	def __init__(self, data_csv,
						nclasses = 2, 
						image_size=128, 
						batch_size=32, 
						nchannels=1,
						mode = 'Train',
						name = None
						samples_per_epoch=None, 
						transform=None):

		self.batch_size = batch_size
		self.image_size = image_size
		self.nchannels = nchannels
		self.nclasses = nclasses
		self.transform = transform
		self.name = name
		self.paths  = []
		self.labels = []

		labels = pd.read_csv(data_csv)['Y'].values
		paths = pd.read_csv(data_csv)['X'].values

		index = np.arange(len(paths))
		np.random.shuffle(index)

		labels = labels[index]
		paths  = paths[index] 
		
		if mode == 'Train':
			minarr = ([np.sum(labels == i) for i in range(nclasses)])
			mincount = np.min(minarr)
			for i in range(nclasses):
				self.paths.extend(paths[labels == i][:mincount])
				self.labels.extend(labels[labels == i][:mincount])

			self.paths = np.array(self.paths); self.labels = np.array(self.labels)

		elif mode == 'Valid':
			self.paths = np.array(paths); self.labels = np.array(labels)

		print("============== paths: {}, labels: {} ================".format(len(self.paths), len(self.labels)))
			
		assert len(np.unique(self.labels)) == nclasses
		self.len_arr = [sum(self.labels == arr) for arr in np.unique(self.labels)]

		index = np.arange(len(self.paths))
		np.random.shuffle(index)

		self.paths = self.paths[index]
		self.labels = self.labels[index]

		if samples_per_epoch is None:
			if mode == 'Train': self.samples_per_epoch = 50*len(self.paths)
			else: self.samples_per_epoch = len(self.paths)
		else:
			self.samples_per_epoch = samples_per_epoch
		
		if mode.lower() in ['train', 'valid', 'test']:
			self.mode = mode.lower()
		else:
			raise ValueError("mode should be one among ['Train', 'Valid', 'Test'], given argument: {}".format(mode))


	def __len__(self):
		"""
			Denotes the number of batches per epoch
		"""
		return int(np.floor(self.samples_per_epoch / self.batch_size))


	def __getitem__(self, index):
		# Generate indexes of the batch
		X1, X2, X3, y = self.__data_generation(index)
		if self.name == 'combined':
			return [X1, X2, X3], y
		elif self.name == "axial":
			return X1, y
		elif self.name == 'coronal':
			return X2, y
		elif self.name == 'sagittal':
			return X3, y


	def _standardize_volume(self, volume, mask=None):
		"""
			volume: volume which needs to be normalized
			mask: brain mask, only required if you prefer not to
				consider the effect of air in normalization
		"""
		if mask != None: volume = volume*mask

		mean = np.mean(volume[volume != 0])
		std = np.std(volume[volume != 0])
		
		return (volume - mean)/std


	def _normalize_volume(self, volume, mask=None, _type='MinMax'):
		"""
			volume: volume which needs to be normalized
			mask: brain mask, only required if you prefer not to 
				consider the effect of air in normalization
			_type: {'Max', 'MinMax', 'Sum'}
		"""
		if mask != None: volume = mask*volume
		
		min_vol = np.min(volume)
		max_vol = np.max(volume)
		sum_vol = np.sum(volume)

		if _type == 'MinMax':
			return (volume - min_vol) / (max_vol - min_vol)
		elif _type == 'Max':
			return volume/max_vol
		elif _type == 'Sum':
			return volume/sum_vol
		else:
			raise ValueError("Invalid _type, allowed values are: {}".format('Max, MinMax, Sum'))
		
	
	def _augmentation(self, volume):
		"""
			Augmenters that are safe to apply to masks
			Some, such as Affine, have settings that make them unsafe, so always
			test your augmentation on masks
		"""
		volume_shape = volume.shape
		det = self.transform.to_deterministic()
		volume = det.augment_image(volume)

		assert volume.shape == volume_shape, "Augmentation shouldn't change volume size"
		return volume

	def _resizeVolume(self, volume):
		"""
			resizes the original volume such that every patch is 
			75% of original volume

			volume: numpy 3d tensor
		"""
		ratio = 1.0

		orig_size = (int(self.image_size/ratio),
						int(self.image_size/ratio),
						int(self.image_size/ratio))
		resized_volume = resize_sitk(volume, orig_size)
		return resized_volume

	
	def _get_random_slices(self, volume):
		"""
		"""
		dimensions = volume.shape
		img = np.zeros((dimensions[0], dimensions[1], 3))
		x = np.random.randint(dimensions[0]//4, 3*dimensions[0]//4)
		z = np.random.randint(dimensions[1]//4, 3*dimensions[1]//4)
		y = np.random.randint(dimensions[2]//4, 3*dimensions[2]//4)
		slice_x = volume[x, :, :]
		slice_y = volume[:, y, :]
		slice_z = volume[:, :, z]
                                        
		return slice_x[..., None], slice_y[..., None], slice_z[..., None]

	
	def _center_align(self, volume):
		"""
		"""
		return volume

	def _axis_align(self, volume):
		"""
		"""
		return volume


	def __data_generation(self, index):
		"""
			balanced data loader
		"""
		X1, X2, X3 = [], [], []; Y = []
		nclass_batch = self.batch_size//self.nclasses
		for i in range(nclass_batch):
			for ii in np.unique(self.labels):
				try:
					pid_path = self.paths[self.labels == ii][int(index*nclass_batch + i) % self.len_arr[ii]]
					label = ii # np.eye(self.nclasses)[ii] 
					
					
					volume, affine, size = load_vol(pid_path)
					volume = self._axis_align(volume)
					volume = self._center_align(volume)
					volume = self._resizeVolume(volume)
					volume = self._standardize_volume(volume)
					volume = self._normalize_volume(volume)
									
					if (self.mode.lower() == 'train') and self.transform:
						volume = self._augmentation(volume)
					
					ax, sg, co = self._get_random_slices(volume)
					
					if ax.shape == sg.shape == co.shape: 
						X1.append(ax)
						X2.append(sg)
						X3.append(co)
						Y.append(label)
				except:
					continue

		X1, X2, X3, Y = np.array(X1), np.array(X2), np.array(X3), np.array(Y)

		index = np.arange(len(X1))
		np.random.shuffle(index)
		X1, X2, X3, Y = X1[index], X2[index], X3[index], Y[index]
		return X1, X2, X3, Y


if __name__ == '__main__':

	dir_path = os.path.abspath('../../csv/QC/train_test_fold_1/csv/')

	
	csv_path = os.path.join(dir_path, 'validation.csv')
 
	augmentation = iaa.SomeOf((0, 3), 
			[
				iaa.Fliplr(0.5),
				iaa.Flipud(0.5),
				iaa.Noop(),
				iaa.OneOf([iaa.Affine(rotate=90),
						   iaa.Affine(rotate=180),
						   iaa.Affine(rotate=270)]),
				# iaa.GaussianBlur(sigma=(0.0, 0.2)),
			])

	# Parameters
	train_transform_params = {'image_size': 128,
						  'batch_size': 4,
						  'nclasses': 2,
						  'nchannels': 1,
						  'samples_per_epoch': None,
						  'transform': augmentation
						 }

	valid_transform_params = {'image_size': 128,
						  'batch_size': 8,
						  'nclasses': 2,
						  'nchannels': 1,
						  'samples_per_epoch': None,                        
						  'transform': None
						 }

	# Generators
	training_generator = DataGeneratoronFly(csv_path, **train_transform_params)
	# print (training_generator.__len__())

	validation_generator = DataGeneratoronFly(csv_path, **valid_transform_params)
	print (validation_generator.__len__())


	"""
	for X, y in training_generator:
		print (X.shape, y.shape)
		print (y[:4])
		imshow(X[0,:,:,64, 0], X[1,:,:,64, 0], X[2,:,:,64, 0], X[3,:,:,64, 0])        
	
	for ep in range(5):
		print ("============================")
		for X, y in validation_generator:
			print (X.shape, y.shape)
			print (y[:4])
		# imshow(X[0,:,:,64, 0], X[1,:,:,64, 0], X[2,:,:,64, 0], X[3,:,:,64, 0])  
	"""
	
	import time        
	import matplotlib.pyplot as plt
	start_time = time.time()    
	for i, (X, y) in enumerate(validation_generator):
		elapsed_time = time.time() - start_time
		start_time = time.time()
		plt.subplot(1,3,1)
		plt.imshow(X[0][0][:,:,0])
		plt.subplot(1,3,2)
		plt.imshow(X[1][0][:,:,0])
		plt.subplot(1,3,3)
		plt.imshow(X[2][0][:,:,0])
		plt.title(str(y[0]))
		plt.savefig(str(i)+'_.png')
		print (y, type(X))   
		print (X[0].shape, X[1].shape, X[2].shape) 
		print (i, "Elapsed Time", np.round(elapsed_time, decimals=2), "seconds")
		pass	
