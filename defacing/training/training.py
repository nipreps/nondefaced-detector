import argparse
from keras.utils import multi_gpu_model
from ..models.modelN import Submodel, CombinedClassifier
from ..helpers.metrics import specificity, sensitivity
from ..helpers.utils import *
from ..dataloaders.dataloader import DataGeneratoronFly
from keras.models import load_model
from keras import losses, metrics
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam
from keras.losses import binary_crossentropy,mse
from imgaug import augmenters as iaa
from keras import backend as K
import os
import sys
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import timeit

# Random Seeds
# np.random.seed(0)
# random.seed(0)
# tf.set_random_seed(0)


class trainer(object):
    """
        training module for mri deidentification module

        train_csv_path:
        validation_csv_path:

        model_save_path: path to save models
        resume_weights_path: weights path to resume training
        initial_epoch:
        n_Epoch: total number of epochs of training

    """

    def __init__(self,
                 train_csv_path,
                 valid_csv_path,
                 job_info_file,
                 model_save_path = '../weights',
                 image_size = 64,
                 batch_size=32,
                 initial_epoch=0,
                 nepochs=50,
                 dropout=0.4,
                 nclasses = 2,
                 nchannels = 1,
                 gpus=0):
        
        self.augmentation = iaa.SomeOf((0, 3),
                                  [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Noop(),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            # iaa.GaussianBlur(sigma=(0.0, 0.2)),
        ])

        self.image_size = image_size

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        logdir_path = os.path.join(model_save_path, 'tb_logs')
        if not os.path.exists(logdir_path):
            os.makedirs(logdir_path)

        self.train_csv_path = train_csv_path
        self.valid_csv_path = valid_csv_path
       	self.job_info_file = job_info_file
        self.model_save_path = model_save_path
        self.logdir_path = logdir_path
        self.initial_epoch = initial_epoch
        self.nepochs = nepochs
        self.gpus = gpus
        self.dropout = dropout
        self.batch_size = batch_size
        self.nclasses = nclasses
        self.nchannels = nchannels


    
    def train(self):
        
        names = ['axial', 'coronal', 'sagittal', 'combined']

        for name in names:
            # Parameters
            train_transform_params = {'image_size': self.image_size,
                                      'batch_size': self.batch_size,
                                      'nclasses': self.nclasses,
                                      'nchannels': self.nchannels,
                                      'mode': 'Train',
                                      'name': name,
                                      'samples_per_epoch': None,
                                      'transform': self.augmentation
                                      }

            valid_transform_params = {'image_size': self.image_size, 
                                      'batch_size': self.batch_size,
                                      'nclasses': self.nclasses,
                                      'nchannels': self.nchannels,
                                      'mode': 'Valid',
                                      'name': name,
                                      'samples_per_epoch': None,
                                      'transform': None
                                      }

            self.training_generator = DataGeneratoronFly(
                        self.train_csv_path, **train_transform_params)
            self.validation_generator = DataGeneratoronFly(
                        self.valid_csv_path, **valid_transform_params)
            
            print("No. of training and validation batches are:",
                self.training_generator.__len__(),
                self.validation_generator.__len__())
            
            os.makedirs(os.path.join(self.logdir_path, name), exist_ok = True)
            self.tbCallback = TensorBoard(log_dir=self.logdir_path,
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=False)

            
            os.makedirs(os.path.join(self.model_save_path, name), exist_ok = True)
            self.model_checkpoint = ModelCheckpoint(
                os.path.join(
                    self.model_save_path,
                    name,
                    'best-wts.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min')
            ## model defn
            
            if name == 'combined':
                lr = 5e-5 
                model = CombinedClassifier(input_shape = (self.image_size, self.image_size), 
                                dropout = self.dropout, wts_root = self.model_save_path)
            else:
                lr = 1e-4
                model = Submodel(input_shape = (self.image_size, self.image_size), dropout = self.dropout, name = name,
                                include_top = True, weights=None)

            dataloaders = [self.training_generator, self.validation_generator]
            self.fit(model, dataloaders, lr)


    def fit(self, model, dataloaders, lr=1e-3):
        """
            transfer_learning: trains decoder network for 6 epochs
                and finetunes whole network for n_Epochs
        """

        with open(self.job_info_file, 'a') as fh:
            model.summary(print_fn=lambda x : fh.write(x + '\n'))
        

        print("*"*10 + "Training from scratch" + "*"*10)
        try:
            # pass
            model = multi_gpu_model(
                model, gpus=self.gpus, cpu_relocation=True)
            print("Training using multiple GPUs..")
        except BaseException:
            print("Training using single GPU or CPU..")

        print(model.summary())
        
        model.compile(loss = 'binary_crossentropy', 
                        optimizer = 'adam', # Adam(lr=lr, amsgrad=True), 
                        metrics=['accuracy',
                                    sensitivity, 
                                    specificity])

        model.fit_generator(
            generator=dataloaders[0],
            epochs=self.nepochs,
            steps_per_epoch = dataloaders[0].__len__(),
            verbose=1,
            validation_data=dataloaders[1],
            validation_steps = dataloaders[1].__len__(),
            callbacks=[
                self.tbCallback,
                self.model_checkpoint],
            use_multiprocessing=True,
            workers=32,
            initial_epoch=self.initial_epoch)
	
        del model
        K.clear_session()
