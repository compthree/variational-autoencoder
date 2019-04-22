import os
import numpy as np

# File names:
FILE_MODEL_DETAILS = 'model_details.json'

# Directories:
DIR_DATA = 'data'
DIR_MODELS = 'models'
DIR_SAVED_MODEL = 'saved_model'

# File paths:
PATH_ROOT = os.getcwd()
# NOTE: you may need to change this path name depending on your cv2 installation:
PATH_FACEDATA = "/anaconda/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"

# Lists:
LIST_TRAIN_TESTA_TESTB_KEYS = ['train', 'testa', 'testb']
LIST_ATTRIBUTES = ['graph',
                   'encoder_list',
                   'decoder_list',
                   'learning_rate',
                   'model_name',
                   'project_name',
                   'inputs_shape_list',
                   'inputs',
                   'inputs_shape',
                   '_inputs_shape',
                   'layer_shape_list',
                   '_layer_shape_list',
                   'optimizer',
                   'init',
                   'saver',
                   'sess',
                   'latent_means',
                   'latent_lstd2',
                   'latent_stdvs',
                   'encoder_shape_list',
                   '_encoder_shape_list',
                   'latent_shape',
                   '_latent_shape',
                   'latent_sample',
                   'output_means',
                   'output_lstd2',
                   'output_stdvs',
                   'decoder_shape_list',
                   '_decoder_shape_list',
                   'output_shape',
                   '_output_shape',
                   'output_sample',
                   'loss', # loss of the most recent mini-batch.
                   'project_path',
                   'models_path',
                   'model_path',
                   'num_trained_epochs',
                   'model_details_path',
                   'timestamp',
                   'avg_loss', # average mini-batch loss over an epoch.
                   'saved_model_path',
                   'builder',
                   'is_training',
                   'use_batch_normalization',
                   'averaging_axes_length',
                   'loss_type',
                   'loss_inputs_shape_list',
                   'percept_list',
                   '_percept_shape_list',
                   'percept_shape_list',
                   'encoder_loss_weight',
                   'decoder_loss_weight',
                   'is_variational',
                   'encoder_loss',
                   'decoder_loss']

LIST_MODEL_DETAILS_KEYS = ['encoder_list',
                          'decoder_list',
                          'inputs_shape_list',
                          'learning_rate',
                          'project_name',
                          'model_name',
                          'num_trained_epochs',
                          'use_batch_normalization',
                          'loss_type',
                          'encoder_loss_weight',
                          'decoder_loss_weight',
                          'is_variational']

LIST_MODEL_DETAILS_REQUIRED_KEYS = ['encoder_list',
                                    'decoder_list',
                                    'inputs_shape_list',
                                    'learning_rate',
                                    'project_name',
                                    'model_name',
                                    'num_trained_epochs',
                                    'use_batch_normalization',
                                    'loss_type',
                                    'encoder_loss_weight',
                                    'decoder_loss_weight',
                                    'is_variational']

LIST_MODEL_DETAILS_OPTIONAL_KEYS = ['percept_list',
                                    'averaging_axes_length']

PERCEPT_LIST = [{'layer_type': 're-size', 'output_shape': [224, 224], 'loss_weight': 0},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [64],  'pool_shape': [1, 1], 'activation': 'relu', 'loss_weight': 0.333},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [64],  'pool_shape': [2, 2], 'activation': 'relu', 'loss_weight': 0},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [128], 'pool_shape': [1, 1], 'activation': 'relu', 'loss_weight': 0.333},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [128], 'pool_shape': [2, 2], 'activation': 'relu', 'loss_weight': 0},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [256], 'pool_shape': [1, 1], 'activation': 'relu', 'loss_weight': 0.333}]

# Example 1 of an 'inputs_shape_list, encoder_list, decoder_list, model_details_dict' quadruple.
# The interpretation of this example is hopefully self-evident from the choice of names or from
# quick inspection of the source code in 'variational_autoencoder.py'.
inputs_shape_list = [64, 64, 3]
encoder_list = [{'layer_type': 'convolu', 'kernel_shape': [4, 4], 'output_chann': [32],  'pool_shape': [2, 2], 'activation': 'relu'},
                {'layer_type': 'convolu', 'kernel_shape': [4, 4], 'output_chann': [64],  'pool_shape': [2, 2], 'activation': 'relu'},
                {'layer_type': 'convolu', 'kernel_shape': [4, 4], 'output_chann': [128], 'pool_shape': [2, 2], 'activation': 'relu'},
                {'layer_type': 'convolu', 'kernel_shape': [4, 4], 'output_chann': [256], 'pool_shape': [2, 2], 'activation': 'relu'},
                {'layer_type': 'reshape', 'output_shape': [4 * 4 * 256]},
                {'layer_type': 'full_cn', 'output_shape': [100]}]
decoder_list = [{'layer_type': 'full_cn', 'output_shape': [4 * 4 * 256], 'activation': 'relu'},
                {'layer_type': 'reshape', 'output_shape': [4, 4, 256]},
                {'layer_type': 'deconvo', 'kernel_shape': [4, 4], 'output_chann': [128], 'pool_shape': [2, 2], 'activation': 'relu'},
                {'layer_type': 'deconvo', 'kernel_shape': [4, 4], 'output_chann': [64],  'pool_shape': [2, 2], 'activation': 'relu'},
                {'layer_type': 'deconvo', 'kernel_shape': [4, 4], 'output_chann': [32],  'pool_shape': [2, 2], 'activation': 'relu'},
                {'layer_type': 'deconvo', 'kernel_shape': [4, 4], 'output_chann': [3],   'pool_shape': [2, 2], 'activation': 'sigmoid'}]
# For perceptual loss only:
percept_list = [{'layer_type': 're-size', 'output_shape': [224, 224], 'loss_weight': 0},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [64],  'pool_shape': [1, 1], 'activation': 'relu', 'loss_weight': 0.333},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [64],  'pool_shape': [2, 2], 'activation': 'relu', 'loss_weight': 0},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [128], 'pool_shape': [1, 1], 'activation': 'relu', 'loss_weight': 0.333},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [128], 'pool_shape': [2, 2], 'activation': 'relu', 'loss_weight': 0},
                {'layer_type': 'convolu', 'kernel_shape': [3, 3], 'output_chann': [256], 'pool_shape': [1, 1], 'activation': 'relu', 'loss_weight': 0.333}]

model_details_dict = {'encoder_list': encoder_list,
                      'decoder_list': decoder_list,
                      'percept_list': percept_list,
                      'inputs_shape_list': inputs_shape_list,
                      'project_name': 'insert_project_name_here',
                      'model_name': 'insert_model_name_here',
                      'loss_type': 'perceptual',
                      'is_variational': True, # True = variational autoencoder, False = autoencoder.
                      'encoder_loss_weight': 1.0,
                      'decoder_loss_weight': 1.0,
                      'learning_rate': 0.001}

# Example 2 of an 'inputs_shape_list, encoder_list, decoder_list, model_details_dict' quadruple.
# The interpretation of this example is hopefully self-evident from the choice of names or from
# quick inspection of the source code in 'variational_autoencoder.py'.
inputs_shape_list = [28, 28, 1]
encoder_list = [{'layer_type': 'reshape'},
                {'layer_type': 'full_cn', 'output_shape': [500], 'activation': 'relu'},
                {'layer_type': 'full_cn', 'output_shape': [250], 'activation': 'relu'},
                {'layer_type': 'full_cn', 'output_shape': [2]}]

decoder_list = [{'layer_type': 'full_cn', 'output_shape': [250], 'activation': 'relu'},
                {'layer_type': 'full_cn', 'output_shape': [500], 'activation': 'relu'},
                {'layer_type': 'full_cn', 'output_shape': [int(np.prod(inputs_shape_list))], 'activation': 'sigmoid'},
                {'layer_type': 'reshape', 'output_shape': inputs_shape_list}]

model_details_dict = {'encoder_list': encoder_list,
                      'decoder_list': decoder_list,
                      'inputs_shape_list': inputs_shape_list,
                      'project_name': 'insert_project_name_here',
                      'model_name': 'insert_model_name_here',
                      'loss_type': 'pixel',
                      'is_variational': True,
                      'encoder_loss_weight': 1.0,
                      'decoder_loss_weight': 1.0,
                      'learning_rate': 0.001,
                      'use_batch_normalization': True,
                      'averaging_axes_length': 'long'} # 'long' = average over all spatial dimensions together,
                                                       # 'short' = average over each spatial dimension separately.

