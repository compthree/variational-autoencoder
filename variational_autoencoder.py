import os
import json
import time
import datetime
import numpy as np
import tensorflow as tf
import constants as const
import utils.utils as utils

class VariationalAutoencoder(object):
    
    # A list of class attributes:
    __slots__ = const.LIST_ATTRIBUTES

    # Class setter:
    def __setattr__(self, attr_str, value):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        # We reinitialize the class if any of these attributes are directly changed:
        if attr_str in ['encoder_list',
                        'decoder_list',
                        'inputs_shape_list',
                        'learning_rate']:
            object.__setattr__(self, attr_str, value)
            self.__init__()
            
        # Otherwise, we do not allow the user to (directly) update the attributes:
        else:
            raise NotImplementedError
            
    # Class deleter:
    def __delattr__(self, attr_str):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        # We don't allow (direct) deletion of attributes:
        if attr_str in []:
            object.__delattr__(self, attr_str)
            
        else:
            raise NotImplementedError

    # Class initializer:
    def __init__(self, model_details_dict):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # The user must specify a project name and a model name:
        for key in ['model_name', 'project_name']:
            assert key in model_details_dict
            assert isinstance(model_details_dict[key], str)

        # Set the inputs to class attributes:
        for key in model_details_dict:
            object.__setattr__(self, key, model_details_dict[key])

        # Set a timestamp as a class attribute:
        object.__setattr__(self, 'timestamp', datetime.datetime.now().isoformat().replace(':', '-'))
        
        # Set the model details path and its parent directory paths as class attributes:
        self.set_model_details_path()

        # Load the model details if the necessary file exists:
        if os.path.exists(self.model_details_path):
            self.load_model_details()

        # Otherwise, make a new directory to store the model details and dump them there:
        else:
            self.make_model_path()

        # Include any of these potentially missing class attributes:
        if not hasattr(self, 'learning_rate'):
            object.__setattr__(self, 'learning_rate', 0.001)
        if not hasattr(self, 'num_trained_epochs'):
            object.__setattr__(self, 'num_trained_epochs', 0)
        if not hasattr(self, 'use_batch_normalization'):
            object.__setattr__(self, 'use_batch_normalization', False)
        if not hasattr(self, 'loss_type'):
            object.__setattr__(self, 'loss_type', 'pixel')
        if not hasattr(self, 'encoder_loss_weight'):
            object.__setattr__(self, 'encoder_loss_weight', 1.0)
        if not hasattr(self, 'decoder_loss_weight'):
            object.__setattr__(self, 'decoder_loss_weight', 0.5)
        if not hasattr(self, 'is_variational'):
            object.__setattr__(self, 'is_variational', True)
        if self.use_batch_normalization and not hasattr(self, 'averaging_axes_length'):
            object.__setattr__(self, 'averaging_axes_length', 'long')

        # Save the model details:
        self.save_model_details()
        
        # Initialize a new empty graph:
        tf.reset_default_graph()
        object.__setattr__(self, 'graph', tf.Graph())
        
        # Fill the graph:
        with self.graph.as_default():
        
            # Inputs placeholder:
            inputs = tf.placeholder(name = 'inputs',
                                    dtype = tf.float32,
                                    shape = [None] + self.inputs_shape_list)
            object.__setattr__(self, 'inputs', inputs)

            # Training flag:
            is_training = tf.placeholder_with_default(False,
                                                      name = 'training_flag',
                                                      shape = [])
            object.__setattr__(self, 'is_training', is_training)
            
            # Create the variational autoencoder:
            with tf.variable_scope('variational_autoencoder', reuse = tf.AUTO_REUSE):

                # Initialize the layer shapes (tensors) class attribute:
                object.__setattr__(self, '_inputs_shape', tf.shape(inputs, name = '_inputs_shape'))
                object.__setattr__(self, '_layer_shape_list', [self._inputs_shape])

                # Create the neural network and capture the output:
                self._create_network()
            output_sample = tf.identity(self.output_sample, name = 'output_sample')

            # Create the loss function:
            with tf.variable_scope('loss_function', reuse = tf.AUTO_REUSE):
                self._compute_losses()
            loss = tf.identity(self.loss, name = self.loss_type + '_loss')

            # Create the optimizer:
            with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'variational_autoencoder')
                optimizer = (tf.train
                    .AdamOptimizer(learning_rate = self.learning_rate)
                    .minimize(self.loss, var_list = var_list))
            object.__setattr__(self, 'optimizer', optimizer)

            # Create the variable initializer:
            init = tf.global_variables_initializer()
            object.__setattr__(self, 'init', init)

            # Create the saver:
            saver = tf.train.Saver()
            object.__setattr__(self, 'saver', saver)

            # Create a builder for making a SavedModel (to be implemented later):
            # builder = tf.saved_model.builder.SavedModelBuilder(self.saved_model_path + '_' + self.timestamp)
            # object.__setattr__(self, 'builder', builder)

    # Define the 'model_details_dict' class attribute:
    @property
    def model_details_dict(self):

        model_details_dict = {}

        if hasattr(self, 'encoder_list'):
            model_details_dict['encoder_list'] = self.encoder_list

        if hasattr(self, 'decoder_list'):
            model_details_dict['decoder_list'] = self.decoder_list

        if hasattr(self, 'percept_list'):
            model_details_dict['percept_list'] = self.percept_list

        if hasattr(self, 'inputs_shape_list'):
            model_details_dict['inputs_shape_list'] = self.inputs_shape_list

        if hasattr(self, 'learning_rate'):
            model_details_dict['learning_rate'] = self.learning_rate

        if hasattr(self, 'project_name'):
            model_details_dict['project_name'] = self.project_name

        if hasattr(self, 'model_name'):
            model_details_dict['model_name'] = self.model_name

        if hasattr(self, 'num_trained_epochs'):
            model_details_dict['num_trained_epochs'] = self.num_trained_epochs

        if hasattr(self, 'use_batch_normalization'):
            model_details_dict['use_batch_normalization'] = self.use_batch_normalization

        if hasattr(self, 'averaging_axes_length'):
            model_details_dict['averaging_axes_length'] = self.averaging_axes_length

        if hasattr(self, 'loss_type'):
            model_details_dict['loss_type'] = self.loss_type

        if hasattr(self,'encoder_loss_weight'):
            model_details_dict['encoder_loss_weight'] = self.encoder_loss_weight

        if hasattr(self,'decoder_loss_weight'):
            model_details_dict['decoder_loss_weight'] = self.decoder_loss_weight

        if hasattr(self, 'is_variational'):
            model_details_dict['is_variational'] = self.is_variational

        return model_details_dict

    # Enter a new context to start a tensorflow session:
    def __enter__(self):
        
        # Launch the session:
        sess = tf.Session(graph = self.graph)
        object.__setattr__(self, 'sess', sess)

        # Run the variable initializer (must do this before getting layer shapes below):
        self.sess.run(self.init)

        # Set the input, latent, and output shapes (numbers) as class attributes:
        object.__setattr__(self, 'inputs_shape', self.get_inputs_shape())
        object.__setattr__(self, 'latent_shape', self.get_latent_shape())
        object.__setattr__(self, 'output_shape', self.get_output_shape())

        # Set the layer shapes (numbers), without the batch dimension, as class attributes:
        object.__setattr__(self, 'layer_shape_list',   self.get_layer_shape_list())
        object.__setattr__(self, 'encoder_shape_list', self.get_encoder_shape_list())
        object.__setattr__(self, 'decoder_shape_list', self.get_decoder_shape_list())

        # Set the layer shapes for the perceptual loss layers:
        if self.loss_type == 'perceptual':
            object.__setattr__(self, 'percept_shape_list', self.get_percept_shape_list())

        # Try to load a previously trained model for either inference or future training:
        checkpoint_file_name = os.path.join(self.model_path, 'checkpoint')
        if os.path.exists(checkpoint_file_name):
            try:
                self.load_model()
            except:
                raise Exception('Failed to restore model at {}.'.format(self.model_path))
    
        return self
        
    # Exit the context:
    def __exit__(self, exc_type, exc_value, traceback):
        
        # Close the session:
        self.sess.close()
        
        # Reset the graph:
        tf.reset_default_graph()
        
    # Create the variational autoencoder network:
    def _create_network(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        # Create the encoder:
        with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
            shifted_inputs = tf.add(self.inputs, -0.5, name = 'shift')
            latent_means, latent_lstd2, latent_stdvs = self._create_subnetwork(shifted_inputs,
                                                                               self.encoder_list)
        
        # Name the output tensors from the encoder subgraph:
        latent_means = tf.identity(latent_means, name = 'latent_means')
        latent_lstd2 = tf.identity(latent_lstd2, name = 'latent_lstd2')
        latent_stdvs = tf.identity(latent_stdvs, name = 'latent_stdvs')
        
        # Set the output tensors from the encoder subgraph as class attributes:
        object.__setattr__(self, 'latent_means', latent_means)
        object.__setattr__(self, 'latent_lstd2', latent_lstd2)
        object.__setattr__(self, 'latent_stdvs', latent_stdvs)
        
        # Set the layer shapes (tensors) of the encoder subgraph as class attributes:
        object.__setattr__(self, '_encoder_shape_list', self._layer_shape_list)
        object.__setattr__(self, '_latent_shape',       self._layer_shape_list[-1])
        
        # Create the latent variable sampler subgraph:
        with tf.variable_scope('latent_sample', reuse = tf.AUTO_REUSE):

            if self.is_variational:
                # Randomly draw from the latent variable distribution:
                latent_sample = tf.random_normal(tf.shape(self.latent_means),
                                                 mean = self.latent_means,
                                                 stddev = self.latent_stdvs,
                                                 name = 'latent_sample',
                                                 dtype = tf.float32)
            else:
                # Encoder subnetwork outputs feed directly into the decoder subnetwork:
                latent_sample = self.latent_means

            # Set the latent sample as a class attribute:
            object.__setattr__(self, 'latent_sample', latent_sample)
        
        # Create the decoder:
        with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
            output_means, output_lstd2, output_stdvs = self._create_subnetwork(self.latent_sample,
                                                                               self.decoder_list)

        # Name the output tensors from the decoder subgraph:
        output_means = tf.identity(output_means, name = 'output_means')
        output_lstd2 = tf.identity(output_lstd2, name = 'output_lstd2')
        output_stdvs = tf.identity(output_stdvs, name = 'output_stdvs')
        
        # Set the output tensors from the decoder subgraph as class attributes:
        object.__setattr__(self, 'output_means', output_means)
        object.__setattr__(self, 'output_lstd2', output_lstd2)
        object.__setattr__(self, 'output_stdvs', output_stdvs)
        
        # Set the layer shapes of the decoder subgraph as class attributes:
        object.__setattr__(self, '_decoder_shape_list', self._layer_shape_list[len(self._encoder_shape_list) - 1:])
        object.__setattr__(self, '_output_shape',       self._layer_shape_list[-1])

        # Create the output variable sampler subgraph:
        with tf.variable_scope('output_sample', reuse = tf.AUTO_REUSE):

            # Just use the mean for the output sample (instead of a random draw):
            output_sample = tf.identity(self.output_means, 'output_sample')

            # Set the ouput sample as a class attribute:
            object.__setattr__(self, 'output_sample', output_sample)
        
    def _create_subnetwork(self, inputs, layer_list):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        layer = inputs
        
        # Get the index of the last layer in 'layer_list' whose type is not 'reshape':
        j = min([k for k in range(len(layer_list)) if \
                 all([layer_dict['layer_type'] == 'reshape' \
                 for layer_dict in layer_list[k + 1:]])])

        # Create each layer as specified in 'layer_list':
        for i, layer_dict in enumerate(layer_list):
            
            # Create layers up to but not including layer j:
            if i < j:

                # Compute the i:th layer output:
                layer = self._create_layer(layer, layer_dict, tag = str(i + 1))

                # Set the i:th layer shape to a class attribute:
                object.__setattr__(self,
                                   '_layer_shape_list',
                                    self._layer_shape_list + [tf.shape(layer,
                                                                       name = 'layer_' + str(i + 1) + '_output_shape')])
            
            # Create the j:th layer:
            elif i == j:

                # Layer j is actually two parallel layers. One layer outputs means and the other
                # outputs logarithms of variances (lstd2) for the Gaussian distribution of the latent random variable:
                means_layer_dict = dict(layer_dict)
                lstd2_layer_dict = dict(layer_dict)

                # The lstd2 layer can have any sign, so we do not apply an activation to it:
                lstd2_layer_dict['activation'] = 'identity'
                
                # Compute the j:th layer outputs:
                output_means = self._create_layer(layer, means_layer_dict, tag = 'means')
                output_lstd2 = self._create_layer(layer, lstd2_layer_dict, tag = 'lstd2')
                output_stdvs = tf.identity(tf.sqrt(tf.exp(output_lstd2)),  name = 'stdvs')
                
            # Create the remaining layers:
            else:

                # The remaining layers are 'reshape' layers (really, there should be no more than one): 
                assert layer_dict['layer_type'] == 'reshape'
                 
                # Compute the i:th layer output:
                output_means = self._create_layer(output_means, layer_dict, tag = str(i + 1) + '_means')
                output_lstd2 = self._create_layer(output_lstd2, layer_dict, tag = str(i + 1) + '_lstd2')
                output_stdvs = self._create_layer(output_stdvs, layer_dict, tag = str(i + 1) + '_stdvs')
                
            # Set the i:th layer shape to a class attribute:
            if i >= j:
                object.__setattr__(self,
                                   '_layer_shape_list',
                                   self._layer_shape_list + [tf.shape(output_means,
                                                                      name = 'layer_' + str(i + 1) + '_output_shape')])
                 
        return output_means, output_lstd2, output_stdvs
    
    def _create_layer(self, inputs, layer_dict, tag):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        layer_type = layer_dict['layer_type']
        with tf.variable_scope('layer_' + tag + '_'+ layer_type, reuse = tf.AUTO_REUSE):
        
            # Fully connected layer:
            if layer_type == 'full_cn':
                output = self._create_full_cn_layer(inputs, layer_dict, tag)
                
            # Convolution layer:
            elif layer_type == 'convolu':
                output = self._create_convolu_layer(inputs, layer_dict, tag)
                
            # Deconvolution layer (can only accept 4-D tensors):
            elif layer_type == 'deconvo':
                output = self._create_deconvo_layer(inputs, layer_dict, tag)
                    
            # Reshape layer:
            elif layer_type == 'reshape':
                output = self._create_reshape_layer(inputs, layer_dict, tag)

            elif layer_type == 're-size':
                output = self._create_re_size_layer(inputs, layer_dict, tag)

            else:
                raise Exception('Layer type {} must be, but is not one of, "convolu", "deconvo", "full_cn", "reshape", or "re-size".'.format(layer_type))

        return output        

    def _create_full_cn_layer(self, inputs, layer_dict, tag):

        '''

        Description:

        Inputs:

        Output:

        '''
        
        # Because this layer is fully connected, the input tensor must have rank 2:
        self.check_tensor_rank(inputs, [2])
        
        # Get the activation function:
        activation = self._get_activation(layer_dict)
        
        # Get the shape of the weights matrix and biases vector:
        inputs_shape = inputs.get_shape().as_list()
        weight_shape = inputs_shape[1:] + layer_dict['output_shape']
        biases_shape = layer_dict['output_shape']

        # Initialize the weights and biases using the 'fan-in' method:
        with tf.variable_scope('variable_init', reuse = tf.AUTO_REUSE):
            num_inputs = tf.reduce_prod(inputs_shape[1:])
            num_inputs = tf.cast(num_inputs, dtype = tf.float32, name = 'number_of_inputs')
            stddev = tf.sqrt(2 / num_inputs, name = 'fan_in_stddev')
            weight_init = tf.truncated_normal(weight_shape, stddev = stddev, dtype = tf.float32)
            biases_init = tf.truncated_normal(biases_shape, stddev = stddev, dtype = tf.float32)

        # Create the weights and biases variables:
        weight = tf.get_variable(name  = 'weight_' + tag,
                                 dtype = tf.float32,
                                 initializer = weight_init)
        biases = tf.get_variable(name  = 'biases_' + tag,
                                 dtype = tf.float32,
                                 initializer = biases_init)

        # Layer operations:
        output = tf.matmul(inputs, weight, name = 'weight_mul_' + tag)
        output = tf.add(output, biases,    name = 'biases_add_' + tag)

        # Perform batch-normalization:
        if self.use_batch_normalization:
            with tf.variable_scope('batch_normalization', reuse = tf.AUTO_REUSE):
                output = self._perform_batch_normalization(output,
                                                           averaging_axes_length = self.averaging_axes_length)

        # Activation:
        output = activation(output, name = 'activation_' + tag)
            
        return output
    
    def _create_convolu_layer(self,
                              inputs,
                              layer_dict,
                              tag):

        '''

        Description: Creates a single convolution layer:
            1. convolution
            2. add biases
            3. activation
            4. max-pool

        Inputs:

        Output:

        '''
        
        # Get the activation function:
        activation = self._get_activation(layer_dict)
        
        # Get the shape of the weights matrix and biases vector:
        inputs_shape = inputs.get_shape().as_list()
        kernel_shape = layer_dict['kernel_shape'] + inputs_shape[-1:] + layer_dict['output_chann']
        biases_shape = layer_dict['output_chann']
        
        # Initialize the weights and biases using the 'fan-in' method:
        with tf.variable_scope('variable_init', reuse = tf.AUTO_REUSE):
            num_inputs = tf.reduce_prod(kernel_shape[:-1])
            num_inputs = tf.cast(num_inputs, dtype = tf.float32, name = 'number_of_inputs')
            stddev = tf.sqrt(2 / num_inputs, name = 'fan_in_stddev')
            kernel_init = tf.truncated_normal(kernel_shape, stddev = stddev, dtype = tf.float32)
            biases_init = tf.truncated_normal(biases_shape, stddev = stddev, dtype = tf.float32)

        # Create the weights and biases variables:
        kernel = tf.get_variable(name  = 'kernel_' + tag,
                                 dtype = tf.float32,
                                 initializer = kernel_init)
        biases = tf.get_variable(name  = 'biases_' + tag,
                                 dtype = tf.float32,
                                 initializer = biases_init)
    
        # Layer operations:
        output = tf.nn.convolution(inputs,
                                   kernel,
                                   padding = 'SAME',
                                   name = 'convolve_' + tag)
        output = tf.add(output, biases, name = 'biases_add_' + tag)

        # Perform batch-normalization:
        if self.use_batch_normalization:
            with tf.variable_scope('batch_normalization', reuse = tf.AUTO_REUSE):
                output = self._perform_batch_normalization(output,
                                                           averaging_axes_length = self.averaging_axes_length)

        # Activation and max-pooling:
        output = activation(output, name = 'activation_' + tag)
        output = tf.nn.pool(output,
                            window_shape = layer_dict['pool_shape'],
                            pooling_type = 'MAX',
                            padding = 'SAME',
                            strides = layer_dict['pool_shape'],
                            name = 'mx_pooling_' + tag)
        
        return output
        
    def _create_deconvo_layer(self, inputs, layer_dict, tag):

        '''

        Description: creates a single deconvolution layer:
            1. unpool
            2. "transpose" convolution
            3. add biases
            4. activation

        Inputs:

        Output:

        '''

        # 'conv2d_transpose' requires its input tensor to be of rank 4:
        self.check_tensor_rank(inputs, [4])

        # Apply the unpooling operation first, before getting the shape of the
        # input tensor below, as this operation changes that shape:
        inputs = self._unpool(inputs,
                              window_shape = layer_dict['pool_shape'])
        
        # Get the activation function:
        activation = self._get_activation(layer_dict)
        
        # Get the shape of the weights matrix and biases vector:
        inputs_shape = inputs.get_shape().as_list()
        kernel_shape = layer_dict['kernel_shape'] + layer_dict['output_chann'] + inputs_shape[-1:]
        biases_shape = layer_dict['output_chann']
        
        # Initialize the weights and biases using the 'fan-in' method:
        with tf.variable_scope('variable_init', reuse = tf.AUTO_REUSE):
            num_inputs = tf.reduce_prod(kernel_shape[:-2] + kernel_shape[-1:])
            num_inputs = tf.cast(num_inputs, dtype = tf.float32, name = 'number_of_inputs')
            stddev = tf.sqrt(2 / num_inputs, name = 'fan_in_stddev')
            kernel_init = tf.truncated_normal(kernel_shape, stddev = stddev, dtype = tf.float32)
            biases_init = tf.truncated_normal(biases_shape, stddev = stddev, dtype = tf.float32)
        
        # Create the weights and biases variables:
        kernel = tf.get_variable(name  = 'kernel_' + tag,
                                 dtype = tf.float32,
                                 initializer = kernel_init)
        biases = tf.get_variable(name  = 'biases_' + tag,
                                 dtype = tf.float32,
                                 initializer = biases_init)
        
        # Layer operations: 
        output = tf.nn.conv2d_transpose(inputs,
                                         kernel,
                                         strides = [1, 1, 1, 1],
                                         output_shape = tf.concat([tf.shape(inputs)[:3], layer_dict['output_chann']], axis = 0),
                                         padding = 'SAME',
                                         name = 'deconvolve_' + tag)
        output = tf.add(output, biases,  name = 'biases_add_' + tag)

       # Perform batch-normalization:
        if self.use_batch_normalization:
            with tf.variable_scope('batch_normalization', reuse = tf.AUTO_REUSE):
                output = self._perform_batch_normalization(output,
                                                           averaging_axes_length = self.averaging_axes_length)

        # Activation:
        output = activation(output, name = 'activation_' + tag)
        
        return output
    
    def _unpool(self, inputs, window_shape = [2, 2]):

        '''

        Description: an n-dimensional version of the unpooling operation from
        https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

        Borrowed from "https://github.com/tensorflow/tensorflow/issues/2169".

        Inputs:
            -'inputs' (tf.Tensor) = a tensor of shape [batch, d_0, d_1, ..., d_n, channels]
            -'window_shape' = a list of positive integers [m_0, m_1, ..., m_n].

        Output:
            - (tf.Tensor) = a tensor of shape [batch, d0 * m_0, d1 * m_1, ..., d_n * m_n, channels]

        '''
        
        with tf.variable_scope('unpooling_layer', reuse = tf.AUTO_REUSE):
            
            # Flatten the first (batch) dimension of 'inputs' into its second dimension:
            inputs_shape = inputs.get_shape().as_list()
            dim = len(inputs_shape[1:-1])
            output = tf.reshape(inputs, [-1] + inputs_shape[-dim:], name = 'flatten')

            # Insert zeros:
            for i in range(dim, 0, -1):
                output = tf.concat([output] + [tf.zeros_like(output)] * (window_shape[-i] - 1), i, name = 'insert_zeros')

            # Reshape appropriately:
            output_size = [-1] + [s * f for s, f in zip(inputs_shape[1:-1], window_shape[::-1])] + [inputs_shape[-1]]
            output = tf.reshape(output, output_size, name = 'reshape')
            
        return output
    
    def _create_reshape_layer(self, inputs, layer_dict, tag):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # Form the specified new shape while preserving the batch (first) dimension:
        if 'output_shape' in layer_dict:
            new_shape = tf.concat([tf.shape(inputs)[:1], layer_dict['output_shape']], axis = 0)

        # Otherwise flatten together all but the batch (first) dimension:
        else:
            new_shape = tf.concat([tf.shape(inputs)[:1], [tf.reduce_prod(inputs.get_shape()[1:])]], axis = 0)

        # Reshape the inputs:
        output = tf.reshape(inputs, new_shape, name = 'reshape_' + tag)
        
        return output

    def _create_re_size_layer(self, inputs, layer_dict, tag):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # 'tf.image.resize_images' requires its input tensor to be of rank 3 or 4:
        self.check_tensor_rank(inputs, [3, 4])

        output = tf.image.resize_images(inputs, layer_dict['output_shape'])

        return output
    
    def _get_activation(self, layer_dict):

        '''
        
        Description:

        Inputs:

        Output:

        '''
    
        if 'activation' not in layer_dict:
            return tf.identity

        elif layer_dict['activation'] == 'relu':
            return tf.nn.relu

        elif layer_dict['activation'] == 'sigmoid':
            return tf.nn.sigmoid

        elif layer_dict['activation'] == 'identity':
            return tf.identity
        
        else:
            raise Exception('Activation must be one of "relu", "sigmoid", or "identity".')

    def _perform_batch_normalization(self,
                                     inputs,
                                     train_decay = 0.99,
                                     averaging_axes_length = 'long'):

        '''Adopted from https://r2rt.com/implementing-batch-normalization-in-tensorflow.html'''

        # Prevent overflow from dividing by a batch variance of zero:
        epsilon = 1e-3

        # Two possible ways to average over datapoint features in a batch:
        if averaging_axes_length == 'long':
            # Average over all but the last 'channel' dimension of 'inputs':
            cutoff = len(inputs.get_shape().as_list()) - 1
        else:
            assert averaging_axes_length == 'short'
            # Average over only the 'batch' dimension of 'inputs':
            cutoff = 1

        # Get or initialize the shift and scale variables for batch normalization:
        scale = tf.get_variable(name = 'scale',
                                dtype = tf.float32,
                                shape = inputs.get_shape()[cutoff:],
                                initializer = tf.initializers.ones)
        shift = tf.get_variable(name = 'shift',
                                dtype = tf.float32,
                                shape = inputs.get_shape()[cutoff:],
                                initializer = tf.initializers.zeros)

        # Get or initialize the population mean and variance variables:
        # (Note: with 'trainable = False', these variables are not part
        # of the dataflow for training.)
        pop_mean = tf.get_variable(name = 'population_mean',
                                   dtype = tf.float32,
                                   shape = inputs.get_shape()[cutoff:],
                                   initializer = tf.initializers.zeros,
                                   trainable = False)
        pop_std2 = tf.get_variable(name = 'population_std2',
                                   dtype = tf.float32,
                                   shape = inputs.get_shape()[cutoff:],
                                   initializer = tf.initializers.ones,
                                   trainable = False)

        # Get the mean and variance of all datapoints in this batch:
        batch_mean, batch_std2 = tf.nn.moments(inputs, axes = list(range(cutoff)))

        # The 'decay' factor must be one during inference so we do not change the estimates
        # of the population mean and std2 for use in the batch normalization sub-layer:
        with tf.variable_scope('set_decay_weight', reuse = tf.AUTO_REUSE):
            decay = tf.cond(self.is_training,
                            lambda: train_decay, # Return if training.
                            lambda: 1.0,         # Return if not training.
                            name = 'train_condition')
            decay = tf.cast(decay, tf.float32)

        # Tensors 'train_mean' and 'train_std2' are not part of the dataflow for training.
        # Their only purpose is to update the population mean and std2 estimates:
        with tf.variable_scope('update_population_stats', reuse = tf.AUTO_REUSE):
            with tf.variable_scope('update_population_mean', reuse = tf.AUTO_REUSE):
                weighted_pop_mean   = tf.multiply(pop_mean,   decay,     name = 'weighted_pop_mean')
                weighted_batch_mean = tf.multiply(batch_mean, 1 - decay, name = 'weighted_batch_mean')
                train_mean = tf.assign(pop_mean, weighted_pop_mean + weighted_batch_mean)
            with tf.variable_scope('update_population_std2', reuse = tf.AUTO_REUSE):
                weighted_pop_std2   = tf.multiply(pop_std2,   decay,     name = 'weighted_pop_std2')
                weighted_batch_std2 = tf.multiply(batch_std2, 1 - decay, name = 'weighted_batch_std2')
                train_std2 = tf.assign(pop_std2, weighted_pop_std2 + weighted_batch_std2)

        # Because they're not part of the dataflow for training, 'train_mean' and 'train_std2'
        # are not updated unless we force updates using the 'tf.control_dependencies' context:
        with tf.control_dependencies([train_mean, train_std2]):
            use_mean, use_std2 = tf.cond(self.is_training,
                                         lambda: (batch_mean, batch_std2), # Return if training.
                                         lambda: (pop_mean,   pop_std2),   # Return if not training.
                                         name = 'set_batch_norm_mean_std2')

        # Finally compute the outputs of the batch-normalization sub-layer:
        return tf.nn.batch_normalization(inputs,
                                         use_mean,
                                         use_std2,
                                         shift,
                                         scale,
                                         epsilon)

    def _compute_losses(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # Compute the loss from the decoder subnetwork:
        with tf.variable_scope('decoder_loss', reuse = tf.AUTO_REUSE):

            # Compute the perceptual loss:
            if self.loss_type == 'perceptual':

                layer = tf.concat([self.inputs, self.output_means], axis = 0)
                decoder_loss = tf.constant(0, tf.float32, name = 'initialize_decoder_loss')

                # At least one loss weight must be positive:
                assert any([layer_dict['loss_weight'] > 0 for layer_dict in self.percept_list])

                # Get the running total of all positive weights:
                total = sum([max(layer_dict['loss_weight'], 0) for layer_dict in self.percept_list])

                # Build the perceptual loss network:
                for i, layer_dict in enumerate(self.percept_list):

                    # Compute the i:th layer output:
                    layer = self._create_layer(layer, layer_dict, tag = str(i + 1))

                    # Compute the loss at the i:th layer (ignore layers with nonpositive weights):
                    if layer_dict['loss_weight'] > 0:
                        with tf.variable_scope('layer_loss_' + str(i + 1), reuse = tf.AUTO_REUSE):
                            cut = tf.floordiv(tf.shape(layer)[0], 2)
                            source, reform = tf.split(layer, [cut, cut], axis = 0)
                            layer_loss = tf.reduce_sum(tf.square(source - reform),
                                                       axis = tf.range(1, tf.rank(source)))
                            layer_wght = tf.divide(layer_dict['loss_weight'], total, name = 'normalized_weight')
                            decoder_loss += tf.multiply(layer_wght, layer_loss, name = 'weighted_layer_loss')

                    # Set the i:th layer shape to a class attribute:
                    object.__setattr__(self,
                                       '_layer_shape_list',
                                        self._layer_shape_list + [tf.shape(layer,
                                                                           name = 'layer_' + str(i + 1) + '_output_shape')])

                # Set the layer shapes of the perceptual loss subgraph as class attributes:
                mark = len(self._encoder_shape_list) + len(self._decoder_shape_list)
                object.__setattr__(self, '_percept_shape_list', self._layer_shape_list[mark - 1:])
            
            # Compute the pixel loss:
            else:
                if self.loss_type != 'pixel':
                    raise Exception('Loss type must be either "perceptual" or "pixel".')

                decoder_loss = -tf.reduce_sum(self.inputs * tf.log(1e-10 + self.output_means)
                                              + (1 - self.inputs) * tf.log(1e-10 + 1 - self.output_means), 
                                              axis = tf.range(1, tf.rank(self.output_means)))

            decoder_loss = tf.reduce_mean(decoder_loss)

        if not self.is_variational:
            object.__setattr__(self, 'loss', decoder_loss)
            return

        # Compute the loss from the encoder subnetwork (KL-divergence):
        with tf.variable_scope('encoder_loss', reuse = tf.AUTO_REUSE):
            encoder_loss = -0.5 * tf.reduce_sum(1 + self.latent_lstd2
                                                  - tf.square(self.latent_means)
                                                  - tf.exp(self.latent_lstd2),
                                                axis = tf.range(1, tf.rank(self.latent_lstd2)))
            encoder_loss = tf.reduce_mean(encoder_loss)
        
        # Sum and average together the two components to get the overall loss:
        wgt_enc_loss = tf.multiply(self.encoder_loss_weight, encoder_loss, name = 'weighted_encoder_loss')
        wgt_dec_loss = tf.multiply(self.decoder_loss_weight, decoder_loss, name = 'weighted_decoder_loss')
        overall_loss = tf.add(encoder_loss, decoder_loss, name = 'combined_loss')
        
        # Set the loss as a class attribute:
        object.__setattr__(self, 'encoder_loss', encoder_loss)
        object.__setattr__(self, 'decoder_loss', decoder_loss)
        object.__setattr__(self, 'loss', overall_loss)

    def _get_tensor_value(self, output_tensor, feed_dict):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        return self.sess.run(output_tensor, feed_dict = feed_dict)
    
    def update_network(self, input_data):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        feed_dict = {self.inputs: input_data, self.is_training: True}
        _, loss = self._get_tensor_value((self.optimizer, self.loss), feed_dict)
        
        return loss
    
    def train_model(self,
                    input_data,
                    batch_size = 100,
                    display_step = 5,
                    num_training_epochs = 75):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        num_samples = len(input_data)
    
        print_start = time.time()
        epoch_begin = self.num_trained_epochs
        for epoch in range(epoch_begin, num_training_epochs):

            epoch_start = time.time()

            avg_loss = 0.
            num_batches_per_epoch = int(num_samples / batch_size)
            
            # Loop over all batches
            for i in range(num_batches_per_epoch):

                # Get the data batch:
                data_batch = input_data[i * batch_size: (i + 1) * batch_size]

                # Fit training using batch data:
                loss = self.update_network(data_batch)

                # Compute average loss
                avg_loss += loss * (batch_size / num_samples)

                # Check that the loss is not diverging:
                # print('loss:', loss)
                # print('encode:', self.encode(input_data[:1]))

            object.__setattr__(self, 'avg_loss', avg_loss)
            object.__setattr__(self, 'num_trained_epochs', self.num_trained_epochs + 1)

            # Print out progress and save model checkpoint:
            if (epoch + 1) % display_step == 0:
                
                delta = round(time.time() - print_start, 3)
                print_out = 'Epoch {} of {} completed. Total time elapsed so far: {}'
                print_out = print_out.format(epoch + 1, num_training_epochs, delta)
                
                print('\n')
                print(print_out)
                print('-' * len(print_out))
                print('Loss over all input data:', round(self.avg_loss, 3))
                print('Time to train last epoch:', round(time.time() - epoch_start, 3))

                # Save the model weights:
                self.save_model()
    
    def encode(self, input_data):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self.latent_means, feed_dict), self._get_tensor_value(self.latent_stdvs, feed_dict)
        
    def decode(self, latent_sample = None):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        if latent_sample is None:
            latent_sample = np.random.normal(size = [1] + list(self.get_latent_shape()))

        feed_dict = {self.latent_sample: latent_sample}
        return self._get_tensor_value(self.output_means, feed_dict)
    
    def reform(self, input_data):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self.output_sample, feed_dict)
    
    def restore_model(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        self.saver.restore(self.sess, self.save_path)

    def check_tensor_rank(self, tensor, rank_list):

        assert len(tensor.get_shape().as_list()) in rank_list
        
    def get_inputs_shape(self, input_data = None):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return self._get_tensor_value(self._inputs_shape, feed_dict)[1:]
        
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._inputs_shape, feed_dict)
    
    def get_latent_shape(self, input_data = None):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return self._get_tensor_value(self._latent_shape, feed_dict)[1:]
        
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._latent_shape, feed_dict)
    
    def get_output_shape(self, input_data = None):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return self._get_tensor_value(self._output_shape, feed_dict)[1:]
        
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._output_shape, feed_dict)
    
    def get_layer_shape_list(self, input_data = None):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return [x[1:] for x in self._get_tensor_value(self._layer_shape_list, feed_dict)]

        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._layer_shape_list, feed_dict)

    def get_encoder_shape_list(self, input_data = None):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return [x[1:] for x in self._get_tensor_value(self._encoder_shape_list, feed_dict)]
        
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._encoder_shape_list, feed_dict)
        
    def get_decoder_shape_list(self, input_data = None):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return [x[1:] for x in self._get_tensor_value(self._decoder_shape_list, feed_dict)]
        
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._decoder_shape_list, feed_dict)

    def get_percept_shape_list(self, input_data = None):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        if not hasattr(self, '_percept_shape_list'):
            return []

        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return [x[1:] for x in self._get_tensor_value(self._percept_shape_list, feed_dict)]
        
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._percept_shape_list, feed_dict)

    def set_project_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        project_path = os.path.join(const.PATH_ROOT, self.project_name)
        object.__setattr__(self, 'project_path', project_path)

    def set_models_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_project_path()
        models_path = os.path.join(self.project_path, const.DIR_MODELS)
        object.__setattr__(self, 'models_path', models_path)

    def set_model_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_models_path()
        model_path = os.path.join(self.models_path, self.model_name)
        object.__setattr__(self, 'model_path', model_path)
        saved_model_path = os.path.join(self.model_path, const.DIR_SAVED_MODEL)
        object.__setattr__(self, 'saved_model_path', saved_model_path)

    def set_model_details_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_model_path()
        model_details_path = os.path.join(self.model_path, const.MODEL_DETAILS)
        object.__setattr__(self, 'model_details_path', model_details_path)

    def make_project_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_project_path()
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)

    def make_models_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_models_path()
        if not os.path.exists(self.models_path):
            self.make_project_path()
            os.mkdir(self.models_path)

    def make_model_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_model_path()
        if not os.path.exists(self.model_path):
            self.make_models_path()
            object.__setattr__(self, 'model_name', self.model_name + '_' + self.timestamp)
            object.__setattr__(self, 'model_path', self.model_path + '_' + self.timestamp)
            saved_model_path = os.path.join(self.model_path, 'saved_model')
            object.__setattr__(self, 'saved_model_path', saved_model_path)
            os.mkdir(self.model_path)
            # os.mkdir(self.saved_model_path)
            self.set_model_details_path()

    def check_model_details_dict_format(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        model_details_dict = dict(self.model_details_dict)

        assert 'use_batch_normalization' in model_details_dict
        if model_details_dict['use_batch_normalization']:
            assert 'averaging_axes_length' in model_details_dict
            model_details_dict.pop('averaging_axes_length')
        assert set(model_details_dict) == set(const.LIST_MODEL_DETAIL_KEYS)

        # TO DO: check that the value for each key of 'model_details_dict'
        # is the correct datetype and has the correct format.

        # Also, check that the decoder means layer is an int between 0 and 1:

        return True

    def load_model_details(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        with open(self.model_details_path) as file_handle:

            loaded_model_details_dict = json.load(file_handle)

            for key in loaded_model_details_dict:

                if key not in self.model_details_dict:
                    continue

                if key == 'num_trained_epochs':
                    continue

                if self.model_details_dict[key] != loaded_model_details_dict[key]:
                    exception = 'The "input model details dict" does not equal ' \
                    + 'the "loaded model details dict" at the key "{}": '.format(key) \
                    + '{} != {}. '.format(self.model_details_dict[key],
                                          loaded_model_details_dict[key]) \
                    + 'If you mean to load an existing model, then drop this key ' \
                    + 'from the "input model details dict." If you mean to create a ' \
                    + 'new model, then use a model name different from ' \
                    + '"{}" because a model by that name already exists.'.format(self.model_name)
                    raise Exception(exception)

            for key in loaded_model_details_dict:
                object.__setattr__(self, key, loaded_model_details_dict[key])
            self.check_model_details_dict_format()

    def save_model_details(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.check_model_details_dict_format()

        with open(self.model_details_path, 'w+') as file_handle:
            json.dump(self.model_details_dict, file_handle, default = utils.default)

    def load_model(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # Load the weights and checkpoint files:
        assert os.path.exists(self.model_path)
        save_model_path = os.path.join(self.model_path, self.model_path.split('/')[-1])
        self.saver.restore(self.sess, save_model_path)

    def save_model(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # Save the weights and checkpoint files only:
        assert os.path.exists(self.model_path)
        save_model_path = os.path.join(self.model_path, self.model_path.split('/')[-1])
        self.saver.save(self.sess, save_model_path)
        self.save_model_details()

    def load_initial_loss_weights(self, weight_dict):

        if self.loss_type == 'pixel':
            pass

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'loss_function/decoder_loss')
        var_name_list = [''.join(var.name.split(':0')[:-1]) for var in var_list]

        input_weight_name_list = sorted([x for x in sorted(weight_dict.keys()) if x.endswith('W')])
        input_biases_name_list = sorted([x for x in sorted(weight_dict.keys()) if x.endswith('b')])

        graph_weight_name_list = [name for name in var_name_list if 'biases' not in name]
        graph_biases_name_list = [name for name in var_name_list if 'biases' in name]
        
        assert len(graph_weight_name_list) <= len(input_weight_name_list)
        assert len(graph_biases_name_list) <= len(input_biases_name_list)
        
        for i in range(len(graph_weight_name_list)):
            
            inputs_name = input_weight_name_list[i]
            target_name = graph_weight_name_list[i]
            
            with tf.variable_scope('', reuse = True):
                inputs = weight_dict[inputs_name]
                target = tf.get_variable(target_name, dtype = tf.float32)
                self.sess.run(tf.assign(target, inputs))

        for i in range(len(graph_biases_name_list)):
            
            inputs_name = input_biases_name_list[i]
            target_name = graph_biases_name_list[i]
            
            with tf.variable_scope('', reuse = True):
                inputs = weight_dict[inputs_name]
                target = tf.get_variable(target_name, dtype = tf.float32)
                self.sess.run(tf.assign(target, inputs))

    def _strip_consts(self, max_const_size = 32):

        '''
        
        Description: Strip large constant values from 'graph_def'.

        Inputs:

        Output:

        '''
        
        strip_def = tf.GraphDef()
        graph_def = self.graph.as_graph_def()
        
        for n0 in graph_def.node:
            
            n = strip_def.node.add() 
            n.MergeFrom(n0)
            
            if n.op == 'Const':
                
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>"%size
                    
        return strip_def
    
    def show_graph(self, max_const_size = 32):

        '''
        
        Description: Borrowed from
        https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter

        Inputs:

        Output:

        '''

        from IPython.display import display, HTML
            
        strip_def = self._strip_consts(max_const_size = max_const_size)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data = repr(str(strip_def)), id = 'graph' + str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
        display(HTML(iframe))
