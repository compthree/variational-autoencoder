import os

# Suppress 'INFO' messages from tensorflow:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import json
import time
import datetime
import numpy as np
import tensorflow as tf
import constants as const
import utils.utils as utils

class VariationalAutoencoder(object):

    # Prevents tensorflow from printing out INFO or WARNING messages:
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # A list of class attributes:
    __slots__ = const.LIST_ATTRIBUTES

    # Class setter:
    def __setattr__(self, attr_name, value):

        '''
        
        Description: class setter that re-instantiates the class if the user
        changes an attribute that fundamentally alters the class instance.
        If the attribute to be reset is not from an allowed list, then an
        exception is raised.

        Inputs:
            -'attr_name' (str) = the name of the attribute to reset.

        Return:
            -'value' (?) = the value to assign to the attribute.

        '''
        
        # We reinitialize the class if any of these attributes are directly changed:
        if attr_name in ['encoder_list',
                        'decoder_list',
                        'inputs_shape_list',
                        'learning_rate']:
            object.__setattr__(self, attr_name, value)
            self.__init__()
            
        # Otherwise, we do not allow the user to (directly) update the attributes:
        else:
            raise NotImplementedError
            
    # Class deleter:
    def __delattr__(self, attr_name):

        '''
        
        Description: class deleter, currently not implemented. As such,
        it is presently not possible to delete attributes from this class.

        Inputs:
            -'attr_name' (str) = the name of the attribute to delete.

        Return:
            -None

        '''
        
        # We don't allow (direct) deletion of attributes:
        if attr_name in []:
            object.__delattr__(self, attr_name)
            
        else:
            raise NotImplementedError

    # Class initializer:
    def __init__(self, model_details_dict):

        '''
        
        Description: initializer for the 'VariationalAutoencoder' class.

        Inputs:
            -'model_details_dict' (dict) = input dictionary containing the architecture
            of the variational autoencoder. We don't give a strict account of the structure
            that this dictionary must have. Rather, we defer to the examples at the bottom
            'constants.py'.

        Return:
            -None

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

        # By now, we must have the following attributes:
        for attr_name in ['encoder_list', 'decoder_list', 'inputs_shape_list']:
            assert hasattr(self, attr_name)

        # Set default values to these class attributes if missing:
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
        if not self.use_batch_normalization and hasattr(self, 'averaging_axes_length'):
            object.__delattr__(self, 'averaging_axes_length')
        if self.loss_type == 'perceptual' and not hasattr(self, 'percept_list'):
           object.__setattr__(self, 'percept_list', const.PERCEPT_LIST)
        if not self.loss_type == 'perceptual' and hasattr(self, 'percept_list'):
            object.__delattr__(self, 'percept_list')

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

    # Define the 'model_details_dict' class attribute, for export to a '.json' file:
    @property
    def model_details_dict(self):

        model_details_dict = {}

        for attr_name in const.LIST_MODEL_DETAILS_REQUIRED_KEYS + const.LIST_MODEL_DETAILS_OPTIONAL_KEYS:

            if hasattr(self, attr_name):
                model_details_dict[attr_name] = object.__getattribute__(self, attr_name)

        return model_details_dict

    # Enter a new context to start a tensorflow session:
    def __enter__(self):

        '''
        
        Description: context launcher that instantiates a tensorflow session,
        injects a simple input tensor into the graph to determine the shapes
        of the various layers and set them as class attributes, and loads weights
        from checkpoint files, if they exist, into the graph.

        Inputs:
            -None

        Return:
            -'self'

        '''
        
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

        '''
        
        Description: exits the context by closing the tensorflow session
        and resetting the default graph.

        Inputs: (these are standard)
            -'exc_type' (?) =
            -'exc_value' (?) =
            -'traceback' (?) =

        Return:
            -None

        '''
        
        # Close the session:
        self.sess.close()
        
        # Reset the graph:
        tf.reset_default_graph()
        
    # Create the variational autoencoder network:
    def _create_network(self):

        '''
        
        Description: build the graph for the variational autoencoder network
        (encoder and decoder).

        Inputs:
            -None

        Return:
            -None

        '''
        
        # Create the encoder:
        with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
            # Center inputs to [-0.5, 0.5] for fan-in weight initialization:
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

            # In a variational autoencoder, latent variables are random:
            if self.is_variational:

                # Randomly draw from the latent variable distribution:
                latent_sample = tf.random_normal(tf.shape(self.latent_means),
                                                 mean = self.latent_means,
                                                 stddev = self.latent_stdvs,
                                                 name = 'latent_sample',
                                                 dtype = tf.float32)

            # In an autoencoder, latent variables are deterministic:
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
        
        Description: builds the graph for a subnetwork, either the encoder or the decoder.

        Inputs:
            -'inputs' (tf.Tensor) = input tensor into the subnetwork.
            -'layer_list' (list) = list of dictionaries, with the i:th dictionary
            specifying the structure of the i:th layer. See examples at the bottom
            of 'constants.py' for valid structures for these dictionaries.

        Return:
            -'output_means' (tf.Tensor) = tensor of means output from the last layer.
            -'output_lstd2' (tf.Tensor) = tensor of log-variances output from the last layer.
            -'output_stdvs' (tf.Tensor) = tensor of standard deviations derived from 'output_lstd2'.

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
        
        Description: switch for creating a single layer for a subnetwork (either encoder
        or decoder) based on what is specified by 'layer_dict'.


        Inputs:
            -'inputs' (tf.Tensor) = input tensor to the layer.
            -'layer_dict' (dict) = dictionary specifying the structure of the layer. See 
            examples at the bottom of 'constants.py' for valid structures for these dictionaries.
            -'tag' (str) = tag to use in names of variables appearing in this layer.

        Return:
            -'output' (tf.Tensor) = the output from this layer.

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

            # Resize layer (for images only):
            elif layer_type == 're-size':
                output = self._create_re_size_layer(inputs, layer_dict, tag)

            # We did not specify a valid layer type:
            else:
                exception = 'Layer type {} must be, but is not one of, "convolu", \
                            "deconvo", "full_cn", "reshape", or "re-size".'.format(layer_type)
                raise Exception(exception)

        return output        

    def _create_full_cn_layer(self, inputs, layer_dict, tag):

        '''

        Description: creates a single fully-connected layer:
            1. weight multiplication
            2. add biases
            3. activation
        Batch-normalization is optional.

        Inputs:
            -'inputs' (tf.Tensor) = input tensor to the layer.
            -'layer_dict' (dict) = dictionary specifying the structure of the layer. See 
            examples at the bottom of 'constants.py' for a valid structure for this dictionary.
            -'tag' (str) = tag to use in names of variables appearing in this layer.

        Return:
            -'output' (tf.Tensor) = the output from this layer.

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
            weight_init, biases_init = self._get_fan_in_init(num_inputs,
                                                             weight_shape,
                                                             biases_shape)

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
        Batch-normalization is optional.

        Inputs:
            -'inputs' (tf.Tensor) = input tensor to the layer.
            -'layer_dict' (dict) = dictionary specifying the structure of the layer. See 
            examples at the bottom of 'constants.py' for a valid structure for this dictionary.
            -'tag' (str) = tag to use in names of variables appearing in this layer.

        Return:
            -'output' (tf.Tensor) = the output from this layer.

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
            kernel_init, biases_init = self._get_fan_in_init(num_inputs,
                                                             kernel_shape,
                                                             biases_shape)

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
        Batch-normalization is optional.

        Inputs:
            -'inputs' (tf.Tensor) = input tensor to the layer.
            -'layer_dict' (dict) = dictionary specifying the structure of the layer. See 
            examples at the bottom of 'constants.py' for a valid structure for this dictionary.
            -'tag' (str) = tag to use in names of variables appearing in this layer.

        Return:
            -'output' (tf.Tensor) = the output from this layer.

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
            kernel_init, biases_init = self._get_fan_in_init(num_inputs,
                                                             kernel_shape,
                                                             biases_shape)
        
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
        https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf.
        Adapted from code posted at "https://github.com/tensorflow/tensorflow/issues/2169".

        Inputs:
            -'inputs' (tf.Tensor) = a tensor of shape [batch, d_0, d_1, ..., d_n, channels]
            -'window_shape' (list) = a list of positive integers [m_0, m_1, ..., m_n].

        Return:
            -'output' (tf.Tensor) = a tensor of shape [batch, d0 * m_0, d1 * m_1, ..., d_n * m_n, channels].

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

        Description: reshapes the tensor to whatever shape is specified in 'layer_dict'.

        Inputs:
            -'inputs' (tf.Tensor) = input tensor to the layer.
            -'layer_dict' (dict) = dictionary specifying how to reshape 'inputs'. See 
            examples at the bottom of 'constants.py' for a valid structure for this dictionary.
            -'tag' (str) = tag to use in names of variables appearing in this layer.

        Return:
            -'output' (tf.Tensor) = the output from this layer.

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
        
        Description: re-sizes the tensor to whatever size is specified in 'layer_dict'.
        Used only for image data.

        Inputs:
            -'inputs' (tf.Tensor) = input tensor to the layer.
            -'layer_dict' (dict) = dictionary specifying how to re-size 'inputs'. See 
            examples at the bottom of 'constants.py' for a valid structure for this dictionary.
            -'tag' (str) = tag to use in names of variables appearing in this layer.

        Return:
            -'output' (tf.Tensor) = the output from this layer.

        '''

        # 'tf.image.resize_images' requires its input tensor to be of rank 3 or 4:
        self.check_tensor_rank(inputs, [3, 4])

        output = tf.image.resize_images(inputs, layer_dict['output_shape'])

        return output
    
    def _get_activation(self, layer_dict):

        '''
        
        Description: returns the activation function specified for use by 'layer_dict'.

        Inputs:
            -'layer_dict' (dict) = dictionary specifying the structure of the layer. See 
            examples at the bottom of 'constants.py' for a valid structure for this dictionary.

        Return:
            -'activation' (tf function) = tensorflow activation function.

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

    def _get_fan_in_init(self, num_inputs, weight_shape, biases_shape):

        '''
        
        Description: uses the He et al. 'fan-in' method for weight and bias initialization.

        Inputs:
            -'num_inputs' (tf.Tensor) = the number of input units to each node of the
            present layer.
            -'weight_shape' (list) = the shape of the weight tensor used in the present layer.
            -'biases_shape' (list) = the shape of the biases tensor used in the present layer.

        Return:
            -'weight_init' (tf.Tensor) = tensor of normal-distributed initialzed values for 'weight' tensor.
            -'biases_init' (tf.Tensor) = tensor of normal-distributed initialzed values for 'biases' tensor.

        '''

        # The essence of He initialization is to use the following standard deviation:
        num_inputs = tf.cast(num_inputs, dtype = tf.float32, name = 'number_of_inputs')
        stddev = tf.sqrt(2 / num_inputs, name = 'fan_in_stddev')
        weight_init = tf.truncated_normal(weight_shape, stddev = stddev, dtype = tf.float32)
        biases_init = tf.truncated_normal(biases_shape, stddev = stddev, dtype = tf.float32)

        return weight_init, biases_init

    def _perform_batch_normalization(self,
                                     inputs,
                                     train_decay = 0.99,
                                     averaging_axes_length = 'long'):

        '''
        
        Description: inserts a batch-normalization layer.
        Adopted from https://r2rt.com/implementing-batch-normalization-in-tensorflow.html.

        Inputs:
            -'inputs' (tf.Tensor) = input tensor to the layer.
            -'train_decay' (float) = weight used for adjusting the population mean
            estimate with each iteration by moving average.
            -'averaging_axes_length' (str) = may be either of
                -'long'  --> average over all spatial dimensions of 'inputs'.
                -'short' --> average over each spatial dimension of 'inputs' separately.

        Return:
            -'output' (tf.Tensor) = the output from this layer.

        '''

        # Prevent overflow from dividing by a batch variance of zero:
        epsilon = 1e-3

        # Two possible ways to average over datapoint features in a batch:
        if averaging_axes_length == 'long':

            # Average over all but the last 'channel' dimension of 'inputs':
            cutoff = len(inputs.get_shape().as_list()) - 1

        else:

            # Average over only the 'batch' dimension of 'inputs':
            assert averaging_axes_length == 'short'
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

        # Set the 'decay' factor. This factor must be one during inference so we do not change 
        # the estimates of the population mean and std2 for use in the batch normalization sub-layer:
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

        Return:

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

        # If we are using an autoencoder, then we only have the decoder loss:
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
        overall_loss = tf.add(wgt_enc_loss, wgt_dec_loss, name = 'combined_loss')
        
        # Set the loss as a class attribute:
        object.__setattr__(self, 'encoder_loss', encoder_loss)
        object.__setattr__(self, 'decoder_loss', decoder_loss)
        object.__setattr__(self, 'loss',         overall_loss)

    def _get_tensor_value(self, output, feed_dict):

        '''
        
        Description: evaluates the 'output' tensor using input data from 'feed_dict'.
            
        Inputs:
            -'output' (tf.Tensor) = the 'output' tensor to evaluate.
            -'feed_dict' (dict) = input data used to evaluate 'output'.

        Return:
            -'output' (np.ndarray) = numerical values of the evaluted 'output' tensor.

        '''
        
        return self.sess.run(output, feed_dict = feed_dict)
    
    def update_network(self, input_data):

        '''
        
        Description: run the optimizer and evaluate the loss with mini-batch 'input_data'.

        Inputs:
            -'input_data' (np.ndarray) = the mini-batch input data used for the
            optimization step.

        Return:
            -loss (float) = the value of the loss, evaluated over mini-batch 'input_data'
            after optimization.

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
        
        Description: train the model using the 'input_data' as training data.
        Training automatically picks up from whatever is recorded for
        'num_training_epochs' in 'model_details.json', using weights loaded
        from the checkpoint files.

        Inputs:
            -'input_data' (np.ndarray) = the training data.
            -'batch_size' (int) = the number of datapoints in each mini-batch.
            -'display_step' (int) = the skip step for training progress printouts.
            -'num_training_epochs' (int) = the number of training epochs.

        Return:
            -None

        '''
        
        # Compute the number of mini-batches per epoch:
        num_samples = len(input_data)
        num_batches_per_epoch = int(num_samples / batch_size)
    
        # For printing out progress updates:
        print_start = time.time()

        # Begin training from where we left off (initialized to 0 or
        # pulled from 'model_details.json' from a previous training session).
        epoch_begin = self.num_trained_epochs

        # Begin training:
        for epoch in range(epoch_begin, num_training_epochs):

            # For printing out progress updates:
            epoch_start = time.time()

            # Initialize the average loss over mini-batches for this epoch to 0:
            avg_loss = 0.
            
            # Loop over all batches
            for i in range(num_batches_per_epoch):

                # Get the data mini-batch:
                data_batch = input_data[i * batch_size: (i + 1) * batch_size]

                # Fit training using mini-batch data:
                loss = self.update_network(data_batch)

                # Update the average loss
                avg_loss += loss * (batch_size / num_samples)

            # Set 'avg_loss' and 'num_trained_epochs' as class attributes:
            object.__setattr__(self, 'avg_loss', avg_loss)
            object.__setattr__(self, 'num_trained_epochs', self.num_trained_epochs + 1)

            # Print out progress and save a model checkpoint:
            if (epoch + 1) % display_step == 0:
                
                # Print out progress:
                delta = round(time.time() - print_start, 3)
                print_out = 'Epoch {} of {} completed. Total time elapsed so far: {}'
                print_out = print_out.format(epoch + 1, num_training_epochs, delta)
                print('\n')
                print(print_out)
                print('-' * len(print_out))
                print('Loss over all input data:', round(self.avg_loss, 3))
                print('Time to train last epoch:', round(time.time() - epoch_start, 3))

                # Save the model weights in checkpoint files:
                self.save_model()
    
    def encode(self, input_data):

        '''
        
        Description: evaluate the tensors 'self.latent_means' and 'self.latent_stdvs' using
        the values of 'input_data'.

        Inputs:
            -'input_data' (np.ndarray) = input data values.

        Return:
            -'latent_means' (np.ndarray) = the means output from the encoder network.
            -'latent_stdvs' (np.ndarray) = the standard deviations output from the encoder network.

        '''
        
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self.latent_means, feed_dict), self._get_tensor_value(self.latent_stdvs, feed_dict)
        
    def decode(self, latent_sample = None):

        '''
        
        Description: evaluate the tensors 'self.output_means' using the values of 'latent_sample', 
        the latent variable values generated from random sampling.

        Inputs:
            -'latent_sample' (np.ndarray) = latent variable input values.

        Return:
            -'output_means' (np.ndarray) = the means output from the decoder network.

        '''
        
        # If no sample value for the latent variables is provided, we generate it randomly:
        if latent_sample is None:
            latent_sample = np.random.normal(size = [1] + list(self.get_latent_shape()))

        feed_dict = {self.latent_sample: latent_sample}
        return self._get_tensor_value(self.output_means, feed_dict)
    
    def reform(self, input_data):

        '''
        
        Description: evaluate the tensors 'self.output_sample' using the values of 'input_data'.
        In other words, this reconstructs 'input_data' by passing it through the whole network.

        Inputs:
            -'input_data' (np.ndarray) = input data values.

        Return:
            -'output_means' (np.ndarray) = the means output from the decoder network.

        '''

        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self.output_sample, feed_dict)

    def check_tensor_rank(self, tensor, rank_list):

        '''
        
        Description: checks the rank of 'tensor' is among those in 'rank_list'.

        Inputs:
            -tensor (tf.Tensor) = a tensorflow tensor, whose rank is to be checked.
            -rank_list (list) = a list of acceptable ranks for 'tensor'.

        Return:
            -None

        '''

        assert len(tensor.get_shape().as_list()) in rank_list
        
    def get_inputs_shape(self, input_data = None):

        '''
        
        Description: gets the shape of 'input_data'. (This method is used
        only for setting this shape as a class attribute.)

        Inputs:
            -'input_data' (np.ndarray) = the input data whose shape is to be
            determined.

        Return:
            -(np.ndarray) = the shape of 'input_data'.

        '''

        # If no input data is provided, then feed the graph with an array
        # of zeros to get the shape of 'input_data', and ignore the 'batch' dimension:
        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return self._get_tensor_value(self._inputs_shape, feed_dict)[1:]
        
        # Otherwise, feed the graph to determine the shape of 'input_data':
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._inputs_shape, feed_dict)
    
    def get_latent_shape(self, input_data = None):

        '''
        
        Description: gets the shape of tensors in the latent space output by the encoder
        by feeding 'input_data' into the graph.

        Inputs:
            -'input_data' (np.ndarray) = the input data used to determine this shape.

        Return:
            -(np.ndarray) = the shape of tensors in the latent space.

        '''

        # If no input data is provided, then feed the graph with an array
        # of zeros to get the shape, and ignore the 'batch' dimension:
        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return self._get_tensor_value(self._latent_shape, feed_dict)[1:]
        
        # Otherwise, feed the input data into the graph to determine the shape:
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._latent_shape, feed_dict)
    
    def get_output_shape(self, input_data = None):

        '''
        
        Description: gets the shape of tensors in the output space of the decoder
        by feeding 'input_data' into the graph.

        Inputs:
            -'input_data' (np.ndarray) = the input data used to determine this shape.

        Return:
            -(np.ndarray) = the shape of tensors in the output space of the decoder.

        '''

        # If no input data is provided, then feed the graph with an array
        # of zeros to get the shape, and ignore the 'batch' dimension:
        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return self._get_tensor_value(self._output_shape, feed_dict)[1:]
        
        # Otherwise, feed the input data into the graph to determine the shape:
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._output_shape, feed_dict)
    
    def get_layer_shape_list(self, input_data = None):

        '''
        
        Description: get the shapes of all layers in the network (encoder and decoder)
        by feeding 'input_data' into the graph.

        Inputs:
        -'input_data' (np.ndarray) = the input data used to determine the shapes.

        Return:
        -(list) = the list of all layer shapes, in the order that they are stacked
        in the network.

        '''

        # If no input data is provided, then feed the graph with an array
        # of zeros to get the layer shapes, and ignore the 'batch' dimension:
        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return [x[1:] for x in self._get_tensor_value(self._layer_shape_list, feed_dict)]

        # Otherwise, feed the input data into the graph to determine the layer shapes:
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._layer_shape_list, feed_dict)

    def get_encoder_shape_list(self, input_data = None):

        '''
        
        Description: get the shapes of all layers in the encoder subnetwork
        by feeding 'input_data' into the graph.

        Inputs:
        -'input_data' (np.ndarray) = the input data used to determine the shapes.

        Return:
        -(list) = the list of all layer shapes, in the order that they are stacked
        in the encoder subnetwork.

        '''

        # If no input data is provided, then feed the graph with an array
        # of zeros to get the encoder layer shapes, and ignore the 'batch' dimension:
        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return [x[1:] for x in self._get_tensor_value(self._encoder_shape_list, feed_dict)]
        
        # Otherwise, feed the input data into the graph to determine the encoder layer shapes:
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._encoder_shape_list, feed_dict)
        
    def get_decoder_shape_list(self, input_data = None):

        '''
        
        Description: get the shapes of all layers in the decoder subnetwork
        by feeding 'input_data' into the graph.

        Inputs:
        -'input_data' (np.ndarray) = the input data used to determine the shapes.

        Return:
        -(list) = the list of all layer shapes, in the order that they are stacked
        in the decoder subnetwork.

        '''

        # If no input data is provided, then feed the graph with an array
        # of zeros to get the decoder layer shapes, and ignore the 'batch' dimension:
        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return [x[1:] for x in self._get_tensor_value(self._decoder_shape_list, feed_dict)]
        
        # Otherwise, feed the input data into the graph to determine the decoder layer shapes:
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._decoder_shape_list, feed_dict)

    def get_percept_shape_list(self, input_data = None):

        '''
        
        Description: if using perceptual loss ("self.loss == 'perceptual'"), get
        the shapes of all layers in the perceptual subnetwork by feeding 'input_data'
        into the graph.

        Inputs:
        -'input_data' (np.ndarray) = the input data used to determine the shapes.

        Return:
        -(list) = the list of all layer shapes, in the order that they are stacked
        in the perceptual subnetwork.

        '''

        # Skip if there is no perceptual subnetwork:
        if not hasattr(self, '_percept_shape_list'):
            return []

        # If no input data is provided, then feed the graph with an array
        # of zeros to get the perceptual layer shapes, and ignore the 'batch' dimension:
        if input_data is None:

            feed_dict = {self.inputs: np.zeros([1] + self.inputs_shape_list)}
            return [x[1:] for x in self._get_tensor_value(self._percept_shape_list, feed_dict)]
        
        # Otherwise, feed the input data into the graph to determine the perceptual layer shapes:
        feed_dict = {self.inputs: input_data}
        return self._get_tensor_value(self._percept_shape_list, feed_dict)

    def set_project_path(self):

        '''
        
        Description: set the path to the project as a class attribute.

        Inputs:
            -None

        Return:
            -None

        '''

        project_path = os.path.join(const.PATH_ROOT, self.project_name)
        object.__setattr__(self, 'project_path', project_path)

    def set_models_path(self):

        '''
        
        Description: set the path to the 'models' directory for this project
        as a class attribute.

        Inputs:
            -None

        Return:
            -None

        '''

        self.set_project_path()
        models_path = os.path.join(self.project_path, const.DIR_MODELS)
        object.__setattr__(self, 'models_path', models_path)

    def set_model_path(self):

        '''
        
        Description: set the path to the model used by this instance of
        'VariationalAutoencoder'. (The name of the model is given by
        'model_details_dict[model_name]' during class instantiation.)

        Inputs:
            -None

        Return:
            -None

        '''

        self.set_models_path()
        model_path = os.path.join(self.models_path, self.model_name)
        object.__setattr__(self, 'model_path', model_path)
        saved_model_path = os.path.join(self.model_path, const.DIR_SAVED_MODEL)
        object.__setattr__(self, 'saved_model_path', saved_model_path)

    def set_model_details_path(self):

        '''
        
        Description: set the path to 'model_details.json', which stores all
        structure and training details of the model used by this instance of
        'VariationalAutoencoder'.

        Inputs:
            -None

        Return:
            -None

        '''

        self.set_model_path()
        model_details_path = os.path.join(self.model_path, const.FILE_MODEL_DETAILS)
        object.__setattr__(self, 'model_details_path', model_details_path)

    def make_project_path(self):

        '''
        
        Description: creates the project directory if it does not exist. 

        Inputs:
            -None

        Return:
            -None

        '''

        self.set_project_path()
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)

    def make_models_path(self):

        '''
        
        Description: creates the 'models' directory for this project if
        it does not exist. 

        Inputs:
            -None

        Return:
            -None

        '''

        self.set_models_path()
        if not os.path.exists(self.models_path):
            self.make_project_path()
            os.mkdir(self.models_path)

    def make_model_path(self):

        '''
        
        Description: creates a directory for the present model if it does
        not exist. This directory is timestamped to avoid conflicts.

        Inputs:
            -None

        Return:
            -None

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
        
        Description: checks that 'self.model_details_dict' has an acceptable
        format. Examples of acceptable formats are found at the bottom of
        'constants.py'.

        Inputs:
            -None

        Return:
            -None

        '''

        # We need to modify this dictionary slightly before performing
        # checks, so we create a copy:
        model_details_dict = dict(self.model_details_dict)

        # In 'self.model_details_dict', if the i:th key of 'key_list_1' has
        # the i:th value in 'value_list', then 'self.model_details_dict' must
        # contain the i:th key in 'key_list_2'. Otherwise, 'self.model_details_dict'
        # must not contain the i:th key in 'key_list_2'.
        key_list_1 = ['use_batch_normalization', 'loss_type']
        key_list_2 = ['averaging_axes_length', 'percept_list']
        value_list = [True, 'perceptual']
        for key_1, key_2, value in zip(key_list_1, key_list_2, value_list):
            if model_details_dict[key_1] == value:
                if key_2 not in model_details_dict:
                    exception = 'model_details_dict has the key-value pair "' + str(key_1) + ': ' + str(value) + '". ' \
                    + 'It must therefore contain the key "' + str(key_2) + '", but does not.'
                    raise Exception(exception)
                model_details_dict.pop(key_2)
            else:
                if key_2 in model_details_dict:
                    exception = 'model_details_dict does not have the key-value pair "' + str(key_1) + ': ' + str(value) + '". ' \
                    + 'It must therefore not contain the key "' + str(key_2) + '", but does.'
                    raise Exception(exception)

        # Check that 'self.model_details_dict' contains all of the required keys:
        if set(model_details_dict) != set(const.LIST_MODEL_DETAILS_REQUIRED_KEYS):

            flag = False

            exception = 'The following keys should not be present in "model_details_dict":\n'
            for key in model_details_dict:
                if key not in const.LIST_MODEL_DETAILS_REQUIRED_KEYS:
                    exception += '\t\t-' + key + '\n'
                    flag = True
            if flag:
                raise Exception(exception)

            exception = 'The following keys must be present in "model_details_dict" but are missing:\n'
            for key in const.LIST_MODEL_DETAILS_REQUIRED_KEYS:
                if key not in model_details_dict:
                    exception += '\t\t-' + key + '\n'
                    flag = True
            if flag:
                raise Exception(exception)

            print(model_details_dict)
            raise Exception('Something is incorrect about the structure of model_details_dict. See above.')


        # TO DO: use schema package to check that the value for each key of
        # 'model_details_dict' is the correct datetype and has the correct format.

        # Also, check that the decoder means layer is an int between 0 and 1:

        return True

    def load_model_details(self):

        '''
        
        Description: loads 'model_details_dict' from 'model_details.json' if
        a model by the name given at the end of 'self.model_path' exists. Raises
        and exception if values in the loaded 'model_details_dict' conflict with
        values already found in 'self.model_details_dict.'

        Inputs:
            -None

        Return:
            -None

        '''

        with open(self.model_details_path) as file_handle:

            # Load the saved model details:
            loaded_model_details_dict = json.load(file_handle)

            # For each key in both 'loaded_model_details_dict' and
            # 'self.model_details_dict', check that their corresponding
            # values match:
            for key in loaded_model_details_dict:

                # Skip this key if it is not in both dictionaries:
                if key not in self.model_details_dict:
                    continue

                # Ignore the number of trained epochs. We'll use the value stored
                # in 'loaded_model_details_dict' when we resume training.
                if key == 'num_trained_epochs':
                    continue

                # Raise an exception if the value of this key does not match between dictionaries:
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

            # Set each key, value pair in 'loaded_model_details_dict' as a class attribute:
            for key in loaded_model_details_dict:
                object.__setattr__(self, key, loaded_model_details_dict[key])
            self.check_model_details_dict_format()

    def save_model_details(self):

        '''
        
        Description: save 'self.model_details_dict' in 'model_details.json'.

        Inputs:
            -None

        Return:
            -None

        '''

        # Check that the format of 'self.model_details_dict' is correct:
        self.check_model_details_dict_format()

        # Save 'self.model_details_dict' in 'model_details.json':
        with open(self.model_details_path, 'w+') as file_handle:
            json.dump(self.model_details_dict, file_handle, default = utils.default)

    def load_model(self):

        '''
        
        Description: load weights and checkpoint files from a previously trained model.

        Inputs:
            -None

        Return:
            -None

        '''

        # Load the weights and checkpoint files:
        assert os.path.exists(self.model_path)
        save_model_path = os.path.join(self.model_path, self.model_path.split('/')[-1])
        self.saver.restore(self.sess, save_model_path)

    def save_model(self):

        '''
        
        Description: save weights and checkpoint files from the presently trained model.

        Inputs:
            -None

        Return:
            -None

        '''

        # Save the weights and checkpoint files only:
        assert os.path.exists(self.model_path)
        save_model_path = os.path.join(self.model_path, self.model_path.split('/')[-1])
        self.saver.save(self.sess, save_model_path)
        self.save_model_details()

    def load_initial_loss_weights(self, weight_dict):

        '''
        
        Description: if "self.loss_type == 'perceptual'", then loads weights from the
        pretrained 'vgg16' convolution neural network for the perceptual loss function,
        and assign these values to the weights of the perceptual loss subnetwork.
        See 'const.PERCEPT_LIST' to learn the structure of the part of vgg16 that we
        use here. Weights can be found here: "https://www.cs.toronto.edu/~frossard/post/vgg16/".
        Note: the code written here is taylored to work with the weights found and the
        above website. It will need to be retooled to load weights from files with different
        formatting.

        Inputs:
            -'weight_dict' (dict) = dictionary of layer weights and biases. Obtained
            for example by doing 'weights = np.load("path/to/weights.npz")'.

        Return:
            -None

        '''

        # Don't load the weights if we are not using perceptual loss:
        if self.loss_type == 'pixel':
            pass

        # Get a list of all variable names in the decoder part of the loss function:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'loss_function/decoder_loss')
        var_name_list = [''.join(var.name.split(':0')[:-1]) for var in var_list]

        # Get a list of all variable names in the vgg16 network ('W' --> 'weights', 'b' --> 'biases').
        input_weight_name_list = sorted([x for x in sorted(weight_dict.keys()) if x.endswith('W')])
        input_biases_name_list = sorted([x for x in sorted(weight_dict.keys()) if x.endswith('b')])

        # Get a list of all names of weights and biases in the decoder part of the loss function:
        graph_weight_name_list = [name for name in var_name_list if 'biases' not in name]
        graph_biases_name_list = [name for name in var_name_list if 'biases' in name]
        
        # We are using a subset of the weights and biases that came with vgg16:
        assert len(graph_weight_name_list) <= len(input_weight_name_list)
        assert len(graph_biases_name_list) <= len(input_biases_name_list)
        
        # With the above sorting, the elements of 'graph_weight_name_list' and
        # 'input_weight_name_list' are ordered to match each other:
        for i in range(len(graph_weight_name_list)):
            
            inputs_name = input_weight_name_list[i] # Contains value to be assigned.
            target_name = graph_weight_name_list[i] # Contains weight name to be assigned value.
            
            # Make the value assignment to each weight in the perceptual loss subnetwork:
            with tf.variable_scope('', reuse = True):
                inputs = weight_dict[inputs_name]
                target = tf.get_variable(target_name, dtype = tf.float32)
                self.sess.run(tf.assign(target, inputs))

        # With the above sorting, the elements of 'graph_biases_name_list' and
        # 'input_biases_name_list' are ordered to match each other:
        for i in range(len(graph_biases_name_list)):
            
            inputs_name = input_biases_name_list[i] # Contains value to be assigned.
            target_name = graph_biases_name_list[i] # Contains bias name to be assigned value.
            
            # Make the value assignment to each bias in the perceptual loss subnetwork:
            with tf.variable_scope('', reuse = True):
                inputs = weight_dict[inputs_name]
                target = tf.get_variable(target_name, dtype = tf.float32)
                self.sess.run(tf.assign(target, inputs))

    def _strip_consts(self, max_const_size = 32):

        '''
        
        Description: strip large constant values from 'graph_def'. Borrowed from
        "https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter".

        Inputs:
            -'max_const_size': threshold size.

        Return:
            -'strip_def'...

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
        
        Description: display 'self.graph' in a Jupyter notebook via tensorboard. Borrowed from
        "https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter".

        Inputs:
            -'max_const_size': threshold size, for use in '_strip_consts'.

        Return:
            -None

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
