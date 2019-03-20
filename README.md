# Variational-autoencoder
A tensorflow implementation of a variational autoencoder.

## Requirements:
1. tensorflow v1.13.1
2. PIL
3. matplotlib
4. IPython.display (for tensorboard visualization in a jupyter notebook)

## Usage:
Run the enclosed `vae_mnist.ipynb` notebook in a tensorflow environment.

## Tensorflow graph:

From the enclosed `vae_mnist.ipynb` notebook, you can run tensorboard to see the entire computational graph.<br>

A look at the entire tensorflow graph:<br>
![tensorflow graph](other/images/tensorflow_graph.png?raw=true "tensorflow_graph")

A look at the variational autoencoder graph:<br>
![variational autoencoder graph](other/images/variational_autoencoder.png?raw=true "variational_autoencoder")

A look at the encoder graph (three-layer fully-connected):<br>
![encoder graph](other/images/encoder.png?raw=true "encoder")

A look at the decoder graph (three-layer fully-connected):<br>
![decoder graph](other/images/decoder.png?raw=true "decoder")
