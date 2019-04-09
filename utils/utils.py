import os
import pickle
import numpy as np
import constants as const

def default(num):

    '''
    
    Description:

    Inputs:

    Output:

    '''

    if isinstance(num, np.int64):
        return int(num)  
    raise TypeError

def load_mnist_data():

    '''
    
    Description:

    Inputs:

    Output:

    '''

    data_dict = {}
    for key in const.LIST_TRAIN_TESTA_TESTB_KEYS:
        data_dict[key] = {}
        path = os.path.join(const.PATH_ROOT, 'mnist/data/mnist-tf-' + key + '.pkl')
        with open(path, 'rb') as file_handle:
            data_tuple = pickle.load(file_handle)
            data_dict[key]['inputs'] = np.stack(data_tuple[0] / 255)
            data_dict[key]['inputs'] = np.expand_dims(np.copy(data_dict[key]['inputs']), axis = 3)
            data_dict[key]['target'] = np.stack(data_tuple[1])

    return data_dict
