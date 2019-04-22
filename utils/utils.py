import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import constants as const

def default(num):

    '''
    
    Description: converts 'np.int64' datatypes, which are non-json-serializable,
    to 'int' datatype.

    Inputs:
        -'num' (np.int64) = an integer of type 'np.int64'.

    Return:
        -'int(num)' (int) = conversion of 'num' to type 'int'.


    '''

    if isinstance(num, np.int64):
        return int(num)  
    raise TypeError

def load_data(project_name, pickle_name):

    '''
    
    Description: data loader for two different projects:
        -'mnist'
        -'celebA'
    This function is taylored specifically for the data organization of these
    projects and needs to be adapted in order to work with other projects.

    Inputs:
        -'project_name' (str) = the name of the project from which we will load data.
        -'pickle_name' (str) = the name of the pickle file containing the data.

    Return:
        -'data_dict' (data_dict) = for the 'mnist' project, a data dictionary with form
            data_dict[key_1][key_2]
        with key_1 in ['train', 'testa', 'testb'] and key_2 in ['inputs', 'target'].
        OR
        -'images_array' (np.ndarray) = a stacked 4-D array of shape 'num_images' x 64 x 64 x 3.
        -'features_images_dict' (dict) = a dictionary with structure
            {'face features': [list of indices of images having that face feature]}.

    '''

    # Data loading for the 'mnist' project. Expected file structure is
    # root 
    #   |--mnist
    #       |--const.DIR_DATA
    #       |--'pickle_name'-tf-train.pkl
    #       |--'pickle_name'-tf-testa.pkl
    #       |--'pickle_name'-tf-testb.pkl
    if project_name == 'mnist':
        data_dict = {}
        for key in const.LIST_TRAIN_TESTA_TESTB_KEYS:
            data_dict[key] = {}
            path = os.path.join(const.PATH_ROOT, project_name, const.DIR_DATA, pickle_name + '-' + key + '.pkl')
            with open(path, 'rb') as file_handle:
                data_tuple = pickle.load(file_handle)
                data_dict[key]['inputs'] = np.stack(data_tuple[0] / 255)
                data_dict[key]['inputs'] = np.expand_dims(np.copy(data_dict[key]['inputs']), axis = 3)
                data_dict[key]['target'] = np.stack(data_tuple[1])

        return data_dict

    # Data loading for the 'celebA' project. Expected file structure is
    # root 
    #   |--celebA
    #       |--const.DIR_DATA
    #               |--'pickle_name'.pkl
    #               |--'pickle_name'_details.pkl (optional, not used for training)
    elif project_name == 'celebA':

        # Get face image data:
        path = os.path.join(const.PATH_ROOT, project_name, const.DIR_DATA, pickle_name + '.pkl')
        with open(path, 'rb') as file_handle:
            images_array = pickle.load(file_handle)
            images_array = np.true_divide(images_array, 255.0)

        # Looks for face attribute labels too:
        path = os.path.join(const.PATH_ROOT, project_name, const.DIR_DATA, pickle_name + '_details.pkl')
        try:
            with open(path, 'rb') as file_handle:
                features_images_dict = pickle.load(file_handle)
        except:
            features_images_dict = {}

        return images_array, features_images_dict

def find_face(image):

    '''
    
    Description: uses OpenCV to detect and crop the face in an image. Adopted from

    Inputs:
        -'image' (np.ndarray) = an image with one face, generally front-and-center.

    Return:
        -'vertex_tuple_list' (list) = a list of 4-tuples of vertices for the faces
        found in 'image'. In our application, at most one face should be found.

    '''
    
    # NOTE: you may need to change this path name depending on your cv2 installation:
    assert os.path.exists(const.PATH_FACEDATA)
    cascade = cv2.CascadeClassifier(const.PATH_FACEDATA)
    
    # Padding for the frame of the found face.
    pad = 30

    image = cv2.imread(image)

    shape = (image.shape[1], image.shape[0])
    frame = cv2.resize(image, shape)

    # Get x, y, width, and height of face bounding boxes:
    face_tuple_list = cascade.detectMultiScale(frame)

    # Pad each bounding box and get the 'left', 'right',
    # 'top', and 'bottom' coordinate values of the bounding box:
    vertex_tuple_list = []
    for face_tuple in face_tuple_list:

        x, y, w, h = face_tuple
        
        # Pad the bounding box:
        true_left = max(0, x - pad)
        true_rght = min(x + w + pad, image.shape[0] - 1)
        true_lowr = min(y + h + pad, image.shape[1] - 1)
        true_uppr = max(0, y - pad)
        
        # Get the 'left', 'right', 'top', and 'bottom' coordinate values 
        # of the bounding box:
        side_length = min(true_rght - true_left, true_lowr - true_uppr)

        x_delta = int((true_rght - true_left - side_length) / 2)
        y_delta = int((true_lowr - true_uppr - side_length) / 2)
        
        final_left = true_left + x_delta
        final_rght = true_rght - x_delta
        final_lowr = true_lowr - y_delta
        final_uppr = true_uppr + y_delta
        
        # Add to this list of padded bounding boxes:
        vertex_tuple_list += [(final_left, final_uppr, final_rght, final_lowr)]

    return vertex_tuple_list

def crop_image(file_path, vertex_tuple, size):


    '''
    
    Description: crops '.jpg' image found at 'file_path' using vertices from 
    'vertex_tuple', and re-sizes the cropped image to 'size' by 'size'.

    Inputs:
        -'file_path' (str) = path to the '.jpg' image to be cropped.
        -'vertex_tuple' (tuple) = ('left', 'top', 'right', 'bottom') tuple
        for the bounding box used for cropping.
        -'size' (int) = for re-sizing the cropped image to 'size' x 'size'.

    Return:
        -image (np.ndarray) = the cropped imaged, re-sized to 'size' x 'size'.

    '''
    
    image = Image.open(file_path)
    image = image.crop(vertex_tuple)
    image = image.resize((size, size))
    image = np.asarray(image)
    
    return image

def get_cropped_faces(project_name, images_name, pickle_name, attrib_name = None, cutoff = None):

    '''
    
    Description: crops faces from images in the 'img_align_celeba' folder, downloaded from
    "Align&Cropped Images" from here: "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html".
    Optionally, we can also use face feature annotations found under "Attribute Annotations"
    at the same website. We use this for face interpolation experiments later.

    This function creates a pickle file storing the 'cutoff x 64 x 64 x 3' cropped face images
    thus produced. This file can be quite large (for example, over 1GB for 100,000 cropped face images).

    Before calling this function, the file structure is
    root 
      |--'project_name' (should be 'celebA')
          |--const.DIR_DATA
                  |--'images_name' (should be 'img_align_celeba')
                  |--'attrib_name' (optional, should be 'list_attr_celeba.txt').

    After calling this function, the file structure is
    root 
      |--'project_name' (should be 'celebA')
          |--const.DIR_DATA
                  |--'images_name' (should be 'img_align_celeba')
                  |--'attrib_name' (optional, should be 'list_attr_celeba.txt')
                  |--'pickle_name'.pkl (output stacked cropped face images)
                  |--'pickle_name'_details.pkl (optional, only if 'attrib_name' is given).

    Inputs:
        -'project_name' (str) = the name of the project (should be 'celebA').
        -'images_name' (str) = the name of the directory containing the '.jpg' face images
        (should be called 'img_align_celeba').
        -'pickle_name' (str) = the name of the pickle file continaing the 'cutoff x 64 x 64 x 3'
        numpy array of stacked cropped face images produced by iterating over original face
        images in the 'image_name' directory.
        -'attrib_name' (str) = the name of the file containing the face attribute annotations
        (should be a text file called 'list_attr_celeba.txt').
        -'cutoff' (int) = the maximum number of face images to process.

    Return:
        -None

    '''

    # This takes a while to run, so we watch the time:
    start = time.time()

    # Initialize an empty list of cropped face images:
    image_list = []

    # Path to 'const.DIR_DATA' in our project:
    data_path = os.path.join(const.PATH_ROOT, project_name, const.DIR_DATA)

    # File paths to the directory containing the '.jpg' face images and to
    # the destination of the pickle file containing the cropped face images:
    images_path = os.path.join(data_path, images_name)
    pickle_path = os.path.join(data_path, pickle_name + '.pkl')

    # File paths to the original face attributes file 
    # the destination of the pickle file with the dictionary of face attributes:
    if attrib_name:
        attrib_path = os.path.join(data_path, attrib_name)
        detail_path = os.path.join(data_path, pickle_name + '_details.pkl')

    # If the pickled image file already exists and we don't need face attributes, then break:
    if os.path.exists(pickle_path) and not attrib_name:
        return

    # If the pickled image file and face attributes file already exist, then break:
    if os.path.exists(pickle_path) and attrib_name and os.path.exists(detail_path):
        return

    # Import the file with the face attributes. This is taylored to ingest the file
    # 'list_attr_celeba.txt' found at "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html".
    if attrib_name:
        features_df = pd.read_csv(attrib_path, delim_whitespace = True).set_index('202599')
        images_features_dict = features_df.to_dict('index')
        features_images_dict = {}

    # Iterate over each '.jpg' file found in 'images_path':
    for dir_path, dir_name_list, file_name_list in os.walk(images_path):

        # Process each '.jpg' file in file_name_list:
        for counter, file_name in enumerate(file_name_list):
            
            # Skip if not a '.jpg' file:
            if not file_name.endswith('.jpg'):
                continue

            # Stop if we have processed 'cutoff' images already:
            if cutoff and len(image_list) == cutoff:
                break
                
            # Get the list of face bounding boxes in the current image:
            file_path = os.path.join(dir_path, file_name)
            vertices_list = find_face(file_path)
            
            # Skip if no face was found:
            if len(vertices_list) == 0:
                print('Skipping image {} (no face box vertices).'.format(counter + 1))
                continue
                
            # Crop out and re-size the first face to 64 x 64 x 3
            # (only one face should have been found in the image):
            image = crop_image(file_path, vertices_list[0], 64)

            # Skip the image if it's final shape is note 64 x 64 x 3:
            if image.shape != (64, 64, 3):
                print('Skipping image {} (shape not 64 x 64 x 3).'.format(counter + 1))
                continue
                
            # Add the image to the list of cropped face images:
            image_list += [image]

            # Construct the dictionary 'features_images_dict', with the form
            #   {face_feature: [list of images (indexed by 'counter') with face_feature]}:
            if attrib_name and file_name in images_features_dict:
                for feature in images_features_dict[file_name]:
                    if feature not in features_images_dict:
                        features_images_dict[feature] = []
                    if images_features_dict[file_name][feature] == 1:
                        features_images_dict[feature] += [len(image_list) - 1]

            # Print out progress update:
            if (counter + 1) % 1000 == 0:
                print('Processed {} of {} images.'.format(counter + 1, len(file_name_list)))

        # Stack the images into 'cutoff x 64 x 64 x 3' numpy array (do this at the
        # end to prevent slowdown), and pickle:
        image_array = np.stack(image_list, axis = 0)
        with open(pickle_path, 'wb') as file_handle:
            pickle.dump(image_array, file_handle)

        # Pickle the face attributes stored in 'features_images_dict':
        with open(detail_path, 'wb') as file_handle:
            pickle.dump(features_images_dict, file_handle)

        # Done:
        print('Finished processing images. Time elapsed:', time.time() - start)
        
        # Only process '.jpg' files in the top directory.
        break
