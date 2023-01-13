
from keras.utils import np_utils
from sklearn.datasets import load_files       
import numpy as np
from glob import glob

from keras.preprocessing.image import img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import GlobalAveragePooling2D, Dense  
from keras.utils.np_utils import to_categorical

from keras.preprocessing import image                  
from tqdm import tqdm

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    label = np_utils.to_categorical(np.array(data['target']), 133)# toatal 133 breed of dogs
    return files, label



def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)