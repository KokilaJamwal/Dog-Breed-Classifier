from model import *
from model_tranferlearning import *
from utils_all import *
from PIL import ImageFile
import configparser
import os
import numpy as np
import random as rn

# set variable
rn.seed(0)
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ImageFile.LOAD_TRUNCATED_IMAGES = True   

if __name__ == "__main__" :
    # Take parameters as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--model_type", type=str)
    args = parser.parse_args()

    train_X, train_Y = load_dataset('../data/dogImages/train')
    valid_X, valid_Y = load_dataset('../data/dogImages/valid')
    test_X, test_Y = load_dataset('../data/dogImages/test')

    dog_names = [item[20:-1] for item in sorted(glob("../data/dogImages/train/*/"))]

    # print statistics about the dataset
    print('There are %s total dog images.\n' % len(np.hstack([train_X, valid_X, test_X])))
    print('There are %d training dog images.' % len(train_X))
    print('There are %d validation dog images.' % len(valid_X))
    print('There are %d test dog images.'% len(test_X))
    
    train_tensors = paths_to_tensor(train_X).astype('float32')/255 
    valid_tensors = paths_to_tensor(valid_X).astype('float32')/255
    test_tensors = paths_to_tensor(test_X).astype('float32')/255
    
    if args.model_type=="own": 
        breed_prediction = breed_prediction()
        breed_prediction.build_model(train_tensors)
        breed_prediction.train(train_tensors, train_Y,valid_tensors, valid_Y)
        #mypred = buildtrainmodel(breed_prediction,train_tensors, train_Y,valid_tensors, valid_Y)
        accuracy=breed_prediction.test(test_tensors,test_Y)
        print('Test accuracy: %.4f%%' % accuracy)
    elif args.model_type=="transfer":
        breed_prediction = breed_prediction_trasferlearning()
        breed_prediction.build_model()
        breed_prediction.train(train_Y,valid_Y)
        accuracy=breed_prediction.test(test_Y)
        print('Test accuracy: %.4f%%' % accuracy)

        