# Dog Breed Classifier
In this project, the task is to classify the breed of different dogs by utilizing the image dataset. The motivation came when I tried to find the breed of my dog. Then, I thought, Deep learning methods, especailly Convolution Neural Network might help to find the breed of my pet dog, as potrayed in the below image.  
<img src="https://github.com/KokilaJamwal/dog-breed-classifier/blob/main/data/mydog.jpeg" width="280" height="280">  

In the result, classifier predicted that my dog belongs to `Smooth fox terrier` breed.

## Used different dataset as well  as bottleneck features from different sources  
    1. Dog image dataset: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip  
    2. Bottleneck features: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz 
## How to run this project: 
    1. First, save the data from above link in the data/ folder 
    2. Similarly download the Bottleneck feature in data/ folder 
    3. Run pip install -r requirements.txt
    3. Implemented two different models for the dog breed classification.   
        a. `Custom model` with stacked CNN layers like Convolution layer, Maxpooling layer, . 
                 python3 run.py --model_type own 
        b. `Transfer learning` with botlleneck features downloaded from above link.    
                 python3 run.py --model_type transfer 
## Results: 
    1. Accuracy of custom model is about `18.12%` 
    2. Accuracy with botlleneck features `84.56%`
        
    


    


