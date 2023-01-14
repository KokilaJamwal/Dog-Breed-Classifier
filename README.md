## Dog breed classifier

In this project, the task is to classify the breed of different dogs by utilizing the image dataset. The motivation comes from when I tried to find the breed of my indian dog. Then, I thought, Deep learning may help to find the breed of pet dog, as potrayed in the below image.  
<img src="https://github.com/KokilaJamwal/dog-breed-classifier/blob/main/data/mydog.jpeg" width="280" height="280"> 
Claasfier predicted that my dog belongs to `Smooth fox terrier breed.

In this classifier, we used different dataset as well bottleneck features from different sources  
    1. Dog image dataset: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip  
    2. Bottleneck features: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz 
    
How to run this project:
    1. First, save the data from above link in the data/ folder 
    2. Similarly download the Bottleneck feature model in data/ folder 
    3. Implemented two different models for the dog breed classifciation.   
        a.  Custome Model with stacked CNN layers like Convolution, Maxpooling layer. 
            python3 run.py --model_type own 
        b. Transfer learning with botlleneck features downloaded from above link.    
            python3 run.py --model_type transfer 
Results: 
    1. Accuracy of custom model is about 18.12% 
    2. Accuracy with botlleneck features 84.56%
        
    


    


