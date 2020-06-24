# Colorectal Disease Classification Using Efficiently Scaled Dilation in Convolutional Neural Network
**Official keras implementation of the paper "Colorectal Disease Classification Using Efficiently Scaled Dilation in Convolutional Neural Network
"**

**Abstract** : 
Computer-aided diagnosis systems developed by computer vision researchers have helped doctors to recognize several endoscopic colorectal diseases more rapidly, which allows appropriate treatment and increases the patient’s survival ratio. Herein, we present a robust architecture for endoscopic image classification using an efficient dilation in Convolutional Neural Network (CNNs). It has a high receptive field of view at the deep layers in increasing and decreasing dilation factor to preserve spatial details. We argue that dimensionality reduction in CNN can cause the loss of spatial information, resulting in miss of polyps and confusion in similar-looking images. Additionally, we use a regularization technique called DropBlock to reduce overfitting and deal with noise and artifacts. We compare and evaluate our method using various metrics: accuracy, recall, precision, and F1-score. Our experiments demonstrate that the proposed method provides the F1-score of 0.93 for Colorectal dataset and F1-score of 0.88 for KVASIR dataset. Experiments show higher accuracy of the proposed method over traditional methods when classifying endoscopic colon diseases.

**Code**

**Install dependencies**

      python -m pip install -r requirements.txt
  
 This code was tested with python 3.6
 
 **Train**
 

To train the model on the datasets, put the desire dataset on the following structure, Dataset / (Training, validation , Testing )

     python main.py --mode train --dataroot /...../Datasets --dataset_type Colorectal --epoch 100 --checkckpt_dir ./saved_model --plot yes --batch_size 16 --lr 0.001

**Test**

To test the trained model on the dataset, load the model from the checkpoint directory and provide the dataset path.

      python main.py --mode test --dataroot /...../Datasets --dataset_type Colorectal --load_model /...../model.h5
      

**Visualize**
To visualize the model in Class Activation Mapping (CAM) on the particular image,then run

      python main.py --mode visualize --load_model /...../model.h5 --image_path /...../adenoma.jpg
      
**Results**

![With the proposed model, The similar-looking images were classified successfully. For example:](https://github.com/SahadevPoudel/Colorectal-disease-classification-via-Localization-Guided-Feature-Representation/blob/master/images/result.png)
