# Colorectal-disease-classification-via-Localization-Guided-Feature-Representation
**Official keras implementation of the paper "Colorectal Disease Classification via Localization Guided Feature Representation"**

**Abstract** :\ 
Computer-aided   diagnosis   systems   developed   by
computer  vision  researchers  have  helped  doctors  to  recognize
several  endoscopic  colorectal  images  diseases  more  rapidly.  Nu-
merous  studies  have  focused  on  procedures  based  on  machine
learning and deep learning that maximize the use of endoscopic
colon  images  for  accurate  feature  extraction  and  classification.
However, none of the existing methods analysed the features that
the  system  learned.  In  this  paper,  we  learn  useful  endoscopic
features directly from the raw representations of input data using
Convolutional Neural Networks(CNNs), and gain intuition of the
chosen features based on a Class Activation Mapping (CAM) [1].
As  a  result,  we  find  that  the  use  of  dimensionality  reduction  in
the  CNN  caused  the  loss  of  spatial  details,  resulting  in  miss  of
polyps  and  confound  in  similar-looking  images.  Based  on  these
findings,  we  increased  the  receptive  field  of  view  at  the  deep
layers of the network using dilated convolution. Additionally, we
used  a  regularization  technique  called  DropBlock  [2]  to  reduce
overfitting and deal with noise, and artifacts. We compared and
evaluated  our  method  using  various  metrics:  accuracy,  recall,
precision,   and   F1-score.   Experimental   results   show   that   the
proposed  architecture  outperforms  the  traditional  CNNs  and
existing  methods.

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
