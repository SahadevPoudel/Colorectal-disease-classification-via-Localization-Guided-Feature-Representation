# Colorectal-disease-classification-via-Localization-Guided-Feature-Representation
**Official keras implementation of the paper "Colorectal Disease Classification via Localization Guided Feature Representation"**

**Code**

**Install dependencies**

      python -m pip install -r requirements.txt
  
 This code was tested with python 3.6
 
 **Train**
 

To train the model on the datasets, put the desire datasets on the following structure, Datasets / (Training, validation , Testing )

     python main.py --mode train --dataroot /...../Datasets --dataset_type Colorectal --epoch 100 --checkckpt_dir ./saved_model --plot yes --batch_size 16 --lr 0.001

**Test**

To test the trained model on the dataset, load the model from the checkpoint directory and provide the datasets path.

      python main.py --mode test --dataroot /...../Datasets --dataset_type Colorectal --load_model /...../model.h5
      

**Visualize**
To visualize the model in Class Activation Mapping (CAM) on the particular image,then run

      python main.py --mode visualize --load_model /...../model.h5 --image_path /...../adenoma.jpg
