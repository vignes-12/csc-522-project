### ResNet50:

#### Preprocessing:

- Original Dataset.ipynb: File to convert the images data from each category into a single .npy file (orig_data.npy)
- Split Dataset.ipynb: File to split the data from orig_data.npy into train and test data
- Center Crop.ipynb: File to center crop the image data present in train.npy or test.npy

#### ResNet50_Trainig:

- ResNet-crop.ipynb: File to train the ResNet50 model from scratch on Caltech-256 dataset
- ResNet-crop-tuned.ipynb: File to tune the ResNet50 model using image augmentation and learning rate scheduler

References: 
- https://github.com/xufanxiong/Classification-of-CALTECH-256
