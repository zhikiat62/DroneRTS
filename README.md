# DroneRTS
This is a Face Mask Detection Project using the drone. 

# System Requirement
```
Python version : 3.6.8
Keras
Tensorflow
Sklearn
```

# Dataset
The datasets can be get from the [kaggle](https://www.kaggle.com/spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask) with the categories of with mask and without mask.
Then, create a directory named 'dataset' and put the folder 'with_mask' and 'without_mask' into the folder.

# Installation Guide
Please create a virtual environment for the python project.
Create the virtual environment library.
```
pip install virtualenv
```
After that, create a virtual environemnt and activate it.
```
virtualenv venv
venv\Scripts\activate.bat
```
After activating the virtual environment, please install the requirement.
```
pip install -r requirement.txt
```
And
```
python billyMain.py
```
You are good to go.

# Reference 
The face mask model is trained using Deep Learning and MobileNetV2 for higher accuracy. For more reference, you can refer to this [link](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/).
