## IEEE-Signal-Processing-Cup-2020_Unsupervied-Abnormality-Detection
This research work was based on the proposed problem of Signal Processing Cup(SP Cup) 2020. Here our main goal was to detect abnormalities in the behaviour of the ground and aerial systems based on embedded sensor data in real-time.

## Problem Definition  
<p align = "justify" >
The problem requires us to build a model that can detect abnormalities from unlabeled data collected from embedded sensors placed on a heterogeneous autonomous system that is in motion. The model can be trained only on normal data that is known to contain no abnormality. Once the model has been trained, it should be able to declare abnormality in every timestamp of IMU and camera synchronized data in real time. The model should preferably determine the reason behind the abnormality and also give a continuous score indicating the degree of abnormality.</p>

## Abstract
<p align = "justify">
Abnormality detection in the behaviour of ground and aerial systems is a challenging task specially in an unsupervised way. Embedded sensors such as Inertial Measurement
Unit and digital camera are used to gather information regarding motion of those ground and aerial systems in real time. In this paper, we focus on building an intelligent and heterogenous autonomous system that can detect abnormalities from that information. We have proposed two novel methods for the task, one for the sensor data and the other for the image data. We have used an LSTM Autoencoder for the sensor data and an optical flow based Conv-Autoencoder for the image data along with a mathematical model for the abnormality score. The LSTM model is capable of pinpointing the reason behind abnormality and can also give predictions in real time. Both of our image and sensor models are robust to noise and provide a continuous measure of anomaly score based upon the severity of incidents.</p>

## Proposed Model
<img src = "https://github.com/PrasunDatta/IEEE-Signal-Processing-Cup-2020_Unsupervied-Abnormality-Detection/blob/main/Corresponding%20Images/Proposed%20Model.PNG" align = "center" />

