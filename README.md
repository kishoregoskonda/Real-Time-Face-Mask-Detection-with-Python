# Real-Time-Face-Mask-Detection-with-Python
In the age of Covid-19, face masks and social distancing have become the new normal. In this article, I will introduce you to a machine learning project on real-time face mask detection with Python.

# Face Mask Detection System

Indoor places, such as restaurants and grocery stores, are legally required to have rules in place for the mandatory use of face masks. Having a worker manually examining each person to make sure their mask is on simply defeats the goal of limiting contact with people as much as possible. So, a real-time face mask detection system can be used to address this issue that will not only maximize efficiency but will also ensure to potentially save lives.

The technology behind the real-time face mask detection system is not new. In Machine Learning, face mask detection is the problem of computer vision. Too often we see computer vision applications of this technology in our daily lives. A common example is a face unlocking in smartphones.

The goal of a face mask detection system is to create an image recognition system that understands how image classification works, and it should work with great accuracy so that our model can be applied in the realtime situations. It will work by recognizing the boundaries of the face and predicting whether or not you are wearing a face mask in real-time

# Real-time Face Mask Detection with Python

Now, I’m going to create a convolutional neural network to create a real-time facial mask detection model with Python. Here, I will use three dense layers in our model with respectively 50, 35 and finally 2 neurons. The dense network produces the probability of the binary classification of no mask = 1 and mask = 0:

# Testing The Model in Real-Time

To test our model in real-time, I’ll be using the VideoCapture function in the OpenCV library in Python. The Cascade classifier, designed by OpenCV, was used to detect the frontal face in live video via detectMultiScale. We can use a while loop to continue capturing images from the webcam.

Our machine learning model will then determine whether or not a face mask is worn in real-time. Based on the performance and accuracy of our model, the result of the binary classifier will be indicated by showing a green rectangle superimposed around the section of the face indicating that the person at the camera is wearing a mask, or a red rectangle indicating that the person on camera is not wearing a mask.
