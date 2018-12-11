<img src="https://siaw.unisg.ch/-/media/538460c65a8e4a018d00ee8b97bf8709.jpg" alt="SIAW logo" width="200" align="right">

# Introduction to Tensorflow and Google Cloud Platform



<br>
This repository contains code and other files prepared for a talk on getting started with Tensorflow and deploying a Python app with Flask on GCP for students of the 'Data Science Fundamentals' Course at the University of St. Gallen. 

As requested by Prof. Binswanger, I also presented some of my learnings I gained after having worked in Data Science positions. The Slides linked to contain information on some of my work experience as well as some recommendations for finding a first job for students coming from a non-technical background such as this students of the Data Science Fundamentals course at the University of St. Gallen.

Link to Slides: [*Here*](https://docs.google.com/presentation/d/1J3KZaT9vVsAHCF8O-h_hnPtRjN3arh6xX5-HSJ92_AU/edit?usp=sharing*)
 

# Structure

This introduction is structured into three parts

1. Introduction to Tensorflow 
	* Explanation on what Tensorflow is and who uses it
	* How you would build a simple regression with Tensorflow
	* Writing a basic Image classifier with Keras
2. Deploying a Flask App in the Cloud and making it accessible to third-parties 
	* Products of the Cloud and explaining the industry shift towards it
	* Auto-ML and the ML APIs
	* Deploying a Python script as a Flask App with Google App Engine
3. Starting-points to move towards Data Science with a non-technical background 
	* Resources to get started
	* Getting a first Job in data sciences


## Folders

**Code**: Contains the code which was presented in the lecture (see next section)

**Data**: Contains two sample Images used in the scripts and serves as a folder for any user to store the data for the image classifier. The data has to be downloaded by each user due to the size (see the last section of this read-me for more information on that)

**GCP-App**: This folder contains all the files and scripts which were used to launch the GCP Webapp with the GCP App Engine. In particular the only additional files required are: app.yaml (contains information for the server on the type of app) and requirements.txt (tells the server which python packages it needs to install).

## Code: Overview of the Scripts presented

The course includes 6 notebooks / scripts, which are all described with comments. The Flask app is only available as a script as it cant be run in a Notebook environment. A brief overview of each file is presented here:

**TF-basics**: Basic operations in tensorflow and explains basic concepts such as a tensorflow session.

**Basic-Regression**: An example of what a linear regression would look like using the classic Iris Dataset

**Image-Classifier**: How to create a small image classifier trained on images of cats and dogs and save/load the weights of a network

**GCP-APIs**: Demonstration of the Google Cloud Platform Language API

**Flask-App.py**:  How to create a small Flask app that contains a neural network and deploy it as a webapp

**Basic-Flask-App.py**: Code for a basic Flask App to demonstrate the ease of setting up such a webapp. Run the code in your terminal or in a text-editor like sublime (not in an interactive python kernel like Spyder)


## Downloading the Data for the Image Classification
Script 3 (Image Classifier) uses data not contained in the repository which has to be downloaded. The dataset used is the cats and dogs image dataset taken from Kaggle. Due to its large size it is not contained in the repository. You can download the data from here: 

https://www.kaggle.com/c/dogs-vs-cats


## Questions or further enquiries
If you have any questions about this content or any other enquiries, feel free to send me an email: [dominique.c.a.paul@gmail.com](mailto:dominique.c.a.paul@gmail.com)







