<img src="https://siaw.unisg.ch/-/media/538460c65a8e4a018d00ee8b97bf8709.jpg" alt="SIAW logo" width="200" align="right">

# Intro-to-TF-and-GCP

<br>
Material for a talk on getting started with Tensorflow and deploying a Python app with Flask on GCP for students of the 'Data Science Fundamentals' Course at the University of St. Gallen

Link to Slides: [*No link available yet* ]

--- 

# Structure

This introduction is structured into three parts

1. Introduction to Tensorflow *(30 minutes)*
	* Explanation on what Tensorflow is and who uses it
	* How you would build a simple regression with Tensorflow
	* Simplifying the process of creating a network with Keras
	* Writing a basic Image classifier 
2. Deploying a Flask App in the Cloud and making it accessible to third-parties *(20 minutes)*
	* Products of the Cloud and explaining the industry shift towards it
	* Auto-ML and the ML APIs
	* Deploying a Python script as a Flask App with Google App Engine
3. Starting-points to move towards Data Science with a non-technical background *(5 minutes)
	* Resources to get started
	* Getting a first Job in data sciences

	
### Optional parts which could be added if the time suffices:
* Evaluating and visualising TF models with Tensorboard

### Possible cutbacks:
In case this initial plan is overly ambitious on the amount of material to discuss, a likely reduction of scope could look like this:

* Running the Flask app only locally (instead of on GCP) to at least demonstrate what it would look like. A reference to a tutorial on how to deploy it on GCP would be added for interested students
* Skip the linear regression part of Tensorflow and directly use Keras to explain how to build a network.



## Overview of the Scripts presented

The course includes 6 short scripts which will be discussed. Currently these are scripts but they will probably be replaced by Notebooks to make them easier to read. The scripts are:

**TF-basic.py**: Basic operations in tensorflow and explains basic concepts such as a tensorflow session.

**WDBC.py**: How you would run a basic regression with Tensorflow

**WDBC-keras.py**: Replacing tensorflow code with Keras, which allows us to create neural networks with few lines of code.

**Image-Classifier.py**: How to create a small image classifier

**GCP-APIs.py**: Demonstration of the Google Cloud Platform APIs

**Flask-App.py**:  How to create a small Flask app that can classify images and run it on a server with GCP. 

<br>














