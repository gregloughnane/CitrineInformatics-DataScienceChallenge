# CitrineInformatics-DataScienceChallenge

The idea behind this challenge is to use data to solve a canonical thermodynamics problem: given a pair of elements, predict the stable binary compounds that form on mixing. 
The training labels we've provided are a discretization of the 1D binary phase diagram at 10% intervals. For example, the label for OsTi ([1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]) translates into the following stable compounds:  Os, Os{0.5}Ti{0.5} or OsTi, and Ti. 
The features chosen for this application are based on a naïve application of the Material-Agnostics Platform for Informatics and Exploration (magpie) feature set, 
http://wolverton.northwestern.edu/research/machine-learning
Feel free to prune, extend, or otherwise manipulate this feature set in pursuit of a predictive model!
Input Features 
Your task is to build a machine learning model in Python to predict the full stability vector.  
Please explain your approach, including your choice of algorithm, data-preprocessing, and model accuracy evaluation.  
Please write a framework for hyper-parameter tuning from scratch (not using sci-kit learn’s built in methods). 
Please also estimate the performance of your model on the test set (both precision and recall) and explain your methodology in the write up.  
Include a discussion of when your model may be more or less accurate.
