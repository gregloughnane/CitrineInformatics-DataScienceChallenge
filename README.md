# CitrineInformatics-DataScienceChallenge

The idea behind this challenge is to use data to solve a canonical thermodynamics problem: given a pair of elements, predict the stable binary compounds that form on mixing. 

The training labels we've provided are a discretization of the 1D binary phase diagram at 10% intervals. For example, the label for OsTi ([1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]) translates into the following stable compounds:  Os, Os{0.5}Ti{0.5} or OsTi, and Ti. 

The features chosen for this application are based on a naïve application of the Material-Agnostics Platform for Informatics and Exploration (magpie) feature set, 
http://wolverton.northwestern.edu/research/machine-learning


Task: build a machine learning model in Python to predict the full stability vector.  

In this repository you will find my report that includes:

-Explained approach, including choice of algorithm, data-preprocessing, and model accuracy evaluation.  

-Framework for hyper-parameter tuning built from scratch (not using sci-kit learn’s built in methods). 

-Estimate of the performance of the  model on the test set.

PS - although I didn't get this job, I had a great time with this challenge, I still love the open nature of Citrine Informatics and their platform, and I hope it serves as a useful starting point for someone solving similar problems in the future!  

PPS - All of my final submission work is included, as well as the initial training_data.csv and test_data.csv files provided.

Websites (with other great tools) that may of interest if this repository is...https://citrine.io/ | https://citrination.com/ | https://github.com/CitrineInformatics

