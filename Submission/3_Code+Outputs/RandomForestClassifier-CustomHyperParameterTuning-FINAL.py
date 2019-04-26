# -*- coding: utf-8 -*-
"""
@author: Greg Loughnane

Citrine Informatics Data Science Challenge

Given:  Pair of Elements

Find:   Predict the stable binary compounds that will form upon mixing
        i.e., Calculate full, discretized stability vector
        
Sol'n:  Classical ML --> Use optimized Random Forest classifier (see report)

Notes:  - This program requires the following libraries to be present in your
          virtual environment
        - pandas, sklearn, itertools, time, numpy, matplotlib, seaborn, mlxtend
        
        - This program was written in the Sypder IDE via Anaconda Navigator and 
          will run directly after 1) the appropriate virtual environment is 
          set up, and 2) the working directory in Spyder is set to the  
          unzipped folder, which includes the modified .csv files that are 
          called out in the script and described in the included report
          
        - Hyperparameter tuning can be completed by comma-separating additional
          parameters within each list shown in lines 61-67
          (e.g., max_features = ['log2', 'sqrt'] or min_samples_split = [2, 4])
"""

# Read in Salient Features and Stability Vector Outputs Less Pure Elements
import pandas as pd
FeatureData     = 'training_data_x2-extended&reduced.csv'
end = 83        # see report for description of parameters added/removed
OutputData      = 'training_data_stabilityVec_x2-extended&reduced.csv'

# Read in Elemental Features and Stability Vectors, convert to numpy.ndarrays
Features        = pd.read_csv(FeatureData)
Outputs         = pd.read_csv(OutputData)
x               = Features.iloc[:,2:end].values
y               = Outputs.iloc[:].values

# Initialize Constant Parameters
test_set_size   = 0.3   # 30% Test Set
val_set_size    = 0.1   # 10% Validation Set (for Hyperparameter Tuning)

# Normalize Features columnwise to distributions with mean of 0 std dev of 1 
x = x - x.mean(axis = 0)
X = x / x.std(axis = 0)

# Split data into training, development, and test data (60/10/30 split)
from sklearn.model_selection import train_test_split

# Create Test Set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_set_size)

# Create Train and Validation Sets
partial_x_train, x_dev, partial_y_train, y_dev = \
    train_test_split(X_train, y_train, test_size=val_set_size)

# Define HyperParameters (only optimal parameters shown), do not change order 
max_features        = ['log2']      # see line 97
max_depth           = [None]        # ...
n_estimators        = [30]          # ...
criterion           = ['gini']      # ...
min_samples_leaf    = [1]           # ...
min_samples_split   = [2]           # ...
bootstrap           = ['True']      # see line 103

# Create list of possible combinations to try
import itertools
HyperParamCombos = list(itertools.product(max_features,     
                                          max_depth,       
                                          n_estimators,     
                                          criterion,        
                                          min_samples_leaf,
                                          min_samples_split,
                                          bootstrap))

# Import toolboxes and allocate arrays for storing hyperparameter tuning info
import time
import numpy as np
store_parameters    = []
store_train_acc     = []
store_dev_acc       = []
store_test_acc      = []
store_precisions    = []
store_recalls       = []
store_macro_fscore  = []
store_timings       = []

for row in HyperParamCombos:
    
    # Time each combination to include in consideration
    start_time = time.time()
    
    # Retrieve featurs from combination list for each iteration
    max_features        = row[0]
    max_depth           = row[1]
    n_estimators        = row[2]
    criterion           = row[3]
    min_samples_leaf    = row[4]
    min_samples_split   = row[5]
    bootstrap           = row[6]
    
    # Define form of Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(max_features        = max_features, 
                                max_depth           = max_depth, 
                                n_estimators        = n_estimators,
                                criterion           = criterion,
                                min_samples_leaf    = min_samples_leaf,
                                min_samples_split   = min_samples_split,
                                bootstrap           = bootstrap)
   
    # Train model on 60% of the original data
    rf.fit(partial_x_train, partial_y_train)
    
    # Import plotting toolboxes
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate feature importance for each feature
    feature_imp = pd.Series(rf.feature_importances_).sort_values(ascending=False)
    
    # Create bar graph of feature importances
    plt.figure(figsize=(20,6))
    sns.barplot(x = feature_imp.index, y = feature_imp)
    plt.xlabel('Features')
    plt.ylabel('Feature Importance Score')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()
    
    # Calculate predictions within training, dev, and test sets
    train_predictions   = rf.predict(partial_x_train)
    dev_predictions     = rf.predict(x_dev)
    test_predictions    = rf.predict(X_test)
    
    # Calculate train, dev, and test accuracies via scikit-learn accuracy_score
    from sklearn.metrics import accuracy_score
    print()
    train_acc   = accuracy_score(partial_y_train, train_predictions )
    dev_acc     = accuracy_score(y_dev,           dev_predictions)
    test_acc    = accuracy_score(y_test,          test_predictions)
    
    print("TrainSet  Accuracy   :: ", train_acc)
    print("DevSet    Accuracy   :: ", dev_acc)
    print("TestSet   Accuracy   :: ", test_acc)
    
    # Import scikit-learn metrics module for classification_report
    from sklearn.metrics import classification_report
    print()
    print('Report on Classifications for each "Class" ...')
    print()
    target_names = ['10/90% A/B', '20/80% A/B', '30/70% A/B', '40/60% A/B', \
                    '50/50% A/B', '60/40% A/B', '70/30% A/B', '80/20% A/B', \
                    '90/10% A/B']
    print(classification_report(y_dev, 
                                dev_predictions, 
                                target_names=target_names))
    
    # Extract macro precision, recall, and fscore from the classificaiton report
    from sklearn.metrics import precision_recall_fscore_support as score
    precision, recall, fscore, support = score(y_dev, 
                                               dev_predictions, 
                                               average='macro')
    macro_precision = precision
    macro_recall = recall
    macro_fscore = fscore
    
    # Create Binary Confusion Matrix (in part because for some reason the multi-
    # class confusion matrix didn't want to cooperate, but we can also gain 
    # additional insight into the model in an easy-to-interpret way by doing this
    
    # First Amalgamate each row of dev set predictions and ground truths 
    # into long row vectors to simulate a simple binary prediction
    actual      =   y_dev.flatten()
    predicted   =   dev_predictions.flatten()
    
    # Create confusion matrix (via mlxtend vs. sklearn for looks)
    from mlxtend.evaluate import confusion_matrix 
    cm          =   confusion_matrix(y_target=actual, 
                                     y_predicted=predicted,
                                     binary=True)
    
    # Plot confusion matrix
    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    print('Binary Confusion Matrix (to get a feel for overall performance) :')
    plt.show()
    
    # Print Binary Classification report
    print(classification_report(actual, predicted))
    
    # Print associated confusion matrix (sklearn confusion matrix calcs now)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    print()
    print("True Negatives  ~ Got it Right!                      :: ", tn)
    print("False Positives ~ Type I  Error - reject true null   :: ", fp)
    print("False Negatives ~ Type II Error - accept false null  :: ", fn)
    print("True Positives  ~ Got it Right!                      :: ", tp)
    print()
    
    # Calculate Classical Metrics of Model Performance and print
    precision   = tp / (fp + tp)
    recall      = tp / (tp + fn)
    F1_score    = 2 *(precision * recall) / (precision + recall)
    print("Precision                                            :: ", precision)
    print("Recall                                               :: ", recall)
    print("F1 Score                                             :: ", F1_score)
    print()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity                                          :: ", sensitivity)
    print("Specificity                                          :: ", specificity)
    print()
    
    accuracy    = (tn + tp) / (tn + fp + fn + tp)
    print("Accuracy                                             :: ", accuracy)
    
    end_time = time.time()
    timing = end_time - start_time
    print("Total Model Run Time                                 :: ", timing)
    
    # These are for displaying hyperparameter tuning combinations and outputs
    store_train_acc.append(train_acc)
    store_dev_acc.append(dev_acc)
    store_test_acc.append(test_acc)
    store_parameters.append(row)
    store_precisions.append(macro_precision)
    store_recalls.append(macro_recall)
    store_macro_fscore.append(macro_fscore)
    store_timings.append(timing)
    
# ---------------Display hyperparameter tuning outputs---------------
print()
print('Training Set Accuracies...\n',           np.transpose(store_train_acc))
print()
print('Development Set Accuracies...\n',        np.transpose(store_dev_acc))
print()
print('Model Hyperparameters...\n',             store_parameters)
print()
print('Model Precisions...\n',                  np.transpose(store_precisions))
print()
print('Model Recalls...\n',                     np.transpose(store_recalls))
print()
print('Model F1 Scores...\n',                   np.transpose(store_macro_fscore))
print()
print('Model Timings (seconds)...\n',           np.transpose(store_timings))

# ---------------Perform Analysis of Challenge Data---------------

# Read in data
ChallengeDataInputName  = 'test_data_extended&reduced.csv'
ChallengeDataOutputName = 'PredictedStabilityVectors.txt'
ChallengeData           = pd.read_csv(ChallengeDataInputName)
X_Challenge             = ChallengeData.iloc[:,2:end].values

# Normalize
X_Challenge             = X_Challenge - X_Challenge.mean(axis = 0)
X_Challenge             = X_Challenge / X_Challenge.std(axis = 0)

# Make Predictions
ChallengePredictions    = rf.predict(X_Challenge)
print()
print('For any given Random Forest model chosen:')
print('I expect my blind predictions to perform about this well as the')
print()
print('Test Set Accuracies...\n',                np.transpose(store_test_acc))

# Add back in the Pure Elemental Compositions
n,m                     = ChallengePredictions.shape # for generality
Pures                   = np.ones((n, 1))
ChallengePredictions    = np.hstack((Pures, ChallengePredictions))
ChallengePredictions    = np.hstack((ChallengePredictions, Pures))

# ---------------Convert data to the same format as received---------------
# First convert numpy array to list
ChallengePredictions    = ChallengePredictions.tolist()

# Save list directly to txt, giving each number the correct format
np.savetxt(ChallengeDataOutputName,
           ChallengePredictions, 
           fmt='%1.1f', 
           delimiter=',')

# Open txt file and add brackets to each row
with open(ChallengeDataOutputName) as f:
    lines = f.read().splitlines()
with open(ChallengeDataOutputName, "w") as f:
    for line in lines:
        print("[" + line + "]", file=f)
        
## Add header of "stabilityVec"
with open(ChallengeDataOutputName, 'r') as original: 
    data = original.read()
with open(ChallengeDataOutputName, 'w') as modified: 
    modified.write("stabilityVec\n" + data)

# -----Note: Txt can be copied/pasted directly into as-received CSV format-----
    
print('\nThank you!\n\nBest regards,\nGreg Loughnane')