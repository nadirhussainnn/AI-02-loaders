There are few simple steps to use pre-built classifiers to on binary or multi class datasets

Step - 01:
Install and import necessary python packages i.e Pandas, Matplotlib, scikit-learn etc

Step - 02:
Load data from csv. 

Step - 03:
Pre-process data. 
    - Feature engineering (Select features which are most important for predictor)
    - Data cleaning (Fill or remove null values, convert data into numeric form etc)
    
Step - 04:
Split dataset into train, validate and test sets

Step - 05:
Applly classification algorithm on train_x and train_y

Step - 06:
Validate model on validate_x to predict validate_y. 

Step - 07:
Calculate confusion metrics (acuracy, recall, mcc, fpr) in case of binary class. Else rely on other useful metric

Step - 08:
Deploy model using Pickle or joblib

Step - 09:
Test model on test_x, and compare with test_y to calculate accuracy and other metrics.