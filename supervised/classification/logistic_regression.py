from math import sqrt
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

df = load_data('../loan.csv')  

# Goal: Predict if a person should be given loan or not based on features. 
# Or predict if person has taken loan or not.

'''
print(df.describe())
df.info()

1. Numeric values are not normalized for almost all numeric features
2. Loan ID is unique hance it does not contribute to prediction. df["Loan_ID"].nunique(). print(df["Loan_ID"].value_counts())
3. Dependents has missing values, and has strings in numbers i.e 3+
4. Gender, Married, Self_Employed are categorical features with missing values
5. Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History are numerical with missing values
6. Gender, Education, Married, Self_Employed, Property_Area are categorical with few unique values so must be converted into numerical

'''

'''
EDA: Exploratory data analysis
 - Visualizations
 - Preprocessing
'''

# Visualizations
# subplots(df)

# Lets pre-process dataset by fixing above issues
df = pre_process(df)

y = df['Loan_Status']
x = df.drop(columns = ['Loan_Status'])

# split into train, test sets
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=42, shuffle=True)

x_test_set = X_test
y_test_set = y_test

# Further split train set into train and validate
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.10, random_state=42, shuffle=True)

x_validate_set = X_test
y_validate_set = y_test

# model = LogisticRegression(max_iter=400)
model = LinearSVC(max_iter=1000, random_state=42)

# Add this to handle oversampling issue
# oversampler = SMOTE(random_state=42)
# X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
# model = model.fit(X_resampled, y_resampled)

model = model.fit(X_train, y_train)
y_pred = model.predict(x_validate_set)

tn, fp, fn, tp = confusion_matrix(y_validate_set, y_pred).ravel()
report = classification_report(y_validate_set, y_pred)

accuracy = (tp + tn) / (tn+fp+fn+tp)
'''
the algorithm may give very good accuracy, very goot recall but it may be due to unbalanced dataset i.e normal/anamolous data points may differ a lot. So we use MCC.
if MCC = 1 : Perfect classification
If MCC = -1: Completely wrong classification
If MCC = 0: Random guess
'''

mcc = (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
print("\n-------------------\nValidation Set Stats\n-------------------")
print('Accuracy: %.2f \nMCC:%.2f\nTP:%.0f TN:%.0f FP:%.0f FN:%.0f' % (accuracy*100, mcc, tp, tn, fp, tn))

y_pred = model.predict(x_test_set)

tn, fp, fn, tp = confusion_matrix(y_test_set, y_pred).ravel()
accuracy = (tp + tn) / (tn+fp+fn+tp)
mcc = (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

print("\n-------------------\nTest Set Stats\n-------------------")
print('Accuracy: %.2f \nMCC:%.2f\nTP:%.0f TN:%.0f FP:%.0f FN:%.0f' % (accuracy*100, mcc, tp, tn, fp, tn))
