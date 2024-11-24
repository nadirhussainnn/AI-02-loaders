import pandas as pd
from sklearn.impute import SimpleImputer
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize'] = [4.0, 2.0]
plt.rcParams.update({'font.size': 8})

def load_data(path, delimiter=","):
    return pd.read_csv(path, delimiter=delimiter)


categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
numerical_cols = ['Dependents', 'ApplicantIncome','CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History' ]

def fill_categorical(df):
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    return df

def fill_numerical(df):
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

# Function to clean up the column and retain only numeric values
def remove_chars_from_numericals(df, column_name):
    def clean_value(value):
        if isinstance(value, str):  # Check if the value is a string
            numeric_part = re.findall(r'\d+', value)  # Extract numeric part using regex
            return int(numeric_part[0]) if numeric_part else None  # Convert to int or return None
        return value  # Return as is if already numeric

    df[column_name] = df[column_name].apply(clean_value)
    return df

# I have used LabelEncoder instead of One-hot encoder because
'''
Label encoder does not create column for each value, it just created one column and fills it with 0,1 values depending on unique values column has.
    - Use when the categorical values have a natural order (e.g., "Low", "Medium", "High").
    - May not work well with non-ordinal categories in some algorithms (e.g., linear regression), as it might imply a false ordinal relationship.

On other hand, OneHot encoder creates column for each unique value and puts 1 in the column that matches the true value, for rest it puts 0
    - Use for algorithms that do not assume ordinal relationships between categories (e.g., logistic regression, tree-based models).
    - Can increase the dimensionality of the dataset if there are many categories.

Note that, LabelEncoder and OneHot encoder both keep the original column. Depends on use case if we should keep or not the original column
'''
def convert_to_numeric(df, column):

    label_encoder = LabelEncoder()
    df[column+"_Encoded"] = label_encoder.fit_transform(df[column])
    df.drop(column, axis=1, inplace=True)

    return df

# Let's use MinMax Scaler to normalie data between 0 and 1 range
'''
X' = (X - Xmin) / (Xmax - Xmin)
'''

# We could also use z-normalization provided by StandardScaler. It will make mean 0 and standard deviation of 0 for the values.

def normalize_values(df, column):
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df

def pre_process(df):

    df = df.drop(columns = ["Loan_ID"])
    df = remove_chars_from_numericals(df, 'Dependents')
    df = fill_categorical(df)
    df = fill_numerical(df)
    df = convert_to_numeric(df, 'Gender')
    df = convert_to_numeric(df, 'Education')
    df = convert_to_numeric(df, 'Married')
    df = convert_to_numeric(df, 'Self_Employed')
    df = convert_to_numeric(df, 'Property_Area')
    
    df = normalize_values(df, 'ApplicantIncome')
    df = normalize_values(df, 'CoapplicantIncome')
    df = normalize_values(df, 'LoanAmount')
    return df

def subplots(df):

    fig, axes = plt.subplots(2, 4, figsize=(8, 4))

    fig.suptitle('Frequency of non-numeric columns')
    sns.countplot(ax=axes[0, 0], data=df, x='Gender')
    sns.countplot(ax=axes[0, 1], data=df, x='Married')
    sns.countplot(ax=axes[0, 2], data=df, x='Dependents')
    sns.countplot(ax=axes[0, 3], data=df, x='Education')
    sns.countplot(ax=axes[1, 0], data=df, x='Self_Employed')
    sns.countplot(ax=axes[1, 1], data=df, x='Credit_History')
    sns.countplot(ax=axes[1, 2], data=df, x='Property_Area')
    sns.countplot(ax=axes[1, 3], data=df, x='Loan_Status')

    plt.tight_layout()
    plt.show()