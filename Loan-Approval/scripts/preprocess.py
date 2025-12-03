# scripts/preprocess.py
import pandas as pd

# Fixed mappings for categorical columns
GENDER_MAP = {'Male': 1, 'Female': 0, 'Unknown': -1}
MARRIED_MAP = {'Yes': 1, 'No': 0, 'Unknown': -1}
DEPENDENTS_MAP = {'0': 0, '1': 1, '2': 2, '3+': 3, 'Unknown': -1}
EDUCATION_MAP = {'Graduate': 1, 'Not Graduate': 0, 'Unknown': -1}
SELF_EMPLOYED_MAP = {'Yes': 1, 'No': 0, 'Unknown': -1}
PROPERTY_AREA_MAP = {'Urban': 2, 'Semiurban': 1, 'Rural': 0, 'Unknown': -1}

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure target column exists
    if 'Loan_Status' not in df.columns:
        df['Loan_Status'] = 'N'
    y = df['Loan_Status']

    # Ensure all expected categorical columns exist
    for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']:
        if col not in df.columns:
            df[col] = 'Unknown'

    # Ensure all expected numeric columns exist
    for col in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']:
        if col not in df.columns:
            if col == 'Credit_History':
                df[col] = 1  # assume good credit
            elif col in ['LoanAmount','Loan_Amount_Term']:
                df[col] = 0
            else:
                df[col] = 10000  # default income

    X = df.drop('Loan_Status', axis=1)

    # Standardize column names
    X.columns = X.columns.str.strip().str.replace(' ', '_')

    # Map categorical columns to numeric
    X['Gender'] = X['Gender'].fillna('Unknown').map(GENDER_MAP)
    X['Married'] = X['Married'].fillna('Unknown').map(MARRIED_MAP)
    X['Dependents'] = X['Dependents'].fillna('Unknown').map(DEPENDENTS_MAP)
    X['Education'] = X['Education'].fillna('Unknown').map(EDUCATION_MAP)
    X['Self_Employed'] = X['Self_Employed'].fillna('Unknown').map(SELF_EMPLOYED_MAP)
    X['Property_Area'] = X['Property_Area'].fillna('Unknown').map(PROPERTY_AREA_MAP)

    # Fill numeric missing values
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    # Convert LoanAmount to rupees
    X['LoanAmount'] = X['LoanAmount'].apply(lambda x: max(x * 1000, 50000))

    # Ensure incomes and Loan_Amount_Term are realistic
    for income_col in ['ApplicantIncome', 'CoapplicantIncome']:
        X[income_col] = X[income_col].apply(lambda x: max(x, 10000))
    X['Loan_Amount_Term'] = X['Loan_Amount_Term'].apply(lambda x: max(x, 12))

    # New Feature: Loan-to-Income ratio
    X['Loan_to_Income'] = X['LoanAmount'] / (X['ApplicantIncome'] + X['CoapplicantIncome'] + 1)

    return X, y

# For testing
if __name__ == "__main__":
    X, y = load_and_preprocess_data("data/ProcessedLoan.csv")
    print(X.head())
    print(y.head())
