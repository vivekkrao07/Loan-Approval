import pandas as pd

def predict_new_customer(clf, customer_input, feature_columns):
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([customer_input])

    # One-hot encode with drop_first=True to match training
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Match column order and fill missing with 0
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = clf.predict(input_encoded)[0]
    return prediction, input_encoded
