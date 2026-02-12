# def data preprocessing 
from sklearn.preprocessing import LabelEncoder

def data_preprocessing(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    return df
 
