
import pandas as pd
# data ingestion

def data_ingestion():
    df = pd.read_csv(r'C:\Mental_Health_Prediction_Model\data\raw\survey.csv')
    return df