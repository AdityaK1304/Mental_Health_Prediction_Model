from sklearn.model_selection import train_test_split
# import manipulation lybraries
import pandas as pd
import numpy as np 
# import visualization lybraries
import seaborn as sns
import matplotlib.pyplot as plt

#import machine learning lybraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# model building

def model_building(df):
    X = df.drop("treatment", axis=1)
    y = df["treatment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier()
    }

    model_train = {}

    for name, model in models.items():
        model.fit(X_train,y_train)
        model_train[name] = model
    return model_train, scaler, X_test, y_test