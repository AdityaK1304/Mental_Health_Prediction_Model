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

# model evalution

def model_evaluation(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = accuracy

        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))


        best_model = max(results, key=results.get)
        print("Best Model:", best_model)
        print("Best Accuracy:", results[best_model])

    return best_model