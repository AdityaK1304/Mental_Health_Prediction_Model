
import pandas as pd
from collections import OrderedDict
# data exploration

def data_exploration(df):

    # select numerical columns 

    numerical_col = df.select_dtypes(exclude='object').columns
    stats = []
    for i in numerical_col:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR
        outlier_flag = "Has Outliers" if df[(df[i] < LW) | (df[i] > UW)].shape[0] > 0 else "No Outliers"

        numerical_stats = OrderedDict({
            "Feature": i,
            "Minimum": df[i].min(),
            "Maximum": df[i].max(),
            "Mean": df[i].mean(),
            "Median": df[i].median(),
            "Mode": df[i].mode().iloc[0] if not df[i].mode().empty else np.nan,
            "25%": Q1,
            "75%": Q3,
            "IQR": IQR,
            "Standard Deviation": df[i].std(),
            "Skewness": df[i].skew(),
            "Kurtosis": df[i].kurt(),
            "Outlier Comment": outlier_flag
        })
        stats.append(numerical_stats)
        numerical_stats_report = pd.DataFrame(stats)
    return  numerical_stats_report   

