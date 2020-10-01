import numpy as np
from collections import Counter


def identify_outliers(df, n, features):
    """
    This function accepts a dataframe, a maximum-limit integer, n, and a set of columns.
    First, the function determines the interquartile range (IQR) of each column. Second, it
    identifies rows of the dataframe where these columns have outliers. Third, it stores the 
    indexes of these columns in a list. Fourth, it converts the list into a Counter instance. Finally,
    it returns the indices that occur more than n times in the Counter instance.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = df[col].quantile(0.25)
        # 3rd quartile (75%)
        Q3 = df[col].quantile(0.75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   


'''
Reference
https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
'''