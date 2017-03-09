__author__ = 'Juan Harrington' # please type your name

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

# this function takes the drugcount dataframe as input and output a tuple of 3 data frames: DrugCount_Y1,DrugCount_Y2,DrugCount_Y3
def process_DrugCount(drugcount):
    '''
    you code here
    '''
    #remove the '+' from DrugCount column and convert it to int datatype
    drugcount["DrugCount"] = drugcount["DrugCount"].map(lambda x: x.rstrip("+"))
    drugcount["DrugCount"] = drugcount["DrugCount"].apply(pd.to_numeric)
    # split the table into three data frames by year
    DrugCount_Y1 = drugcount[drugcount["Year"] == "Y1"]
    DrugCount_Y2 = drugcount[drugcount["Year"] == "Y2"]
    DrugCount_Y3 = drugcount[drugcount["Year"] == "Y3"]
    return (DrugCount_Y1,DrugCount_Y2,DrugCount_Y3)

# this function converts strings such as "1- 2 month" to "1_2"
def replaceMonth(string):
    a_new_string = ""
    '''
    you code here
    '''
    #replace - with _ and remove month(s) from string
    a_new_string = "DSFS_" + string.replace(" ","").replace("-","_").replace("months","").replace("month","")
    return a_new_string

# this function processes a yearly drug count data
def process_yearly_DrugCount(aframe):
    processed_frame = None
    '''
    your code here
    '''
    #drop year from data set
    aframe.drop("Year", axis=1, inplace=True)
    # categorical variable recoding
    #print aframe["DSFS"].unique() 
    df_recode = aframe["DSFS"].apply(replaceMonth)
    df_dummy = pd.get_dummies(df_recode)
    # join the dummy variable to the main dataframe
    df_joined = pd.concat([aframe,df_dummy], axis=1)
    #drop DSFS from data set
    df_joined.drop("DSFS", axis=1, inplace=True)
    # group and aggregate drugcount
    df_grouped = df_joined.groupby(df_joined["MemberID"],as_index=False)
    agg_df = df_grouped.agg(np.sum)
    #rename column
    agg_df.rename(columns = {"DrugCount":"Total_DrugCount"},inplace=True)
    processed_frame = agg_df
    return processed_frame

# run linear regression. You don't need to change the function
def linear_regression(train_X, test_X, train_y, test_y):
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_y)
    print 'Coefficients: \n', regr.coef_
    pred_y = regr.predict(test_X) # your predicted y values
    # The root mean square error
    mse = np.mean( (pred_y - test_y) ** 2)
    import math
    rmse = math.sqrt(mse)
    print ("RMSE: %.2f" % rmse)
    from sklearn.metrics import r2_score
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)

# for a real-valued variable, replace missing with median
def process_missing_numeric(df, variable):
    # below is the code I used in the lecture ("exploratory_analysis.py") for dealing with missing values of the variable "age".
    # You need to change the code below slightly
    '''df['Age_missing'] = np.where(df['Age'].isnull(),1,0)
    medianAge = df.Age.median()
    df.Age.fillna(medianAge, inplace= True)'''
    df[variable+'_missing'] = np.where(df[variable].isnull(),1,0)
    medianVariable = df[variable].median()
    df[variable].fillna(medianVariable, inplace=True)

# This function prints the ratio of missing values for each variable. You don't need to change the function
def print_missing_variables(df):
    for variable in df.columns.tolist():
        percent = float(sum(df[variable].isnull()))/len(df.index)
        print variable+":", percent

def main():
    pd.options.mode.chained_assignment = None # remove the warning messages regarding chained assignment. 
    daysinhospital = pd.read_csv('DaysInHospital_Y2.csv') # you may need to change the file path
    drugcount = pd.read_csv('DrugCount.csv') # you may need to change the file path
    li = map(process_yearly_DrugCount, process_DrugCount(drugcount))
    DrugCount_Y1_New = li[0]    
    # your code here to create Master_Assn3 by merging daysinhospital and DrugCount_Y1_New
    Master_Assn3 = pd.merge(daysinhospital, DrugCount_Y1_New, left_on='MemberID', right_on='MemberID', how='left') # left join
    process_missing_numeric(Master_Assn3, 'Total_DrugCount')
    # Your code here for deal with missing values of the dummy variables. Please don't overthink this. You just need to write one line of code to replace all missing values with 0
    Master_Assn3.fillna(0, inplace=True)
    # Your code here to drop the column 'MemberID'. You need to drop the column in place
    Master_Assn3.drop("MemberID", axis=1, inplace=True)
    print Master_Assn3.shape
    print Master_Assn3.head(3)
    '''output:
    ClaimsTruncated  DaysInHospital  Total_DrugCount  DSFS_0_1  DSFS_10_11  \
    0                0               0              3.0       0.0         0.0   
    1                0               0              1.0       1.0         0.0   
    2                1               1             23.0       1.0         1.0   
    
       DSFS_11_12  DSFS_1_2  DSFS_2_3  DSFS_3_4  DSFS_4_5  DSFS_5_6  DSFS_6_7  \
    0         0.0       0.0       0.0       1.0       0.0       0.0       0.0   
    1         0.0       0.0       0.0       0.0       0.0       0.0       0.0   
    2         0.0       0.0       1.0       1.0       1.0       1.0       1.0   
    
       DSFS_7_8  DSFS_8_9  DSFS_9_10  DrugCount_missing  
    0       0.0       0.0        0.0                  0  
    1       0.0       0.0        0.0                  0  
    2       1.0       1.0        1.0                  0  
    '''
    dependent_var = 'DaysInHospital'
    # The next two lines of code creat a list of independent variable names.
    independent_var = Master_Assn3.columns.tolist()
    independent_var.remove(dependent_var)
    # next we split the data into training vs. test. 
    train_X, test_X, train_y, test_y= train_test_split(Master_Assn3[independent_var], Master_Assn3[dependent_var], test_size=0.3, random_state=123)
    print train_X.shape, test_X.shape, train_y.shape, test_y.shape
    linear_regression(train_X, test_X, train_y, test_y)
    '''outputs:
    (76038, 16)
    (53226, 15) (22812, 15) (53226L,) (22812L,)
    ['ClaimsTruncated', 'Total_DrugCount', 'DSFS_0_1', 'DSFS_10_11', 'DSFS_11_12', 'DSFS_1_2', 'DSFS_2_3', 'DSFS_3_4', 'DSFS_4_5', 'DSFS_5_6', 'DSFS_6_7', 'DSFS_7_8', 'DSFS_8_9', 'DSFS_9_10', 'DrugCount_missing']
    ['ClaimsTruncated', 'Total_DrugCount', 'DSFS_0_1', 'DSFS_10_11', 'DSFS_11_12', 'DSFS_1_2', 'DSFS_2_3', 'DSFS_3_4', 'DSFS_4_5', 'DSFS_5_6', 'DSFS_6_7', 'DSFS_7_8', 'DSFS_8_9', 'DSFS_9_10', 'DrugCount_missing']
    Coefficients: 
    [ 0.9318124   0.01375816 -0.06578017 -0.01265249 -0.02528982  0.02518371
      0.02925542 -0.01464571 -0.02716822  0.00613997  0.00668293 -0.05660316
     -0.03220839  0.0174422  -0.18013557]
    RMSE: 1.60
    R2 value: -23.80 # don't worry about the negative value. It simply means the model is bad
    '''

if __name__ == '__main__':
    main()




