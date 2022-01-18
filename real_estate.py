#Our objective is to create a machine learning model for a real estate
#prob for predicting the price of land based on features

#We are already given the training set, so it is supervised model

#Within the supervised model, we need regression not classification

#Batch model within the regression, because we take a predefined group
#rather than an auto updating set

#We will use Root Mean Square Error for performance measure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# We read data from csv file
housing = pd.read_csv("https://github.com/aaryaniitd/Housing_Prices_Project/blob/main/data.csv")
### see the top 5 rows of data using head()
##print(housing.head())
### see the info of the data
##print(housing.info())
### see the number of times each value appears
##print(housing['Chas'].value_counts())
### to explain data
##print(housing.describe())
##m, x = plt.subplots(figsize = (10,7))
##x.hist(housing,bins = 50)
##plt.show()
 
#Train-Test Splitting 1 (for learning purpose only)
##def split_train_test(data, test_ratio):
##    np.random.seed(42)
##    shuffled = np.random.permutation(len(data))
##    test_set_size = int(len(data)*test_ratio)
##    test_indices = shuffled[0:test_set_size]
##    train_indices = shuffled[test_set_size:len(shuffled)]
##    return data.iloc[train_indices], data.iloc[test_indices]

##train_set, test_set = split_train_test(housing, 0.2)
# the problem with shuffling is that my model brings random training set
# and it brings finally all the cases and my model would have already seen
# the test cases. So we use a seed generator, which keeps a randomly picked
# set as fixed.

                         #Train-Test Splitting 2
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['Chas']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#print(strat_test_set['Chas'].value_counts())    
#print(strat_train_set['Chas'].value_counts())

                         # Looking for Correlations
#Scaled between -1 to 1, whatever is +ve increases with increase in
#reference value and whatever is -ve decreases with increase in reference


#print(corr_matrix['Median'].sort_values(ascending = False))

#Scatter Matrix is used to get plot of every attribute with every attribute
#taken 2 at a time(can be same)
from pandas.plotting import scatter_matrix
#We now use the attributes we are interested in
attributes = ["Median", "AvgRooms", "Residential Land", "LowerPopu"]

# Trying out Attribute Combinations
# housing["TaxRoom"] = housing["Tax"]/housing["AvgRooms"]
#print(housing["TaxRoom"])
corr_matrix = housing.corr()
#print(corr_matrix["Median"].sort_values(ascending = False))
housing = strat_train_set.drop("Median",axis = 1)
housing_labels = strat_train_set["Median"].copy()

                         # Missing Attributes
#We have 3 options:
# 1. Get rid of missing data points
# 2. Get rid of whole attribute
# 3. Set the value to some(0, mean or median)

##option 1: a = housing.dropna(subset = ["AvgRooms"])
##          a.shape()

##option 2: housing.drop("AvgRooms", axis = 1).shape (original data same)

##option 3: median = housing["avgRooms"].median()
##          housing["AvgRooms"].fillna(median) (original data remains same)

## so if the data about rooms is missing then we would need the same median again and implement it wherever needed. sklearn has got a class for it called SimpleImputer.
## we create an imputer and use median as the strategy to fit it into our model
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)
## print(imputer.statistics_) : this finds the median for ALL the columns
## now we implement the imputer strategy as and fit it into the columns of our data
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns = housing.columns) #new data created with X axis as the imputer and y axis as the original housing columns
##print(housing_tr["AvgRooms"].describe())


                        # Feature Scaling
# For min-max scaling ((val-min)/(max-min)) we use MinMaxScaler from sklearn                        
# For Standardization((value-mean)/std) we use StandardScaler from sklearn


                        # Creating a Pipeline                        
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    # add as many as you want
    ('std_scaler',StandardScaler()),
    ])

housing_num_tr = my_pipeline.fit_transform(housing_tr)


                      # Selecting a Desired Model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor # This causes overfitting
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


                      # Evaluating the model
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

                      # Using Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# After cross validation, we have selcted decision tree model

def print_scores(scores):
    print("Scores: ",scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

                      # Saving the Model
from joblib import dump,load
dump(model, 'Dragon.joblib')

                      # Testing the Model
X_test = strat_test_set.drop("Median", axis= 1)
Y_test = strat_test_set["Median"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
