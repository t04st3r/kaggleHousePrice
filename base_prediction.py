import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


def rmse(y_pred, y_val):
    """ compute the root mean squared error"""
    return sqrt(mean_squared_error(y_pred, y_val))


def split_cat_num(X):
    """ split dataset into categorical and numerical datasets"""
    cat = []
    for attr in X.columns.values:
        if X[attr].dtype == 'object':
            cat.append(attr)
    num_ds = X.drop(cat, axis=1)
    cat_ds = X[cat]
    return num_ds, cat_ds


# create pandas DataFrame from csv
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the
rf_model_on_full_data.fit(X, y)

train_y = rf_model_on_full_data.predict(X)
print(rmse(train_y, y))

# path to file you will use for predictions
test_data_path = 'test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(test_X)

# create submission file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
