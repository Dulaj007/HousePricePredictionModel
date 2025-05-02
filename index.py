# 1. Import necessary libraries

# pandas is used for loading, reading, and working with data in table format (like CSV files)
import pandas as pd

# train_test_split is a function from scikit-learn that splits your dataset into training and validation (testing) parts
from sklearn.model_selection import train_test_split

# mean_absolute_error is a function to measure how accurate your model is (lower = better); it tells how far off your predictions are from actual values
from sklearn.metrics import mean_absolute_error

# RandomForestRegressor is a machine learning algorithm that builds many decision trees and averages their results (used for predicting numbers)
from sklearn.ensemble import RandomForestRegressor

# 2. Load training data
train_data_path = 'Data/train.csv'
# read the loaded data file
home_data = pd.read_csv(train_data_path)

# 3. Select target (y) and features (X)
# y- is the predict output
y = home_data.SalePrice
# x- is the inputs according to the predict output
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# 4. Split into training and validation data (to evaluate model performance)
# in here we divide our home_data file into traning data and test data to masture MAE
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Example: 80% training, 20% validation
# train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

# 5. Define the model (Random Forest - usually better than single Decision Tree)
model = RandomForestRegressor(random_state=1)

# 6. Train the model
model.fit(train_X, train_y)

# 7. Predict on validation data
val_predictions = model.predict(val_X)

# 8. Evaluate the model using Mean Absolute Error
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

# 9. Now use all the training data to train final model
final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)

# 10. Load test data
test_data_path = 'Data/test.csv'
test_data = pd.read_csv(test_data_path)

# 11. Select the same features from test data
test_X = test_data[features]

# 12. Make predictions on the test data
# in here we make prediction based on brand new testing data set 
test_preds = final_model.predict(test_X)

# 13. Save the predictions to a CSV file for submission or review
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
