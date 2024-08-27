# importing required functions and libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, root_mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# loading the data
file_path = r'Data_filepath'
data = pd.read_excel(file_path, skiprows=2)
data = data.iloc[:92, :]

# Dealing with the missing values
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# excluding row indexed as 14, 30, 33 (R and O) and 58 (Pub)
excluded_indices = [14, 30, 33, 58, 32, 38, 29, 44, 35, 23, 27, 62, 66, 47, 24]
filtered_data = data.drop(index=excluded_indices)

# Extracting inputs and target
X = filtered_data.iloc[:, :4].values  # inputs
Yaa = filtered_data.iloc[:, 4].values  # acetic acid conc.
Ybm = filtered_data.iloc[:, 5].values  # biomethane conc.
Yh = filtered_data.iloc[:, 6].values  # hydrogen conc.

# Defining the cross validation procedure
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the scoring metrics
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)
rmse_scorer = make_scorer(root_mean_squared_error)

# defining a CV function to perform the cross validation and print the results, function can then be applied to all outputs
def perform_cross_validation(model, X, y, y_name):
    # Tuning number of estimators for RFR
    # Setting the paramter grid
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]}
    # Performing the grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                               n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # defining best parameters
    print(f'{y_name} best no. estimators: {grid_search.best_params_}')

    # Perform cross-validation with the best model
    mse_scores = cross_val_score(best_model, X, y, cv=kf, scoring=mse_scorer)
    r2_scores = cross_val_score(best_model, X, y, cv=kf, scoring=r2_scorer)
    rmse_scores = cross_val_score(best_model, X, y, cv=kf, scoring=rmse_scorer)

    # Convert negative MSE to positive
    mse_scores = -mse_scores

    # Printing the mean results for the different folds
    if y_name == 'Acetic Acid': #improves clarity of results in run window
            print(f'')

    print(f'{y_name} Results:')
    print(f'Mean MSE: {np.mean(mse_scores)}')
    print(f'Standard Deviation of MSE: {np.std(mse_scores)}')
    print(f'Mean RMSE {(np.mean(rmse_scores))}')
    print(f'Mean R²: {np.mean(r2_scores)}')
    print(f'Standard Deviation of R²: {np.std(r2_scores)}')
    print(f'') #create a space to separate results

    return best_model  # Return the best model

# Graph plot
# Function to collect predictions for plotting
def collect_predictions(model, X, y):
    predicted = np.zeros_like(y)
    actual = np.zeros_like(y)
    fold = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model.fit(X_train, y_train)

        # Collect predictions
        predicted[test_index] = model.predict(X_test)
        actual[test_index] = y_test

        fold += 1

    return actual, predicted


# Applying cross-validation and collecting predictions for each target variable
best_model_aa = perform_cross_validation(RandomForestRegressor(random_state=42), X, Yaa, 'Acetic Acid')
best_model_bm = perform_cross_validation(RandomForestRegressor(random_state=42), X, Ybm, 'Biomethane')
best_model_h = perform_cross_validation(RandomForestRegressor(random_state=42), X, Yh, 'Hydrogen')

# Defining the function to plot predicted vs. actual results
def plot_predictions(actual, predicted, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Collect predictions for each output using the best model
actual_aa, predicted_aa = collect_predictions(best_model_aa, X, Yaa)
actual_bm, predicted_bm = collect_predictions(best_model_bm, X, Ybm)
actual_h, predicted_h = collect_predictions(best_model_bm, X, Yh)

# Plotting for each output using the defined function
plot_predictions(actual_aa, predicted_aa, 'Actual AA Concentration', 'Predicted AA Concentration', 'Predicted vs Actual AA Concentration')
plot_predictions(actual_bm, predicted_bm, 'Actual Biomethane Concentration', 'Predicted Biomethane Concentration', 'Predicted vs Actual Biomethane Concentration')
plot_predictions(actual_h, predicted_h, 'Actual H Concentration', 'Predicted H Concentration', 'Predicted vs Actual H Concentration')


# Function to plot feature importances
def plot_feature_importances(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.show()


# Feature names (assuming your inputs have meaningful names)
feature_names = filtered_data.columns[:4]

# Train models on the full dataset to get feature importances
best_model_aa.fit(X, Yaa)
best_model_bm.fit(X, Ybm)
best_model_h.fit(X, Yh)

# Plot feature importances for each model
plot_feature_importances(best_model_aa, feature_names, 'Feature Importances for Acetic Acid')
plot_feature_importances(best_model_bm, feature_names, 'Feature Importances for Biomethane')
plot_feature_importances(best_model_h, feature_names, 'Feature Importances for Hydrogen')
