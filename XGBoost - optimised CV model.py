#importing required functions and libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import shap

#loading the data
file_path = r'Data_filepath'
data = pd.read_excel(file_path, skiprows=2)
data = data.iloc[:92, :]

#Dealing with the missing values
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# excluding row indexed as 14, 30, 33 (R and O)
excluded_indices = [14, 30, 33, 58, 32, 38, 29, 44, 35, 23, 27, 62, 66, 47, 24]
filtered_data = data.drop(index=excluded_indices)

#Extracting inputs and target
X = filtered_data.iloc[:, :4].values #inputs
Yaa = filtered_data.iloc[:, 4].values #acetic acid conc.
Ybm = filtered_data.iloc[:, 5].values #biomethane conc.
Yh = filtered_data.iloc[:, 6].values #hydrogen conc.

# Defining the cross-validation procedure
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the scoring metrics
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# defining a CV function to perform the cross-validation and print the results
def perform_cross_validation(model, X, y, y_name):
    # Setting the parameter grid
    param_grid = {
        'n_estimators': [10, 25, 50, 75, 100, 125, 150],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'colsample_bytree': [1.0],
        'alpha': [0, 0.1, 0.2, 0.3, 0.4],
        'lambda': [0, 0.05, 0.1, 0.15, 0.2]
    }

    # Performing the grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # defining best parameters
    print(f'{y_name} best parameters: {grid_search.best_params_}')

    # Perform cross-validation with the best model
    mse_scores = cross_val_score(best_model, X, y, cv=kf, scoring=mse_scorer)
    r2_scores = cross_val_score(best_model, X, y, cv=kf, scoring=r2_scorer)

    # Convert negative MSE to positive
    mse_scores = -mse_scores

    # Printing the mean results for the different folds
    if y_name == 'Acetic Acid':  # improves clarity of results in run window
        print(f'')

    print(f'{y_name} Results:')
    print(f'Mean MSE: {np.mean(mse_scores)}')
    print(f'Standard Deviation of MSE: {np.std(mse_scores)}')
    print(f'Mean RMSE: {np.sqrt(np.mean(mse_scores))}')
    print(f'Mean R²: {np.mean(r2_scores)}')
    print(f'Standard Deviation of R²: {np.std(r2_scores)}')
    print(f'')  # create a space to separate results

    return best_model  # Return the best model

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
best_model_aa = perform_cross_validation(XGBRegressor(objective='reg:squarederror'), X, Yaa, 'Acetic Acid')
best_model_bm = perform_cross_validation(XGBRegressor(objective='reg:squarederror'), X, Ybm, 'Biomethane')
best_model_h = perform_cross_validation(XGBRegressor(objective='reg:squarederror'), X, Yh, 'Hydrogen')

# Collect predictions for each output using the best model
actual_aa, predicted_aa = collect_predictions(best_model_aa, X, Yaa)
actual_bm, predicted_bm = collect_predictions(best_model_bm, X, Ybm)
actual_h, predicted_h = collect_predictions(best_model_h, X, Yh)

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

# Plotting for each output using the defined function
plot_predictions(actual_aa, predicted_aa, 'Actual AA Concentration', 'Predicted AA Concentration', 'Predicted vs Actual AA Concentration')
plot_predictions(actual_bm, predicted_bm, 'Actual Biomethane Concentration', 'Predicted Biomethane Concentration', 'Predicted vs Actual Biomethane Concentration')
plot_predictions(actual_h, predicted_h, 'Actual H Concentration', 'Predicted H Concentration', 'Predicted vs Actual H Concentration')

# SHAP Analysis
# Defining 'feature' (input) names
feature_names = ['Microbial Conc.', 'pH', 'EC', 'Average Current']

# Define the SHAP analysis function
def shap_analysis(model, X, feature_names, y_name):
    # Create SHAP explainer specific to XGBRegressor
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot for {y_name}')
    plt.show()

    # Bar plot of mean SHAP values
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'SHAP Bar Plot for {y_name}')
    plt.show()

# Perform SHAP analysis on the best models
shap_analysis(best_model_aa, X, feature_names, 'Acetic Acid')
shap_analysis(best_model_bm, X, feature_names, 'Biomethane')
shap_analysis(best_model_h, X, feature_names, 'Hydrogen')
