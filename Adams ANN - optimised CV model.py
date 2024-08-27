#importing required libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import shap

# Loading the data
file_path = r'Data_filepath'
data = pd.read_excel(file_path, skiprows=2)
data = data.iloc[:92, :]

# Dealing with the missing values
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Excluding all known anomalies plus additional ANN cleaning (47,
excluded_indices = [14, 30, 33, 58, 32, 38, 29, 44, 35, 23, 27, 62, 66, 47, 24]
filtered_data = data.drop(index=excluded_indices)

# Extracting inputs and target
X = filtered_data.iloc[:, :4].values  # inputs
Yaa = filtered_data.iloc[:, 4].values  # acetic acid conc.
Ybm = filtered_data.iloc[:, 5].values  # biomethane conc.
Yh = filtered_data.iloc[:, 6].values  # hydrogen conc.

# Standardizing inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Yaa = torch.tensor(Yaa, dtype=torch.float32).view(-1, 1)
Ybm = torch.tensor(Ybm, dtype=torch.float32).view(-1, 1)
Yh = torch.tensor(Yh, dtype=torch.float32).view(-1, 1)

# Define the neural network architecture
class ANN(nn.Module): #defines ANN class from PyTorch nn.module
    def __init__(self): #constructer method to initialise layers of the network
        super(ANN, self).__init__() #calls the constructor nn.module
        self.fc1 = nn.Linear(4, 15)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(15, 15)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(15, 1)  # Second hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first hidden layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)  # Output layer (no activation function)
        return x

# Train the model
def train_model(model, X_train, y_train, num_epochs=1250, learning_rate=0.001):
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_train)  # Forward pass
        loss = criterion(outputs, y_train)  # Calculate loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights
    return model

# Cross-validation procedure
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def perform_cross_validation(model, X, y, y_name, num_epochs=1000):
    mse_scores = []
    r2_scores = []
    actual_all = np.zeros(y.shape)
    predicted_all = np.zeros(y.shape)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = train_model(model, X_train, y_train, num_epochs)  # Train the model

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for evaluation
            y_pred = model(X_test)  # Get predictions
            mse = mean_squared_error(y_test.numpy(), y_pred.numpy())  # Calculate MSE
            r2 = r2_score(y_test.numpy(), y_pred.numpy())  # Calculate R²
            mse_scores.append(mse)
            r2_scores.append(r2)
            actual_all[test_index] = y_test.numpy()
            predicted_all[test_index] = y_pred.numpy()

    print_results(y_name, mse_scores, r2_scores)  # Print results
    return model, actual_all.flatten(), predicted_all.flatten()

# Print results function
def print_results(y_name, mse_scores, r2_scores):
    mse_scores = np.array(mse_scores)
    r2_scores = np.array(r2_scores)
    print(f'{y_name} Results:')
    print(f'Mean MSE: {np.mean(mse_scores)}')
    print(f'Standard Deviation of MSE: {np.std(mse_scores)}')
    print(f'Mean RMSE: {np.sqrt(np.mean(mse_scores))}')
    print(f'Mean R²: {np.mean(r2_scores)}')
    print(f'Standard Deviation of R²: {np.std(r2_scores)}\n')

# Plot predictions function
def plot_predictions(actual, predicted, xlabel, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Applying cross-validation and collecting predictions for each target variable
model_aa = ANN()
model_bm = ANN()
model_h = ANN()

best_model_aa, actual_aa, predicted_aa = perform_cross_validation(model_aa, X, Yaa, 'Acetic Acid')
best_model_bm, actual_bm, predicted_bm = perform_cross_validation(model_bm, X, Ybm, 'Biomethane')
best_model_h, actual_h, predicted_h = perform_cross_validation(model_h, X, Yh, 'Hydrogen')

# Plotting for each output using the defined function
plot_predictions(actual_aa, predicted_aa, 'Actual AA Concentration', 'Predicted AA Concentration',
                 'Predicted vs Actual AA Concentration')
plot_predictions(actual_bm, predicted_bm, 'Actual Biomethane Concentration', 'Predicted Biomethane Concentration',
                 'Predicted vs Actual Biomethane Concentration')
plot_predictions(actual_h, predicted_h, 'Actual H Concentration', 'Predicted H Concentration',
                 'Predicted vs Actual H Concentration')

## SHAP Analysis ##
# Defining 'feature' (input) names
feature_names = ['Microbial Conc.', 'pH', 'Electrical Conductivity', 'Average Current']

def SHAP_analysis_ann(model, X, feature_names, y_name):
    # Set the model to evaluation mode
    model.eval()

    # Create a SHAP GradientExplainer
    explainer = shap.GradientExplainer(model, X)

    # Compute SHAP values
    SHAP_values = explainer.shap_values(X)

    # Check the type and shape of SHAP_values
    print(f"SHAP_values type: {type(SHAP_values)}")
    print(f"SHAP_values shape: {SHAP_values.shape}")

    # Reshape SHAP_values from (73, 4, 1) to (73, 4)
    SHAP_values = SHAP_values.squeeze(axis=-1)

    # Summary plot
    shap.summary_plot(SHAP_values, X.detach().numpy(), feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot for {y_name}')
    plt.show()

    # Bar plot of mean SHAP values
    shap.summary_plot(SHAP_values, X.detach().numpy(), feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'SHAP Bar Plot for {y_name}')
    plt.show()


# Perform SHAP analysis on the best models
SHAP_analysis_ann(best_model_aa, X, feature_names, 'Acetic Acid')
SHAP_analysis_ann(best_model_bm, X, feature_names, 'Biomethane')
SHAP_analysis_ann(best_model_h, X, feature_names, 'Hydrogen')




