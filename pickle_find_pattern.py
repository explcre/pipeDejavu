'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the provided data
with open("prof_database.pkl", "rb") as f:
    data = pickle.load(f)

# Function to extract X and Y values from the dataset
def extract_data(dataset, key):
    x = [point[0] for point in dataset[key]]
    y = [point[1] for point in dataset[key]]
    return np.array(x).reshape(-1, 1), np.array(y)

# Prepare data for linear regression
key = ('default', (4, 8))
all_gather_cost_dict = data[key].all_gather_cost_dict


for config, cost_data in all_gather_cost_dict.items():
    X, y = extract_data(all_gather_cost_dict, config)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)

    # Print accuracy results
    print(f"Configuration: {config}")
    print(f"R2 score: {r2:.2f}")
    print(f"Slope: {lr.coef_[0]}")
    print(f"Intercept: {lr.intercept_}\n")

    # Visualization
    plt.scatter(X, y, label=f"{config} R2: {r2:.2f}")
    plt.plot(X, y_pred)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Ranks")
plt.ylabel("Cost")

# Move the legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.title("All Gather Cost vs Ranks for Different Configurations")

# Save the plot
plt.savefig("all_gather_cost_vs_ranks.png", bbox_inches='tight')

# Show the plot
plt.show()
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from decimal import Decimal
from sklearn.preprocessing import StandardScaler
# Load the provided data
with open("prof_database.pkl", "rb") as f:
    data = pickle.load(f)

# Function to extract X and Y values from the dataset
def extract_data(dataset, key):
    x = [point[0] for point in dataset[key]]
    #x = [Decimal(point[0]) for point in dataset[key]]
    y = [point[1] for point in dataset[key]]
    #y = [Decimal(point[1]) for point in dataset[key]]
    
    # Replace extreme values with the maximum finite representable value for float64
    x = np.clip(x, np.finfo(np.float64).min, np.finfo(np.float64).max)
    y = np.clip(y, np.finfo(np.float64).min, np.finfo(np.float64).max)

     # Replace NaN values with the mean of the non-NaN elements in the array
    y = np.where(np.isnan(y), np.nanmean(y), y)
    
    return np.array(x).reshape(-1, 1), np.array(y)

# Function to filter out infinity or large values from X and y
def filter_data(X, y):
    finite_mask = np.isfinite(y)
    return X[finite_mask], y[finite_mask]

# Function to apply log transformation to y values
def apply_log_transform(y):
    return np.log(y)

# List of attributes
attributes = [
    'all_gather_cost_dict',
    'all_reduce_cost_dict',
    'all_to_all_cost_dict',
    'reduce_scatter_cost_dict',
    'available_memory_per_device',
    'dot_cost_dict',
    'conv_cost_dict',
    'op_cost_dict',
]

# Loop through all keys
for key in data.keys():
    # Loop through all attributes
    for attr in attributes:
        attribute_dict = getattr(data[key], attr)

        if not isinstance(attribute_dict, dict):
            continue

        for config, cost_data in attribute_dict.items():
            X, y = extract_data(attribute_dict, config)

            # Filter out infinity or large values from X and y
            #X, y = filter_data(X, y)
             # Apply log transformation to y values
            #y = apply_log_transform(y)#added
             # Initialize the scaler
            scaler = StandardScaler()


            y_norm = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
             # Linear Regression
            lr = LinearRegression()
            lr.fit(X, y_norm)
            y_pred_norm = lr.predict(X)

            # Rescale predictions back to original scale
            y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).reshape(-1)
            r2 = r2_score(y, y_pred)
            '''
            # Linear Regression
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            r2 = r2_score(y, y_pred)
            '''
            # Print accuracy results
            print(f"Key: {key}")
            print(f"Attribute: {attr}")
            print(f"Configuration: {config}")
            print(f"R2 score: {r2:.2f}")
            print(f"Slope: {lr.coef_[0]}")
            print(f"Intercept: {lr.intercept_}\n")

            # Visualization
            plt.scatter(X, y, label=f"{config} R2: {r2:.2f}")
            plt.plot(X, y_pred)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Ranks")
        plt.ylabel("Cost")

        # Move the legend outside of the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.title(f"{attr.capitalize()} vs Ranks for Different Configurations ({key})")

        # Save the plot
        plt.savefig(f"{attr}_vs_ranks_{key}.png", bbox_inches='tight')

        # Show the plot
        plt.show()

        # Clear the plot for the next attribute
        plt.clf()
