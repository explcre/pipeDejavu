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
