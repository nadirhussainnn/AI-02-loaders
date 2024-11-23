import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def lineplot(x,y):
    plt.plot(x,y)
    plt.title("Population growth over time")
    plt.xlabel("Time in years")
    plt.ylabel("Count")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.yaxis.get_major_formatter().set_scientific(False)  # Disable scientific notation
    plt.grid()
    plt.show()
    

#  We show bar chart when we have limited categories that we can display on x-axis, typically x should have discrete values
# we use histogram when we have continuous nominal type i.e Age

# Showing ratio of students over grades
def barchart():
    categories = ['A-1', 'A', 'B-1','B', 'C', 'D', 'F']
    values = [7, 14, 23, 12, 9, 6, 1]

    plt.bar(categories, values, color='#00bbf9')
    plt.title("Vertical Bar Chart")
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.show()

# Let's say we have this data
'''
Gender  Age     Income (in $)
M       23      50000
F       28      49000
F       65      34000
F       87      67000
M       36      11200
M       78      45600
M       56      99000
F       45      87000
M       42      45000
F       90      34000
F       32      23000
'''

# Now that we want to display bar chart to show income (y) on age (x). But as x is continuous variable, we can not categorize it therefore we use "Histogram" that is type of bar chart and it divides input variable into ranges/bins

def histogram():
    
    age = [23, 28, 65, 87, 36, 78, 56, 45, 42, 90, 32]
    income = [50000, 49000, 34000, 67000, 11200, 45600, 99000, 87000, 45000, 34000, 23000]


    plt.hist(age, bins=6, weights=income, color='orange', edgecolor='black', alpha=0.7)
    plt.title("Histogram of Income vs. Age")
    plt.xlabel("Age Bins")
    plt.ylabel("Income ($)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# for knn, k-means we may use this to see data
def scatter():

    age = [23, 28, 65, 87, 36, 78, 56, 45, 42, 90, 32]
    income = [50000, 49000, 34000, 67000, 11200, 45600, 99000, 87000, 45000, 34000, 23000]

    # Plot
    plt.scatter(age, income, color='green', edgecolor='black', )
    plt.title("Income vs. Age")
    plt.xlabel("Age")
    plt.ylabel("Income ($)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# show ratio of each category out of 100
def pie_chart():
    colors = ['red', 'yellow', 'pink', 'brown', 'green', 'magenta', 'aqua']
    explode = (0, 0.1, 0, 0,0,0,0)  # Explode the first slice

    categories = ['A-1', 'A', 'B-1','B', 'C', 'D', 'F']
    values = [7, 14, 23, 12, 9, 6, 1]

    plt.pie(values, labels=categories, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140)
    plt.title("Pie Chart")
    plt.show()

# Also called box plot, that is useful for outlier detection. It shows min, max, median, Q1, Q2 and Q3 quartiles
def whisker_plot():
    # last 10 transactions of user
    data = [10000, 5000, 9500, 6000, 7000, 2400, 1500, 1000, 50000, 2350]

    plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'))
    plt.title("Box Plot of Transactions")
    plt.xlabel("Transaction Amount ($)")
    plt.grid(True, axis='x')
    plt.show()

# Show intensity of values in matrics using colors. We may use it to find correlation of features
# the more the features correlated, the more they tend to move in same direction. +ve means same direction, -ve opposite. 1 means perfect +ve, -1 means perfect -ve

def heatmap():
# Set random seed for reproducibility
    np.random.seed(42)

    # Generate dummy stock data (30 days of prices for 5 stocks)
    n_days = 30
    stock_data = {
        'AAPL': np.random.normal(loc=150, scale=10, size=n_days),  # Apple
        'MSFT': np.random.normal(loc=300, scale=15, size=n_days),  # Microsoft
        'TSLA': np.random.normal(loc=700, scale=25, size=n_days),  # Tesla
        'GOOG': np.random.normal(loc=2800, scale=40, size=n_days),  # Google
        'AMZN': np.random.normal(loc=3500, scale=60, size=n_days),  # Amazon
    }

    # Convert the stock data dictionary to a 2D NumPy array
    stock_values = np.array(list(stock_data.values()))  # Shape: (5, 30)

    # Compute the correlation matrix using np.corrcoef (returns a 2D matrix)
    correlation_matrix = np.corrcoef(stock_values)

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, xticklabels=list(stock_data.keys()), yticklabels=list(stock_data.keys()))

    # Title and labels
    plt.title('Stock Prices of (Apple, Microsoft, Tesla, Google, Amazon)')
    plt.xlabel('Stock Symbols')
    plt.ylabel('Stock Symbols')

    # Show plot
    plt.show()


def error_bars():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    errors = [0.5, 0.2, 0.3, 0.4, 0.1]

    plt.errorbar(x, y, yerr=errors, fmt='o', color='green', ecolor='red', capsize=5)
    plt.title("Error Bars")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

def threeD_scatter():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    z = [1, 2, 3, 4, 5]

    ax.scatter(x, y, z, color='blue', marker='o')
    ax.set_title("3D Scatter Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()

def stacked_bar():
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    values1 = [5, 7, 3, 6,7,8,9]
    values2 = [3, 8, 6, 7,9,4,5]

    plt.bar(categories, values1, color='#00bbf9', label='Group 1')
    plt.bar(categories, values2, bottom=values1, color='#ff595e', label='Group 2')
    plt.title("Stacked Bar Chart")
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.legend()
    plt.show()
