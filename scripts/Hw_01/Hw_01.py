import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Loading Iris dataset
def load_iris_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data_iris = pd.read_csv(url, header=None)
    data_iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]
    return data_iris


iris_data = load_iris_data()
print(iris_data.head())


# summary statistics using numpy
def s_statistics(_, predictors):
    mean = np.mean(iris_data[predictors], axis=0)
    minimum = np.min(iris_data[predictors], axis=0)
    maximum = np.max(iris_data[predictors], axis=0)
    quartiles = np.percentile(iris_data[predictors], [25, 50, 75])
    return mean, minimum, maximum, quartiles


# Creating Plots using plotly
def plot_data():
    # 1. Creating Scatter Plot
    scatter_plot: Figure = px.scatter(
        iris_data,
        x="sepal_width",
        y="sepal_length",
        # z="petal_length",
        color="class",
        title="3-D_Scatter_plot",
    )
    scatter_plot.show()

    # 2. creating Violin Plot
    violin_plot: Figure = px.violin(
        iris_data,
        y="class",
        x="petal_width",
        box=True,
        points="all",
        height=800,
        width=1000,
        title="Violin Plot",
    )
    violin_plot.show()

    # 3. Creating Box Plot
    box_plot: Figure = px.box(iris_data, color="class", points="all", title="Box Plot")
    box_plot.show()

    # 4. Creating Histogram
    his_plot: Figure = px.histogram(
        iris_data, x="sepal_length", color="class", title="Histogram"
    )
    his_plot.show()

    # 5. Scatter plot Matrix
    matrix_plot: Figure = px.scatter_matrix(
        iris_data,
        dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
        color="class",
    )
    matrix_plot.show()

    # Analyzing  and building models


def classifier():
    X = iris_data.iloc[:, [0, 1, 2, 3]]
    Y = iris_data.iloc[:, 4]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1
    )
    # Define pipeline
    # Random Forest
    pipeline = Pipeline(
        [
            ("standard_scale", StandardScaler()),
            ("random_forest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline.fit(X_train, Y_train)
    predict = pipeline.predict(X_test)
    accuracy = np.mean(predict == Y_test)
    print("Random Forest Accuracy:", accuracy)

    # DecisionTreeClassifier
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", DecisionTreeClassifier())]
    )
    pipeline.fit(X_train, Y_train)
    predict = pipeline.predict(X_test)
    accuracy = np.mean(predict == Y_test)
    print("Decision Tree Accuracy:", accuracy)

    # SVM
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", SVC())])
    pipeline.fit(X_train, Y_train)
    predict = pipeline.predict(X_test)
    accuracy = np.mean(predict == Y_test)
    print("SVM Accuracy:", accuracy)


def main():
    predictors = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    data = load_iris_data()
    mean, min, max, quartiles = s_statistics(data, predictors)
    print("Mean:", mean)
    print("Minimum:", min)
    print("Maximum:", max)
    print("Quartile:", quartiles)
    plot_data()
    classifier()


if __name__ == "__main__":
    main()
