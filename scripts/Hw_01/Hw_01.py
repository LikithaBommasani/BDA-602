import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    quartile_list = []
    for pred in predictors:
        quartiles = np.percentile(iris_data[pred], [25, 50, 75])
        quartile_list.append((pred, quartiles))
    return mean, minimum, maximum, quartile_list


# Creating Plots using plotly
def plot_data():
    # 1. Creating Scatter Plot
    scatter_plot: Figure = px.scatter(
        iris_data,
        x="sepal_width",
        y="sepal_length",
        size="petal_length",
        color="class",
        title="Scatter_plot",
    )
    scatter_plot.show()
    scatter_plot.write_html(file="Basic_plots.html", include_plotlyjs="cdn")
    # 2. creating Violin Plot
    violin_plot: Figure = px.violin(
        iris_data,
        y="class",
        x="petal_width",
        box=True,
        color="class",
        points="all",
        height=800,
        width=1000,
        title="Violin Plot",
    )
    violin_plot.show()
    violin_plot.write_html(file="Basic_plots.html", include_plotlyjs="cdn")
    # 3. Creating Box Plot
    box_plot: Figure = px.box(iris_data, color="class", points="all", title="Box Plot")
    box_plot.show()
    box_plot.write_html(file="Basic_plots.html", include_plotlyjs="cdn")
    # 4. Creating Histogram
    his_plot: Figure = px.histogram(
        iris_data, x="sepal_length", color="class", title="Histogram"
    )
    his_plot.show()
    his_plot.write_html(file="Basic_plots.html", include_plotlyjs="cdn")
    # 5. Scatter plot Matrix
    matrix_plot: Figure = px.scatter_matrix(
        iris_data,
        dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
        color="class",
    )
    matrix_plot.show()
    matrix_plot.write_html(file="Basic_plots.html", include_plotlyjs="cdn")

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

    # Difference with Mean of Response


def plot_histogram_and_response_rate(data, predictor, species, response_col="class"):
    predictor_data = data[[predictor, response_col]]

    total_count = np.array(predictor_data[predictor])
    hist_count, bins = np.histogram(
        total_count,
        bins=10,
        range=(np.min(total_count), np.max(total_count)),
    )
    modified_bins = 0.5 * (bins[:-1] + bins[1:])

    species_predictor = predictor_data.loc[predictor_data[response_col] == species]
    species_population = np.array(species_predictor[predictor])
    hist_species_population, _ = np.histogram(species_population, bins=bins)

    response_count = np.zeros(len(hist_count))
    for i in range(len(hist_count)):
        if hist_count[i] != 0:
            response_count[i] = hist_species_population[i] / hist_count[i]

    species_response_rate = len(data.loc[data[response_col] == species]) / len(data)
    species_response_arr = np.array([species_response_rate] * len(modified_bins))

    figure = go.Figure(
        data=go.Bar(
            x=modified_bins,
            y=hist_count,
            name=f"{predictor}",
            marker=dict(color="blue"),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=modified_bins,
            y=response_count,
            yaxis="y2",
            name="Response",
            marker=dict(color="red"),
            connectgaps=True,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=modified_bins,
            y=species_response_arr,
            yaxis="y2",
            mode="lines",
            name=f"{species}",
        )
    )

    figure.update_layout(
        title_text=f"<b> Mean of Response plot for {species} vs {predictor}</b>",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Count"),
            side="left",
            range=[0, 30],
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            range=[-0.1, 1.2],
            overlaying="y",
            tickmode="auto",
        ),
    )

    figure.update_xaxes(title_text=f"{predictor}")

    figure.show()
    figure.write_html(file="Mean_Response_plots.html", include_plotlyjs="cdn")


def plot_all_species(iris_data, predictors, response="class"):
    species_list = iris_data[response].unique()
    for species in species_list:
        for predictor in predictors:
            plot_histogram_and_response_rate(iris_data, predictor, species, response)


def main():
    predictors = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    data = load_iris_data()
    mean, min, max, quartile_list = s_statistics(data, predictors)
    print("Mean:", mean)
    print("Minimum:", min)
    print("Maximum:", max)
    print("Quartile:", quartile_list)
    plot_data()
    classifier()
    plot_all_species(iris_data, predictors, response="class")


if __name__ == "__main__":
    main()
