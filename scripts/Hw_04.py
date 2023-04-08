import random
from typing import List, Tuple

import numpy as np
import pandas
import plotly.graph_objs as go
import seaborn
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

# load data code, plots code and linearregression code is taken from the lecture slides
# Load data : https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/13/4
# Plots: https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/4/2
# linear Regression: https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/3/0


class Load_Test_Datasets:
    def __init__(self):
        self.seaborn_data_sets = ["mpg", "tips", "titanic"]
        self.sklearn_data_sets = ["diabetes", "breast_cancer"]
        self.all_data_sets = self.seaborn_data_sets + self.sklearn_data_sets

    Predictors_for_Titanic = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "embarked",
        "parch",
        "fare",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
        "class",
    ]

    def Fetch_datasets(self) -> List[str]:
        return self.all_data_sets

    def Fetch_sample_datasets(
        self, dataset_name: str = None
    ) -> Tuple[pandas.DataFrame, List[str], str]:

        if dataset_name is None:
            dataset_name = random.choice(self.all_data_sets)
        else:
            if dataset_name not in self.all_data_sets:
                raise Exception(f"Data set choice not valid: {dataset_name}")

        if dataset_name in self.seaborn_data_sets:
            if dataset_name == "mpg":
                dataset = seaborn.load_dataset(name="mpg").dropna().reset_index()
                P_Predictors = [
                    "cylinders",
                    "displacement",
                    "horsepower",
                    "weight",
                    "acceleration",
                    "origin",
                ]
                R_Response = "mpg"
            elif dataset_name == "tips":
                dataset = seaborn.load_dataset(name="tips").dropna().reset_index()
                P_Predictors = [
                    "total_bill",
                    "sex",
                    "smoker",
                    "day",
                    "time",
                    "size",
                ]
                R_Response = "tip"
            elif dataset_name in ["titanic", "titanic_2"]:
                dataset = seaborn.load_dataset(name="titanic").dropna()
                dataset["alone"] = dataset["alone"].astype(str)
                dataset["class"] = dataset["class"].astype(str)
                dataset["deck"] = dataset["deck"].astype(str)
                dataset["pclass"] = dataset["pclass"].astype(str)
                P_Predictors = self.Predictors_for_Titanic
                if dataset_name == "titanic":
                    R_Response = "survived"
                elif dataset_name == "titanic_2":
                    R_Response = "alive"
        elif dataset_name in self.sklearn_data_sets:
            if dataset_name == "diabetes":
                data_l = datasets.load_diabetes()
                dataset = pandas.DataFrame(data_l.data, columns=data_l.feature_names)
            elif dataset_name == "breast_cancer":
                data_l = datasets.load_breast_cancer()
                dataset = pandas.DataFrame(data_l.data, columns=data_l.feature_names)
            dataset["target"] = data_l.target
            P_Predictors = data_l.feature_names
            R_Response = "target"

        # Change category dtype to string
        for predictor in P_Predictors:
            if dataset[predictor].dtype in ["category"]:
                dataset[predictor] = dataset[predictor].astype(str)

        print(f"Data set selected: {dataset_name}")
        dataset.reset_index(drop=True, inplace=True)
        return dataset, P_Predictors, R_Response


def cont_resp_cat_predictor(df, pred, R_Response):
    fig = px.violin(df, x=pred, y=R_Response, box=True, points="all", color=pred)
    fig.update_layout(
        title=f"Continuous response Categorical predictor : {R_Response} vs {pred}",
        xaxis_title=pred,
        yaxis_title=R_Response,
        violinmode="group",
    )
    fig.show()
    # fig.write_html(file=f"{pred} vs {R_Response} plot.html", include_plotlyjs="cdn")


def cont_response_cont_predictor(df, pred, R_Response):
    fig = px.scatter(x=df[pred], y=df[R_Response])
    fig.update_layout(
        title="Continuous R_Response by Continuous Predictor",
        xaxis_title=f"Predictor - {pred}",
        yaxis_title=f"Response - {R_Response}",
    )
    fig.show()
    # fig.write_html(
    # file=f"{pred} vs {R_Response} plot.html", include_plotlyjs="cdn"
    # )  # From HW_01


def cat_resp_cont_predictor(df, pred, R_Response, Y):
    group_labels = ["0", "1"]

    x0 = df[Y == 0][pred]
    x1 = df[Y == 1][pred]

    # Check that x0 and x1 are not empty before creating hist_data and calling create_distplot
    if len(x0) > 0 and len(x1) > 0:
        hist_data = [x0, x1]
        colors = ["slategray", "magenta"]
        fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.4, colors=colors)
        fig_1.update_layout(
            title=f"Continuous Predictor by {R_Response}",
            xaxis_title=f"Predictor- {pred}",
            yaxis_title=f"Response- {R_Response}",
        )
    else:
        print(
            f"No data for {R_Response} {group_labels[0]} or {R_Response} {group_labels[1]}"
        )
    # fig_1.show()
    fig = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig.add_trace(
            go.Violin(
                x=np.repeat(curr_group, len(curr_hist)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title=R_Response,
        yaxis_title=pred,
    )
    fig.show()


def cat_response_cat_predictor(df, pred, R_Response):
    fig = px.density_heatmap(
        df, x=R_Response, y=pred
    )  # Reference: https://plotly.com/python-api-reference/generated/plotly.express.density_heatmap.html
    fig.update_xaxes(title=R_Response)
    fig.update_yaxes(title=pred)  # Reference : https://plotly.com/python/axes/
    fig.show()
    # fig.write_html(file=f"{pred} vs {R_Response} plot.html", include_plotlyjs="cdn")


def check_cat_cont_resposne(df, R_Response):
    unique_vals = df[R_Response].nunique()
    if unique_vals == 2:
        return "Cat"
    else:
        return "Cont"


def check_predictor(df, P_Predictors):
    cat_pred = []
    cont_pred = []
    for predictor in P_Predictors:
        if df[predictor].dtypes == "object" or df[predictor].dtypes == "bool":
            cat_pred.append(predictor)
        else:
            cont_pred.append(predictor)
    return cat_pred, cont_pred


def LinearRegression(dataset, response, pred):
    X = dataset[pred]
    y = dataset[response]

    for idx, column in enumerate(X):
        feature_name = pred[idx]
        predictor = statsmodels.api.add_constant(X[column])

        linear_regression_model = statsmodels.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(linear_regression_model_fitted.summary())

        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        print(t_value, p_value)
        fig = px.scatter(dataset, x=column, y=response)
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title=f"Variable:{response}",
        )
        # fig.show()
        # fig.write_html(
        #   file=f"{feature_name} vs {response} plot.html", include_plotlyjs="cdn"
        # )


def LogisticRegression(dataset, response, pred):
    X = dataset[pred]
    y = dataset[response]

    for idx, column in enumerate(X):
        feature_name = pred[idx]
        predictor = statsmodels.api.add_constant(X[column])
        # Reference: https://deepnote.com/@leung-leah/Untitled-Python-Project-3e2bf4ca-aa22-4756-8bde-17802d2628c4
        linear_regression_model = statsmodels.api.Logit(
            y.astype(float), predictor.astype(float)
        )
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(linear_regression_model_fitted.summary())

        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        print(t_value, p_value)
        fig = px.scatter(dataset, x=column, y=response)
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title=f"Variable:{response}",
        )
        # fig.show()
        # fig.write_html(
        #  file=f"{feature_name} vs {response} plot.html", include_plotlyjs="cdn"
    # )
    return t_value, p_value


# def table_unweighted(dataset, feature, response):
#     feature = dataset[feature]
#     response = dataset[response]
#     # print(feature)
#     feature = feature.iloc[:, 0]
#     # Reference: https://stackoverflow.com/questions/36814100/pandas-to-numeric-for-multiple-columns
#     feature = pandas.Series(feature).apply(
#         pandas.to_numeric, errors="coerce"
#     )  # Convert to numeric
#     feature = feature[~np.isnan(feature)]  # Remove NaN values
#
#     response = pandas.Series(response).apply(pandas.to_numeric, errors="coerce")
#     # print(response)
#     response = response[~np.isnan(response)]
#     bins = 10
#     bin_edges = np.linspace(feature.min(), feature.max(), bins + 1)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     bin_counts, _ = np.histogram(feature, bins=bin_edges)
#     bin_means = [
#         response[(feature >= bin_edges[i]) & (feature < bin_edges[i + 1])].mean()
#         for i in range(bins)
#     ]
#     bin_means = pandas.Series(bin_means).fillna(0).tolist()  # Replace NaN with 0
#     pop_mean = response.mean()
#     bin_diffs = [(bin_mean - pop_mean) ** 2 for bin_mean in bin_means]
#
#     table = pandas.DataFrame(
#         {
#             "LowerBin": bin_edges[:-1],
#             "UpperBin": bin_edges[1:],
#             "BinCenter": bin_centers,
#             "BinCount": bin_counts,
#             "BinMean": bin_means,
#             "PopulationMean": pop_mean,
#             "MeanSquareDiff": bin_diffs,
#         }
#     )
#
#     table = table.sort_values("MeanSquareDiff", ascending=False)
#
#     return table


# def weighted_table(table_unweighted, feature):
#     pop_mean = table_unweighted["PopulationMean"].iloc[0]
#     pop_prop = table_unweighted["BinCount"] / len(feature)
#     bin_means = table_unweighted["BinMean"]
#     w_msd = pop_prop * (bin_means - pop_mean) ** 2
#     weighted_table = table_unweighted.assign(
#         PopulationProportion=pop_prop, MeanSquaredDiff_Weighted=w_msd
#     )
#     weighted_table = weighted_table.sort_values(
#         "MeanSquaredDiff_Weighted", ascending=False
#     )
#
#     return weighted_table


def plot_continuous_predictor_and_categorical_response(df, predictors, response):
    """
    Cont Pred and Cat Response
    """
    continuous_predictors = []
    for predictor in predictors:
        if df[predictor].dtype == "float64" or df[predictor].dtype == "int64":
            continuous_predictors.append(predictor)
    if not continuous_predictors:
        print("No continuous predictors found.")
        return

    for predictor in continuous_predictors:
        predictor_data = np.array(df[predictor].astype(float).dropna().values)

    hist_count, bins = np.histogram(
        predictor_data,
        bins=10,
        range=(np.min(predictor_data), np.max(predictor_data)),
    )
    modified_bins = 0.5 * (bins[:-1] + bins[1:])

    s_predictor = df.query(f"{response} == 1")
    s_population = np.array(s_predictor[predictor])
    hist_s_population, _ = np.histogram(s_population, bins=bins)
    p_response = np.zeros_like(hist_count, dtype=float)
    for i in range(len(hist_count)):
        if hist_count[i] != 0:
            p_response[i] = hist_s_population[i] / hist_count[i]
        else:
            p_response[i] = np.nan

    s_response_rate = len(s_predictor) / len(df)
    s_response_arr = np.array([s_response_rate] * 10)

    figure = go.Figure(
        data=go.Bar(
            x=modified_bins,
            y=hist_count,
            name="Population",
            marker=dict(color="light blue"),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=modified_bins,
            y=p_response,
            yaxis="y2",
            name="Response",
            marker=dict(color="red"),
            connectgaps=True,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=modified_bins,
            y=s_response_arr,
            yaxis="y2",
            mode="lines",
            name="Population mean",
        )
    )

    figure.update_layout(
        title_text=f"<b> Mean of Response plot for {response} vs {predictor}</b>",
        legend=dict(orientation="v"),
        yaxis=dict(title=dict(text="Response"), side="left"),
        yaxis2=dict(
            title=dict(text="Population"),
            side="right",
            range=[-0.1, 1.2],
            overlaying="y",
            tickmode="auto",
        ),
    )

    figure.update_xaxes(title_text="Predictor Bin")

    figure.show()
    return


def plot_categorical_predictor_and_continuous_response(df, predictors, response):
    """
        Create a plot of categorical predictor variables and a continuous response variable.
    .
    """

    categorical_predictors = []

    for predictor in predictors:
        if df[predictor].dtype == "object" or df[predictor].dtype == "bool":
            categorical_predictors.append(predictor)
    if not categorical_predictors:
        print("No categorical predictors found.")
        return

    for predictor in categorical_predictors:
        predictor_data = df[predictor].dropna()
        categories = predictor_data.unique()
        _ = len(categories)

        # Calculate the population count and mean of the response variable for each category.
        hist_count = []
        s_response_arr = []
        p_response = []

        modified_bins = categories
        for i, category in enumerate(categories):
            mask = predictor_data == category
            count = np.sum(mask)
            hist_count.append(count)
            s_response_arr.append((df.loc[mask, response].sum()))
            p_response.append((df.loc[mask, response].sum()) / count)

        # Calculate the population mean of the response variable across all categories.
        total_count = np.sum(hist_count)
        s_response_mean = np.sum(np.array(s_response_arr)) / total_count
        temp_arr = [s_response_mean for i in range(len(p_response))]

        # Plot the population count and response mean for each category.
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=list(modified_bins),
                y=hist_count,
                name="Population",
                marker=dict(color="light blue"),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=list(modified_bins),
                y=p_response,
                yaxis="y2",
                name="Response",
                marker=dict(color="red"),
                connectgaps=True,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=list(modified_bins),
                y=temp_arr,
                yaxis="y2",
                mode="lines",
                name="Population mean",
            )
        )

        figure.update_layout(
            title_text=f"<b>Mean of Response plot for {response} vs {', '.join(categorical_predictors)}</b>",
            legend=dict(orientation="v"),
            yaxis=dict(title=dict(text="Response"), side="left"),
            yaxis2=dict(
                title=dict(text="Population"),
                side="right",
                overlaying="y",
                tickmode="auto",
            ),
        )

        figure.update_xaxes(title_text="Predictor Bin")
        figure.show()


def plot_continuous_predictor_and_continuous_response(df, predictor, response):
    """
    Cont Pred and Cont Response

    """
    predictor_data = np.array(df[predictor].astype(float).dropna().values)
    _ = np.array(df[response].astype(float).dropna().values)

    hist_count = []
    s_response_arr = []
    p_response = []

    hist, bins = np.histogram(
        predictor_data, bins=10, range=(np.min(predictor_data), np.max(predictor_data))
    )

    for i in range(len(bins) - 1):
        mask = (predictor_data >= bins[i]) & (predictor_data < bins[i + 1])
        count = np.sum(mask)
        hist_count.append(count)
        s_response_arr.append((df.loc[mask, response].sum()))
        if count > 0:
            p_response.append((df.loc[mask, response].sum()) / count)

        else:
            p_response.append(np.nan)

    modified_bins = 0.5 * (bins[:-1] + bins[1:])
    total_count = np.sum(hist_count)
    s_response_mean = np.sum(np.array(s_response_arr)) / total_count
    temp_arr = [s_response_mean for i in range(len(p_response))]

    figure = go.Figure(
        data=go.Bar(
            x=modified_bins,
            y=hist_count,
            name="Population",
            marker=dict(color="light blue"),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=modified_bins,
            y=p_response,
            yaxis="y2",
            name="Response",
            marker=dict(color="red"),
            connectgaps=True,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=modified_bins,
            y=temp_arr,
            yaxis="y2",
            mode="lines",
            name="Population mean",
        )
    )

    figure.update_layout(
        title_text=f"<b> Mean of Response plot for {response} vs {predictor}</b>",
        legend=dict(orientation="v"),
        yaxis=dict(title=dict(text="Response"), side="left"),
        yaxis2=dict(
            title=dict(text="Population"),
            side="right",
            overlaying="y",
            tickmode="auto",
        ),
    )

    figure.update_xaxes(title_text="Predictor Bin")

    figure.show()
    return


def plot_categorical_predictor_and_categorical_response(df, predictors, response):
    """
    Categorical Pred and Categorical Response
    """
    categorical_predictors = []
    for predictor in predictors:
        if df[predictor].dtype == "object" or "bool":
            categorical_predictors.append(predictor)
    if not categorical_predictors:
        print("No categorical predictors found.")
        return

    figure = go.Figure()

    for predictor in categorical_predictors:
        predictor_data = df[predictor].dropna()

        categories = predictor_data.unique()

        hist_count = np.zeros(len(categories))
        for i, category in enumerate(categories):
            hist_count[i] = np.sum(predictor_data == category)

        modified_bins = categories

        s_predictor = df.query(f"{response} == 1")
        s_population = s_predictor[predictor].fillna(0)
        hist_s_population = np.zeros(len(categories))
        for i, category in enumerate(categories):
            hist_s_population[i] = np.sum(s_population == category)
        p_response = np.zeros_like(hist_count, dtype=float)
        for i in range(len(hist_count)):
            if hist_count[i] != 0:
                p_response[i] = hist_s_population[i] / hist_count[i]
            else:
                p_response[i] = np.nan

        s_response_rate = len(s_predictor) / len(df)
        s_response_arr = np.array([s_response_rate] * len(categories))

        figure.add_trace(
            go.Bar(
                x=modified_bins,
                y=hist_count,
                name="Population",
                marker=dict(color="light blue"),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=modified_bins,
                y=p_response,
                yaxis="y2",
                name="Response",
                marker=dict(color="red"),
                connectgaps=True,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=modified_bins,
                y=s_response_arr,
                yaxis="y2",
                mode="lines",
                name="Population mean",
            )
        )

    figure.update_layout(
        title_text=f"<b>Mean of Response plot for {response} vs {', '.join(categorical_predictors)}</b>",
        legend=dict(orientation="v"),
        yaxis=dict(title=dict(text="Response"), side="left"),
        yaxis2=dict(
            title=dict(text="Population"),
            side="right",
            range=[-0.1, 1.2],
            overlaying="y",
            tickmode="auto",
        ),
    )

    figure.update_xaxes(title_text="Predictor Bin")

    figure.show()


def Random_Forest_Variable_importance(dataset, response, pred):
    # Reference1 : https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    # Reference2 : https: // sparkbyexamples.com / python / sort - python - dictionary /
    X = dataset[pred]
    y = dataset[response]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance = rf.feature_importances_

    feature_importance = dict(zip(X.columns, importance))

    sorted_imp_list = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    print(f"sorted_imp_list - {sorted_imp_list}")
    # Reference: https://dataindependent.com/pandas/pandas-rank-rank-your-data-pd-df-rank/
    var_importance_df = pandas.DataFrame(
        sorted_imp_list.items(), columns=["Variable", "Importance Score"]
    )
    var_importance_df["Rank"] = var_importance_df["Importance Score"].rank(
        ascending=False
    )
    var_importance_df = var_importance_df.sort_values("Rank")
    print(var_importance_df)
    fig = go.Figure(
        go.Bar(
            x=list(sorted_imp_list.keys()),
            y=list(sorted_imp_list.values()),
            orientation="v",
        )
    )

    fig.update_layout(
        title="Variable Importance",
        xaxis_title="Variable",
        yaxis_title="Importance Score",
    )
    # fig.show()
    # fig.write_html(file=f"{pred} vs {response} plot.html", include_plotlyjs="cdn")

    return sorted_imp_list, var_importance_df


if __name__ == "__main__":
    test_datasets = Load_Test_Datasets()
    for test in test_datasets.Fetch_datasets():
        df, P_Predictors, R_Response = test_datasets.Fetch_sample_datasets(
            dataset_name="titanic"
        )
    print(df.dtypes)
    response_type = check_cat_cont_resposne(df, R_Response)
    cat_pred, cont_pred = check_predictor(df, P_Predictors)
    y = df[R_Response]

    for pred in cat_pred:
        if response_type == "Cat":

            cat_response_cat_predictor(df, pred, R_Response)

        else:

            cont_resp_cat_predictor(df, pred, R_Response)

    for pred in cont_pred:

        if response_type == "Cat":

            cat_resp_cont_predictor(df, pred, R_Response, y)

        else:

            cont_response_cont_predictor(df, pred, R_Response)

    if cont_pred and response_type == "Cont":
        LinearRegression(df, R_Response, cont_pred)
    elif cont_pred and response_type == "Cat":
        LogisticRegression(df, R_Response, cont_pred)

    Random_Forest_Variable_importance(df, R_Response, cont_pred)
    # unweighted = table_unweighted(df, P_Predictors, R_Response)
    # print(unweighted)
    # weighted = weighted_table(unweighted, P_Predictors)
    # print(weighted)
    # print(cont_pred)

    for predictor in cont_pred:
        if response_type == "Cont":
            plot_continuous_predictor_and_continuous_response(df, predictor, R_Response)
        else:
            plot_continuous_predictor_and_categorical_response(
                df, [predictor], R_Response
            )

    for predictor in cat_pred:
        if response_type == "Cat":
            plot_categorical_predictor_and_categorical_response(
                df, [predictor], R_Response
            )
        else:
            plot_categorical_predictor_and_continuous_response(
                df, [predictor], R_Response
            )
