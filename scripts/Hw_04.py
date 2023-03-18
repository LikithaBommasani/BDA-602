import random
from typing import List, Tuple

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
    # fig.show()


def cont_response_cont_predictor(df, pred, R_Response):
    fig = px.scatter(x=df[pred], y=df[R_Response])
    fig.update_layout(
        title="Continuous R_Response by Continuous Predictor",
        xaxis_title=f"Predictor - {pred}",
        yaxis_title=f"Response - {R_Response}",
    )
    # fig.show()


def cat_resp_cont_predictor(df, pred, R_Response):
    group_labels = df[R_Response].unique()
    x0 = df[df[R_Response] == group_labels[0]][pred]
    x1 = df[df[R_Response] == group_labels[1]][pred]
    hist_data = [x0, x1]

    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.4)
    fig.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title=f"Predictor- {pred}",
        yaxis_title=f"Response- {R_Response}",
    )
    # fig.show()


def cat_response_cat_predictor(df, pred, R_Response):
    fig = px.density_heatmap(
        df, pred, R_Response
    )  # Reference: https://plotly.com/python-api-reference/generated/plotly.express.density_heatmap.html
    fig.update_xaxes(title=pred)
    fig.update_yaxes(title=R_Response)  # Reference : https://plotly.com/python/axes/
    # fig.show()


def check_cat_cont_resposne(
    df, R_Response
):  # reference: https://stackoverflow.com/questions/26924904/check-if-dataframe-column-is-categorical
    if (df[R_Response].dtypes) == "object":
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
    fig.show()
    return sorted_imp_list, var_importance_df


if __name__ == "__main__":
    test_datasets = Load_Test_Datasets()
    for test in test_datasets.Fetch_datasets():
        df, P_Predictors, R_Response = test_datasets.Fetch_sample_datasets(
            dataset_name="titanic"
        )
    # Comment this line for other datasets
    df["survived"] = df["survived"].astype("object")
    print(df.dtypes)
    response_type = check_cat_cont_resposne(df, R_Response)
    cat_pred, cont_pred = check_predictor(df, P_Predictors)

    for pred in cat_pred:
        if response_type == "Cat":

            cat_response_cat_predictor(df, pred, R_Response)
        else:

            cont_resp_cat_predictor(df, pred, R_Response)

    for pred in cont_pred:

        if response_type == "Cat":

            cat_resp_cont_predictor(df, pred, R_Response)

        else:

            cont_response_cont_predictor(df, pred, R_Response)

    if cont_pred and response_type == "Cont":
        LinearRegression(df, R_Response, cont_pred)
    elif cont_pred and response_type == "Cat":
        LogisticRegression(df, R_Response, cont_pred)

    Random_Forest_Variable_importance(df, R_Response, cont_pred)
