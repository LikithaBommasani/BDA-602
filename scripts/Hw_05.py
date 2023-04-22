import itertools
import os
import webbrowser

import numpy as np
import pandas
import pandas as pd
import plotly.graph_objs as go
import sqlalchemy
import statsmodels.api
from correlation_hw_05 import cat_cont_correlation_ratio, cat_correlation
from msd_hw_05 import (
    plot_categorical_predictor_and_categorical_response,
    plot_categorical_predictor_and_continuous_response,
    plot_continuous_predictor_and_categorical_response,
    plot_continuous_predictor_and_continuous_response,
)
from plotly import express as px
from plotly import figure_factory as ff
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def check_cat_cont_response(df, R_Response):
    # Tried the method suggested by professor
    # checked if the response had 2 unique values
    unique_vals = df[R_Response].nunique()
    if unique_vals == 2:
        return "Cat"
    else:
        return "Cont"


def check_predictor(df, P_Predictors):
    # split the predictors to cont and cat
    cat_pred = []
    cont_pred = []
    # loop through each predictor
    for predictor in P_Predictors:
        if df[predictor].dtypes == "object" or df[predictor].dtypes == "bool":
            cat_pred.append(predictor)
        else:
            cont_pred.append(predictor)
    return cat_pred, cont_pred


def cont_resp_cat_predictor(df, pred, R_Response):
    fig = px.violin(df, x=pred, y=R_Response, box=True, points="all", color=pred)
    fig.update_layout(
        title=f"Continuous response Categorical predictor : {R_Response} vs {pred}",
        xaxis_title=pred,
        yaxis_title=R_Response,
        violinmode="group",
    )
    print("yes")
    fig.show()


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
        return t_value, p_value


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


def create_correlation_table(correlation, correlation_method, table_title):
    table_data = [
        ["Variable 1", "Variable 2", "Correlation", "plot_var_1", "plot_var_2"]
    ]
    for i in range(len(correlation)):
        for j in range(len(correlation.columns)):
            variable_1 = correlation.index[i]
            variable_2 = correlation.columns[j]
            if variable_1 != variable_2:
                corr_value = correlation.iloc[i, j]
                plot_var_1 = (
                    f'<a target="_blank" rel="noopener noreferrer" href="./plots/'
                    f'{variable_1}-plot.html">Plot for {variable_1}</a>'
                )
                plot_var_2 = (
                    f'<a target="_blank" rel="noopener noreferrer" href="./plots/'
                    f'{variable_2}-plot.html">Plot for {variable_2}</a>'
                )
                table_data.append(
                    [variable_1, variable_2, corr_value, plot_var_1, plot_var_2]
                )

    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    df_sorted = df.sort_values(by="Correlation", ascending=False)

    html_table = df_sorted.to_html(render_links=True, escape=False)

    html_table = f"<h2>{table_title}</h2>" + html_table
    html_table = f'<div style="margin: 0 auto; width: fit-content;">{html_table}</div>'
    return html_table


def plot_correlation_matrix(correlation_matrix, title, xaxis_title, yaxis_title):
    # Reference: https://plotly.github.io/plotly.py-docs/generated/plotly.figure_factory.create_annotated_heatmap.html
    # Reference: https://en.ai-research-collection.com/plotly-heatmap/
    if not correlation_matrix.empty:
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=list(correlation_matrix.columns),
            y=list(correlation_matrix.index),
            colorscale="rdbu",
            zmin=-1,
            zmax=1,
            showscale=True,
        )

        # set plot title and axis labels
        # Reference: https://plotly.com/python/axes/
        fig.update_layout(
            title=title, xaxis=dict(title=xaxis_title), yaxis=dict(title=yaxis_title)
        )
        return fig

    else:
        print("Correlation matrix is empty.")


def cont_cont_correlation(X, cont):
    # Reference:https://www.geeksforgeeks.org/python-pandas-dataframe-corr/
    correlation_matrix = X[cont].corr(method="pearson").round(5)

    fig1 = plot_correlation_matrix(
        correlation_matrix, "Correlation Matrix Pearson", "Variable 1", "Variable 2"
    )

    # show  table
    if not correlation_matrix.empty:
        html2 = create_correlation_table(
            correlation_matrix, " Pearson ", " Correlation Pearson Table"
        )
        with open("my_report.html", "w") as f:
            f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write(html2)
    else:
        print("Correlation matrix is empty.")


def cat_cat_correlation(X, cat):
    variable_1 = []
    variable_2 = []
    Correlation_ratio = []
    for i in range(0, len(cat)):
        for j in range(0, len(cat)):
            variable_1.append(cat[i])
            variable_2.append(cat[j])
            Correlation_ratio.append(round(cat_correlation(X[cat[i]], X[cat[j]]), 5))

    # Create a dataframe with the categorical variable pairs and their correlation ratios
    # Reference : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    df = pd.DataFrame(
        {
            "Variable 1": variable_1,
            "Variable 2": variable_2,
            "Correlation Ratio": Correlation_ratio,
        }
    )

    # Pivot the dataframe to create a correlation matrix
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html
    correlation_matrix = df.pivot(
        index="Variable 1", columns="Variable 2", values="Correlation Ratio"
    )

    fig1 = plot_correlation_matrix(
        correlation_matrix,
        "Categorical-Categorical Correlation Matrix",
        "Variable 1",
        "Variable 2",
    )
    if not correlation_matrix.empty:
        html2 = create_correlation_table(
            correlation_matrix, " Tschuprow ", " Correlation Tschuprow Table"
        )

        with open("my_report.html", "a") as f:
            f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write(html2)
    else:
        print("Correlation matrix is empty.")
        with open("my_report.html", "a") as f:
            f.write("<p>No categorical predictors to analyze.</p>")


def cat_cont_correlation(X, cont, cat):
    # Reference:https://medium.com/@ritesh.110587/correlation-between-categorical-variables-63f6bd9bf2f7
    variable_1 = []
    variable_2 = []
    Correlation_ratio = []
    for i in range(0, len(cat)):
        for j in range(0, len(cont)):
            variable_1.append(cat[i])
            variable_2.append(cont[j])
            Correlation_ratio.append(
                round(cat_cont_correlation_ratio(X[cat[i]], X[cont[j]]), 5)
            )

    # Create a dataframe with the categorical variable pairs and their correlation ratios
    df = pd.DataFrame(
        {
            "Variable 1": variable_1,
            "Variable 2": variable_2,
            "Correlation Ratio": Correlation_ratio,
        }
    )

    # Pivot the dataframe to create a correlation matrix
    correlation_matrix = df.pivot(
        index="Variable 1", columns="Variable 2", values="Correlation Ratio"
    )

    fig1 = plot_correlation_matrix(
        correlation_matrix,
        "Categorical-Continuous Correlation Matrix",
        "Variable 1",
        "Variable 2",
    )

    if not correlation_matrix.empty:
        html2 = create_correlation_table(
            correlation_matrix, " Ratio Table", " Correlation Ratio Table"
        )
        with open("my_report.html", "a") as f:
            f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write(html2)
    else:
        print("Correlation matrix is empty.")


def brute_create_correlation_table(pred_list, table_title):
    brute_table_data = [["plot_var_1", "plot_var_2", "plot"]]
    for i in range(len(pred_list)):
        for j in range(len(pred_list)):
            if i < j:
                variable_1 = pred_list[i]
                variable_2 = pred_list[j]
                plot_var_1 = (
                    f'<a target="_blank" rel="noopener noreferrer" href="./plots/'
                    f'{variable_1}-plot.html">Plot for {variable_1}</a>'
                )
                plot_var_2 = (
                    f'<a target="_blank" rel="noopener noreferrer" href="./plots/'
                    f'{variable_2}-plot.html">Plot for {variable_2}</a>'
                )
                plot = (
                    f'<a target="_blank" rel="noopener noreferrer" href="./bruteforce_plots/'
                    f'{variable_1}-{variable_2}-plot.html">View Plot</a>'
                )
                brute_table_data.append([plot_var_1, plot_var_2, plot])

    df = pd.DataFrame(brute_table_data[1:], columns=brute_table_data[0])
    brute_html_table = df.to_html(render_links=True, escape=False)
    brute_html_table = f"<h2>{table_title}</h2>" + brute_html_table
    brute_html_table = (
        f'<div style="margin: 0 auto; width: fit-content;">{brute_html_table}</div>'
    )
    return brute_html_table


def cat_cat_brute_force(df, cat_pred_list, response):
    # create a new DataFrame to store the binned values
    binned_df = df.copy()

    # iterate over each pair of categorical predictors
    for i, cat_1 in enumerate(cat_pred_list):
        for j, cat_2 in enumerate(cat_pred_list):
            if i < j:
                c1_binned, c2_binned = "binned:" + cat_1, "binned:" + cat_2
                binned_df[c1_binned] = binned_df[cat_1]
                binned_df[c2_binned] = binned_df[cat_2]

                # compute mean of response variable for each binned group
                binned_df_grouped = (
                    binned_df.groupby([c1_binned, c2_binned])[response]
                    .agg(["count", "mean"])
                    .reset_index()
                )

                # calculate the population mean of the response variable
                pop_mean = df[response].mean()

                # calculate unweighted mean square deviation
                binned_df_grouped["unweighted_msd"] = (
                    (binned_df_grouped["mean"] - pop_mean) ** 2
                ) * binned_df_grouped["count"]
                unweighted_msd = (
                    binned_df_grouped["unweighted_msd"].sum()
                    / binned_df_grouped["count"].sum()
                )
                print(unweighted_msd)

                # calculate weighted mean square deviation
                binned_df_grouped["weighted_count"] = binned_df_grouped["count"] / len(
                    df
                )
                binned_df_grouped["weighted_msd"] = (
                    (binned_df_grouped["mean"] - pop_mean) ** 2
                ) * binned_df_grouped["weighted_count"]
                weighted_msd = binned_df_grouped["weighted_msd"].sum()
                print(weighted_msd)

                fig1 = go.Figure(
                    data=go.Heatmap(
                        x=binned_df_grouped[c1_binned].astype(str).tolist(),
                        y=binned_df_grouped[c2_binned].astype(str).tolist(),
                        z=binned_df_grouped["mean"].tolist(),
                        colorscale="rdbu",
                        colorbar=dict(title="Mean(" + response + ")"),
                    )
                )

                fig1.update_layout(
                    title="Correlation heatmap for " + cat_1 + " and " + cat_2,
                    xaxis=dict(title=cat_1),
                    yaxis=dict(title=cat_2),
                    width=800,
                    height=600,
                )

                if not os.path.isdir("bruteforce_plots"):
                    os.mkdir("bruteforce_plots")
                file_path = f"bruteforce_plots/{cat_1}-{cat_2}-plot.html"

                fig1.write_html(file=file_path, include_plotlyjs="cdn")

                # html = brute_create_correlation_table(
                #     cat_pred_list, " Categorical/Categorical Brute_Force Table"
                # )
                #
                # with open("my_report.html", "a") as f:
                #     f.write(html)


def cont_cont_brute_force(df, cont_pred_list, response):
    # create a new DataFrame to store the binned values
    binned_df = df.copy()

    # iterate over each pair of continuous predictors
    for i, cont_1 in enumerate(cont_pred_list):
        for j, cont_2 in enumerate(cont_pred_list):
            if i < j:
                c1_binned, c2_binned = "binned_" + cont_1, "binned_" + cont_2
                binned_df[c1_binned] = (pd.cut(binned_df[cont_1], bins=10)).apply(
                    lambda x: round(x.mid, 3)
                )
                binned_df[c2_binned] = (pd.cut(binned_df[cont_2], bins=10)).apply(
                    lambda x: round(x.mid, 3)
                )

                # compute mean of response variable for each binned group
                binned_df_grouped = (
                    binned_df.groupby([c1_binned, c2_binned])[response]
                    .agg(["count", "mean"])
                    .reset_index()
                )

                # calculate the population mean of the response variable
                pop_mean = df[response].mean()

                # calculate unweighted mean square deviation
                binned_df_grouped["unweighted_msd"] = (
                    (binned_df_grouped["mean"] - pop_mean) ** 2
                ) / 100
                unweighted_msd = binned_df_grouped["unweighted_msd"].sum()

                print(unweighted_msd)

                # calculate weighted mean square deviation
                binned_df_grouped["weighted_count"] = binned_df_grouped["count"] / len(
                    df
                )
                binned_df_grouped["weighted_msd"] = (
                    (binned_df_grouped["mean"] - pop_mean) ** 2
                ) * binned_df_grouped["weighted_count"]
                weighted_msd = binned_df_grouped["weighted_msd"].sum()

                print(weighted_msd)

                fig = go.Figure(
                    data=go.Heatmap(
                        x=binned_df_grouped[c1_binned].astype(str).tolist(),
                        y=binned_df_grouped[c2_binned].astype(str).tolist(),
                        z=binned_df_grouped["mean"].tolist(),
                        colorscale="rdbu",
                        colorbar=dict(title="Mean(" + response + ")"),
                    )
                )

                fig.update_layout(
                    title="Correlation heatmap for " + cont_1 + " and " + cont_2,
                    xaxis=dict(title=cont_1),
                    yaxis=dict(title=cont_2),
                    width=800,
                    height=600,
                )
                if not os.path.isdir("bruteforce_plots"):
                    os.mkdir("bruteforce_plots")
                file_path = f"bruteforce_plots/{cont_1}-{cont_2}-plot.html"

                fig.write_html(file=file_path, include_plotlyjs="cdn")

                # html = brute_create_correlation_table(
                #     cont_pred_list, " Continuous/Continuous Brute_Force Table", unweighted_msd, weighted_msd
                # )
                html = brute_create_correlation_table(
                    cont_pred_list, " Continuous/Continuous Brute_Force Table"
                )

                with open("my_report.html", "a") as f:
                    f.write(html)


def cont_cat_brute_force(df, cont_pred_list, cat_pred_list, response):
    # create a new DataFrame to store the binned values
    binned_df = df.copy()

    # create a list of all possible pairs of continuous and categorical predictors
    pred_pairs = list(itertools.product(cont_pred_list, cat_pred_list))

    # iterate over each pair of predictors
    for cont_pred, cat_pred in pred_pairs:
        # create binned versions of the predictors
        cont_binned = "binned_" + cont_pred
        binned_df[cont_binned] = pd.cut(binned_df[cont_pred], bins=10).apply(
            lambda x: round(x.mid, 3)
        )

        # group by the binned predictors and compute the mean of the response variable
        mean_response = (
            binned_df.groupby([cont_binned, cat_pred])[response].mean().reset_index()
        )

        # calculate the population mean of the response variable
        pop_mean = df[response].mean()

        # calculate unweighted mean square deviation
        mean_response["unweighted_msd"] = (
            (mean_response[response] - pop_mean) ** 2
        ) / len(mean_response)
        unweighted_msd = mean_response["unweighted_msd"].sum()
        print("Unweighted MSD:", unweighted_msd)

        # calculate weighted mean square deviation
        mean_response["weighted_count"] = mean_response.groupby(
            [cont_binned, cat_pred]
        )[response].transform("count") / len(df)
        mean_response["weighted_msd"] = (
            (mean_response[response] - pop_mean) ** 2
        ) * mean_response["weighted_count"]
        weighted_msd = mean_response["weighted_msd"].sum()
        print("Weighted MSD:", weighted_msd)

        # plot the heatmap
        fig = go.Figure(
            data=go.Heatmap(
                x=mean_response[cat_pred],
                y=mean_response[cont_binned],
                z=mean_response[response],
                colorscale="rdbu",
                colorbar=dict(title="Mean(" + response + ")"),
            )
        )

        fig.update_layout(
            title="Correlation heatmap for " + cont_pred + " and " + cat_pred,
            xaxis=dict(title=cat_pred),
            yaxis=dict(title=cont_pred),
            width=800,
            height=600,
        )

        if not os.path.isdir("bruteforce_plots"):
            os.mkdir("bruteforce_plots")
        file_path = f"bruteforce_plots/{cont_pred}-{cat_pred}-plot.html"
        fig.write_html(file=file_path, include_plotlyjs="cdn")

    html_table = cont_cat_brute_create_correlation_table(
        cont_pred_list, cat_pred_list, " Continuous/Categorical Brute_Force Table"
    )

    with open("my_report.html", "a") as f:
        f.write(html_table)


def cont_cat_brute_create_correlation_table(cont_pred_list, cat_pred_list, table_title):
    table_data = [["Variable 1", "Variable 2", "plot"]]
    for cont_pred in cont_pred_list:
        for cat_pred in cat_pred_list:
            plot_var_1 = (
                f'<a target="_blank" rel="noopener noreferrer" href="./plots/'
                f'{cont_pred}-plot.html">Plot for {cont_pred}</a>'
            )
            plot_var_2 = (
                f'<a target="_blank" rel="noopener noreferrer" href="./plots/'
                f'{cat_pred}-plot.html">Plot for {cat_pred}</a>'
            )
            plot = (
                f'<a target="_blank" rel="noopener noreferrer" href="./bruteforce_plots/'
                f'{cont_pred}-{cat_pred}-plot.html">View Plot</a>'
            )
            table_data.append([plot_var_1, plot_var_2, plot])

    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    html_table = df.to_html(render_links=True, escape=False)
    html_table = f"<h2>{table_title}</h2>" + html_table
    html_table = f'<div style="margin: 0 auto; width: fit-content;">{html_table}</div>'
    return html_table


def classifier(df, P_Predictors, R_Response):
    # Reference From hw_01
    x = df[P_Predictors]
    y = df[R_Response]

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

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
    db_user = "root"
    db_pass = "newrootpassword"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://" f"{db_user}:{db_pass}@{db_host}/{db_database}"
    )  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """ SELECT * FROM features_ratio """
    df = pandas.read_sql_query(query, sql_engine)
    print(df.head())
    print(df.dtypes)
    R_Response = "HomeTeamWins"
    ignore_columns = ["game_id", "home_team_id", "away_team_id"]
    P_Predictors = [
        x for x in df.columns if x != R_Response and x not in ignore_columns
    ]
    # print(R_Response)
    # print(P_Predictors)
    response_type = check_cat_cont_response(df, R_Response)
    cat_pred, cont_pred = check_predictor(df, P_Predictors)
    print(f"respsonse:{response_type}")
    print(f"cat_pred:{cat_pred}")
    print(f"cont_pred:{cont_pred}")
    x = df[P_Predictors]
    y = df[R_Response]
    print(y)
    # Plots for cat, cont
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
    # Linear and logistic regression
    if cont_pred and response_type == "Cont":
        LinearRegression(df, R_Response, cont_pred)
    elif cont_pred and response_type == "Cat":
        LogisticRegression(df, R_Response, cont_pred)
    # Random Forest For Variable Importance Ranking
    Random_Forest_Variable_importance(df, R_Response, cont_pred)
    # Correlation
    cont_cont_correlation(x, cont_pred)
    cat_cat_correlation(x, cat_pred)
    cat_cont_correlation(x, cont_pred, cat_pred)
    # Brute_Force
    cont_cont_brute_force(df, cont_pred, R_Response)
    cat_cat_brute_force(df, cat_pred, R_Response)
    cont_cat_brute_force(df, cont_pred, cat_pred, R_Response)
    # Mean_Response_Plots
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
    # Models
    classifier(df, P_Predictors, R_Response)


if __name__ == "__main__":
    main()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    webbrowser.open("file://" + os.path.join(BASE_DIR, "my_report.html"))
