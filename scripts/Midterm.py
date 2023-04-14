import os
import webbrowser

import pandas as pd
from correlation import cat_cont_correlation_ratio, cat_correlation
from Load_Data import Load_Test_Datasets
from msd import (
    plot_categorical_predictor_and_categorical_response,
    plot_categorical_predictor_and_continuous_response,
    plot_continuous_predictor_and_categorical_response,
    plot_continuous_predictor_and_continuous_response,
)
from plotly import figure_factory as ff


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

    #
    # table = go.Table(
    #     header=dict(
    #         values=df_sorted.columns, fill_color="paleturquoise", align="center"
    #     ),
    #     cells=dict(
    #         values=[
    #             df_sorted["Variable 1"],
    #             df_sorted["Variable 2"],
    #             df_sorted["Correlation"],
    #             df_sorted["plot_var_1"],
    #             df_sorted["plot_var_2"],
    #         ],
    #         fill_color="lavender",
    #         align="center",
    #     ),
    # )
    #
    # fig = go.Figure(data=[table])
    # fig.update_layout(title=table_title)
    # return fig


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

        # show plot
        # fig.show()
        # fig.write_html(file=f"{title} plot.html", include_plotlyjs="cdn")
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


def main():
    test_datasets = Load_Test_Datasets()
    for test in test_datasets.Fetch_datasets():
        df, P_Predictors, R_Response = test_datasets.Fetch_sample_datasets(
            dataset_name="tips"
        )

    response_type = check_cat_cont_response(df, R_Response)
    cat_pred, cont_pred = check_predictor(df, P_Predictors)
    x = df[P_Predictors]
    _ = df[R_Response]
    print(f"respsonse:{response_type}")
    print(f"cat_pred:{cat_pred}")
    print(f"cont_pred:{cont_pred}")
    # print(df.adult_male)
    cont_cont_correlation(x, cont_pred)
    cat_cat_correlation(x, cat_pred)
    cat_cont_correlation(x, cont_pred, cat_pred)

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


if __name__ == "__main__":
    main()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    webbrowser.open("file://" + os.path.join(BASE_DIR, "my_report.html"))
