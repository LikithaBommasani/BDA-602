import numpy as np
import pandas as pd
import plotly.graph_objects as go
from Load_Data import Load_Test_Datasets


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


def cont_cont_brute_force(df, cont_pred_list, response):
    # create a new DataFrame to store the binned values
    binned_df = df.copy()

    # iterate over each continuous predictor
    for i, cont_1 in enumerate(cont_pred_list):
        for j, cont_2 in enumerate(cont_pred_list[i + 1 :]):  # noqa: E203
            c1_binning, c2_binning = "Bins:" + cont_1, "Bins:" + cont_2
            binned_df[c1_binning] = (pd.cut(binned_df[cont_1], bins=10)).apply(
                lambda x: round(x.mid, 3)
            )
            binned_df[c2_binning] = (pd.cut(binned_df[cont_2], bins=10)).apply(
                lambda x: round(x.mid, 3)
            )
            # print(binned_df[c2_binning])

            # compute mean of response variable for each binning group
            mean = {response: np.mean}
            # length = {response: np.size}
            binned_df = (
                binned_df.groupby([c1_binning, c2_binning]).agg(mean).reset_index()
            )

            fig = go.Figure(
                data=go.Heatmap(
                    x=binned_df[c1_binning].astype(str).tolist(),
                    y=binned_df[c2_binning].astype(str).tolist(),
                    z=binned_df[response].tolist(),
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
            fig.show()


def main():
    test_datasets = Load_Test_Datasets()
    for test in test_datasets.Fetch_datasets():
        df, P_Predictors, R_Response = test_datasets.Fetch_sample_datasets(
            dataset_name="tips"
        )

    response_type = check_cat_cont_response(df, R_Response)
    cat_pred, cont_pred = check_predictor(df, P_Predictors)
    # x = df[P_Predictors]
    # y = df[R_Response]
    print(f"respsonse:{response_type}")
    print(f"cat_pred:{cat_pred}")
    print(f"cont_pred:{cont_pred}")

    cont_cont_brute_force(df, cont_pred, R_Response)


if __name__ == "__main__":
    main()
