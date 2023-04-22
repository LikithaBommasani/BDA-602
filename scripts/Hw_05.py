import sys

# import numpy as np
import pandas

# import plotly.graph_objs as go
import sqlalchemy
from plotly import express as px

# from plotly import figure_factory as ff


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


def main():
    db_user = "root"
    db_pass = "newrootpassword"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://" f"{db_user}:{db_pass}@{db_host}/{db_database}"
    )  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
             SELECT * FROM features_ratio; """
    df = pandas.read_sql_query(query, sql_engine)
    # print(df.head())
    # print(df.dtypes)
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
    y = df[R_Response]
    print(y)


if __name__ == "__main__":
    sys.exit(main())
