import sys
from random import random
from typing import List, Tuple

import pandas as pd
import seaborn as sns
from plotly import express as px
from sklearn import datasets

# load data and plots code is taken from the lecture slides
# Load data : https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/13/4
# Plots: https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/4/2


def respose_cont_predictor_cont(dataset, cont, response):
    fig = px.scatter(dataset, x=cont, y=response, trendline="ols")
    fig.update_layout(
        title=f"<b> Continuous response Continuous Predictor plot for: {response} vs  {cont}</b>",  # From HW_01
        xaxis_title=f"Predictor:{cont}",
        yaxis_title=f"Response:{response}",
    )
    fig.show()
    fig.write_html(
        file=f"{cont} vs {response} plot.html", include_plotlyjs="cdn"
    )  # From HW_01

    return


class Load_Test_Datasets:
    def __init__(self):
        self.sns_data_sets = ["mpg", "tips", "titanic"]
        self.sklearn_data_sets = ["diabetes", "breast_cancer"]
        self.all_data_sets = self.sns_data_sets + self.sklearn_data_sets

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
    ) -> Tuple[pd.DataFrame, List[str], str]:
        if dataset_name is None:
            dataset_name = random.choice(self.all_data_sets)
        else:
            if dataset_name not in self.all_data_sets:
                raise Exception(f"Data set choice not valid: {dataset_name}")

        if dataset_name in self.sns_data_sets:
            if dataset_name == "mpg":
                dataset = sns.load_dataset(name="mpg").dropna().reset_index()
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
                dataset = sns.load_dataset(name="tips").dropna().reset_index()
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
                dataset = sns.load_dataset(name="titanic").dropna()
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
                dataset = pd.DataFrame(data_l.data, columns=data_l.feature_names)
            elif dataset_name == "breast_cancer":
                data_l = datasets.load_breast_cancer()
                dataset = pd.DataFrame(data_l.data, columns=data_l.feature_names)
            dataset["target"] = data_l.target
            P_Predictors = data_l.feature_names
            R_Response = "target"

        # Change category dtype to string
        for predictor in P_Predictors:
            if dataset[predictor].dtype in ["category"]:
                dataset[predictor] = dataset[predictor].astype(str)

        print(f"Data set selected: {dataset_name}")
        dataset.reset_index(drop=True, inplace=True)
        print(dataset.head())
        print(f" Columns in the dataset:{list(dataset.columns)}")
        print(f" Predictors: {P_Predictors}")
        print(f"Response :{R_Response}")
        #
        # print("Unique Values in Responses", dataset[R_Response].unique())

        if int(dataset[R_Response].nunique()) > 2:
            R_Response_type = "cont"
        else:
            R_Response_type = "cat"

        P_Predictors_type = {}

        for i in P_Predictors:
            # print(dataset[i].dtype, i)
            if dataset[i].dtype == "object":
                P_Predictors_type[i] = "cat"
            else:
                dataset[i] = dataset[i].astype(int)
                P_Predictors_type[i] = "cont"

        return dataset, P_Predictors, R_Response, R_Response_type, P_Predictors_type


def main():
    test_datasets = Load_Test_Datasets()
    for test in test_datasets.Fetch_datasets():
        (
            dataset,
            P_Predictors,
            R_Response,
            R_Response_type,
            P_Predictors_type,
        ) = test_datasets.Fetch_sample_datasets(dataset_name=test)

        if R_Response_type == "cont":

            cat = []
            cont = []

            for i in P_Predictors_type:
                if P_Predictors_type[i] == "cont":
                    cont.append(i)
                else:
                    cat.append(i)

            # for i in cat:
            # cont_resp_cat_predictor(dataset, i, R_Response)
            for i in cont:
                respose_cont_predictor_cont(dataset, i, R_Response)
        # #
        # elif R_Response_type == 'cat':
        #     cat = []
        #     cont = []
        #     for i in P_Predictors_type:
        #         if P_Predictors_type[i] == 'cont':
        #             cont.append(i)
        #         else:
        #             cat.append(i)
        # print(cat, cont)
        # for i in cat:
        #
        #     cat_response_cat_predictor(dataset, i, R_Response)

        # for i in cont:
        # print(dataset[i], dataset[R_Response])
        # cat_resp_cont_predictor(dataset, i, R_Response)

        #


if __name__ == "__main__":
    sys.exit(main())
