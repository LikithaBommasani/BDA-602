import random
from typing import List, Tuple

import pandas
import seaborn
from sklearn import datasets


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


if __name__ == "__main__":
    test_datasets = Load_Test_Datasets()
    for test in test_datasets.Fetch_datasets():
        df, P_Predictors, R_Response = test_datasets.Fetch_sample_datasets(
            dataset_name="test"
        )
