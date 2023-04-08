import os

import numpy as np
import plotly.graph_objs as go

urls_1 = []
urls_2 = []
urls_3 = []
urls_4 = []


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
    # Reference : https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
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

    # figure.show()

    if not os.path.isdir("Cont-Cat-plots"):
        os.mkdir("Cont-Cat-plots")
    file_path = f"Cont-Cat-plots/{response}-{predictor}-plot.html"
    urls_1.append(file_path)
    figure.write_html(file=file_path, include_plotlyjs="cdn")

    return


def plot_categorical_predictor_and_continuous_response(df, predictors, response):
    """
    .   Cat Pred and Cont Response
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
        # figure.show()

        if not os.path.isdir("Cat-Cont-plots"):
            os.mkdir("Cat-Cont-plots")
        file_path = f"Cat-Cont-plots/{response}-{predictor}-plot.html"
        urls_2.append(file_path)
        figure.write_html(file=file_path, include_plotlyjs="cdn")

    return


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

    # figure.show()

    if not os.path.isdir("Cont-Cont-plots"):
        os.mkdir("Cont-Cont-plots")
    file_path = f"Cont-Cont-plots/{response}-{predictor}-plot.html"
    urls_3.append(file_path)
    figure.write_html(file=file_path, include_plotlyjs="cdn")
    return


def plot_categorical_predictor_and_categorical_response(df, predictors, response):
    """
    Cat Pred and Cat Response
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

    # figure.show()

    if not os.path.isdir("Cat-Cat-plots"):
        os.mkdir("Cat-Cat-plots")
    file_path = f"Cat-Cat-plots/{response}-{predictor}-plot.html"
    urls_4.append(file_path)
    figure.write_html(file=file_path, include_plotlyjs="cdn")

    return
