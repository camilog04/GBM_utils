import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def relativity_data(
    df,
    variable,
    target="pure_premium",
    prediction="predict",
    weight="earned_exposures"
):
    temp = df[
        [variable, target, prediction, weight]
    ].copy()

    temp = temp[temp[weight] > 0]

    # Clean categorical/missing levels
    temp[variable] = temp[variable].astype("object")

    temp[variable] = temp[variable].replace(
        ["", " ", "nan", "None"],
        "<MISSING>"
    )

    temp[variable] = temp[variable].fillna("<MISSING>")

    # Weighted actual / predicted
    temp["_actual"] = temp[target] * temp[weight]
    temp["_predicted"] = temp[prediction] * temp[weight]

    summary = (
        temp
        .groupby(variable, dropna=False)
        .agg(
            exposure=(weight, "sum"),
            actual_amount=("_actual", "sum"),
            predicted_amount=("_predicted", "sum")
        )
        .reset_index()
    )

    summary["actual_pp"] = (
        summary["actual_amount"] / summary["exposure"]
    )

    summary["predicted_pp"] = (
        summary["predicted_amount"] / summary["exposure"]
    )

    overall_actual = (
        temp["_actual"].sum() / temp[weight].sum()
    )

    overall_predicted = (
        temp["_predicted"].sum() / temp[weight].sum()
    )

    summary["actual_relativity"] = (
        summary["actual_pp"] / overall_actual
    )

    summary["predicted_relativity"] = (
        summary["predicted_pp"] / overall_predicted
    )

    summary["exposure_pct"] = (
        summary["exposure"] / summary["exposure"].sum()
    )

    return summary



def add_model_relativity(
    rel_df,
    variable,
    glm_model
):
    df = rel_df.copy()

    coef = glm_model.coef()

    prefix = f"{variable}."

    categorical_coefs = {
        key[len(prefix):]: value
        for key, value in coef.items()
        if key.startswith(prefix)
    }

    if categorical_coefs:
        df["_beta"] = (
            df[variable]
            .astype(str)
            .map(categorical_coefs)
            .fillna(0.0)
        )

        df["_raw_model_relativity"] = np.exp(df["_beta"])

    else:
        beta = coef.get(variable, 0.0)

        numeric_x = pd.to_numeric(
            df[variable],
            errors="coerce"
        )

        df["_raw_model_relativity"] = np.exp(
            beta * numeric_x
        )

    valid = (
        df["_raw_model_relativity"].notna()
        & df["exposure"].notna()
        & (df["exposure"] > 0)
    )

    if valid.sum() == 0 or df.loc[valid, "exposure"].sum() == 0:
        df["model_relativity"] = np.nan
    else:
        model_average = np.average(
            df.loc[valid, "_raw_model_relativity"],
            weights=df.loc[valid, "exposure"]
        )

        df["model_relativity"] = (
            df["_raw_model_relativity"] / model_average
        )

    return df.drop(
        columns=["_beta", "_raw_model_relativity"],
        errors="ignore"
    )


def plot_relativity(df, variable):
    plot_df = df.copy()

    # Preserve one common x position for lines and bars
    x = np.arange(len(plot_df))

    labels = (
        plot_df[variable]
        .astype(str)
        .replace("", "<MISSING>")
    )

    fig, (ax_rel, ax_exp) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1],
            "hspace": 0.05
        }
    )

    # -----------------------
    # Relativity plot
    # -----------------------

    ax_rel.plot(
        x,
        plot_df["actual_relativity"],
        marker="o",
        label="Actual"
    )

    ax_rel.plot(
        x,
        plot_df["predicted_relativity"],
        marker="o",
        label="Predicted"
    )

    ax_rel.plot(
        x,
        plot_df["model_relativity"],
        marker="o",
        color="red",
        linewidth=2,
        label="Model Relativity"
    )

    ax_rel.axhline(
        1,
        linestyle="--",
        linewidth=1
    )

    ax_rel.set_ylabel("Relativity")

    ax_rel.set_title(
        f"{variable} - Actual vs Predicted vs Model Relativity"
    )

    ax_rel.legend()

    # -----------------------
    # Exposure bars
    # -----------------------

    ax_exp.bar(
        x,
        plot_df["exposure"]
    )

    ax_exp.set_ylabel("Exposure")
    ax_exp.set_xlabel(variable)

    ax_exp.set_xticks(x)
    ax_exp.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd

def weighted_gini(
    df,
    actual="pure_premium",
    predicted="predict",
    weight="earned_exposures"
):
    temp = df[[actual, predicted, weight]].dropna().copy()
    temp = temp[temp[weight] > 0]

    # Convert pure premium back to loss dollars
    temp["actual_loss"] = temp[actual] * temp[weight]

    # Sort from lowest to highest predicted risk
    temp = temp.sort_values(predicted)

    total_exposure = temp[weight].sum()
    total_loss = temp["actual_loss"].sum()

    temp["cum_exposure"] = (
        temp[weight].cumsum() / total_exposure
    )

    temp["cum_loss"] = (
        temp["actual_loss"].cumsum() / total_loss
    )

    # Add origin
    x = np.concatenate([[0], temp["cum_exposure"].to_numpy()])
    y = np.concatenate([[0], temp["cum_loss"].to_numpy()])

    # Area between Lorenz curve and diagonal
    lorenz_area = np.trapezoid(y, x)

    return 1 - 2 * lorenz_area



def lift_data(
    df,
    actual="pure_premium",
    predicted="predict",
    weight="earned_exposures",
    n_bins=10
):
    temp = df[
        [actual, predicted, weight]
    ].dropna().copy()

    temp = temp[temp[weight] > 0]

    # Sort low -> high predicted risk
    temp = temp.sort_values(predicted).reset_index(drop=True)

    # Cumulative exposure
    total_exposure = temp[weight].sum()

    temp["_cum_exposure"] = (
        temp[weight].cumsum() / total_exposure
    )

    # Exposure-weighted deciles
    temp["decile"] = np.ceil(
        temp["_cum_exposure"] * n_bins
    ).clip(1, n_bins).astype(int)

    # Convert PP to loss dollars
    temp["_actual_loss"] = (
        temp[actual] * temp[weight]
    )

    temp["_predicted_loss"] = (
        temp[predicted] * temp[weight]
    )

    lift = (
        temp
        .groupby("decile")
        .agg(
            exposure=(weight, "sum"),
            actual_loss=("_actual_loss", "sum"),
            predicted_loss=("_predicted_loss", "sum"),
            policies=(weight, "size")
        )
        .reset_index()
    )

    lift["actual_pp"] = (
        lift["actual_loss"] / lift["exposure"]
    )

    lift["predicted_pp"] = (
        lift["predicted_loss"] / lift["exposure"]
    )

    lift["exposure_pct"] = (
        lift["exposure"] / lift["exposure"].sum()
    )

    return lift


def plot_lift(lift_df, title):

    fig, ax = plt.subplots(figsize=(12, 6))

    x = lift_df["decile"]

    ax.plot(
        x,
        lift_df["actual_pp"],
        marker="o",
        linewidth=2,
        label="Actual Pure Premium"
    )

    ax.plot(
        x,
        lift_df["predicted_pp"],
        marker="o",
        linewidth=2,
        label="Predicted Pure Premium"
    )

    ax.set_xlabel("Predicted Risk Decile")
    ax.set_ylabel("Pure Premium")
    ax.set_title(title)

    ax.set_xticks(range(1, 11))
    ax.legend()

    plt.tight_layout()
    plt.show()


def weighted_ks(
    df,
    actual="pure_premium",
    predicted="predict",
    weight="earned_exposures"
):
    temp = df[[actual, predicted, weight]].dropna().copy()
    temp = temp[temp[weight] > 0]

    # Actual loss dollars
    temp["_loss"] = temp[actual] * temp[weight]

    # Sort from lowest to highest predicted risk
    temp = temp.sort_values(predicted).reset_index(drop=True)

    # Cumulative exposure and loss distributions
    temp["cum_exposure"] = (
        temp[weight].cumsum() /
        temp[weight].sum()
    )

    temp["cum_loss"] = (
        temp["_loss"].cumsum() /
        temp["_loss"].sum()
    )

    # KS distance
    temp["ks_distance"] = (
        temp["cum_loss"] - temp["cum_exposure"]
    ).abs()

    ks = temp["ks_distance"].max()

    return ks