import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pathlib
    import polars as pl
    from scipy.stats import gaussian_kde
    return gaussian_kde, np, pathlib, pl, plt


@app.cell
def _(pathlib, pl):
    ann_path = pathlib.Path("/fs/scratch/PAS2136/cain429/beetlepalooza/annotations.json")

    annotations_df = pl.read_json(ann_path)
    annotations_df_ex = annotations_df.explode("measurements").with_columns(
        pl.col("measurements").struct.unnest())

    measurements_per_annoator = annotations_df_ex.pivot(
        values="dist_cm", 
        index="individual_id",    
        on="annotator",
        aggregate_function="first"
    )
    measurements_per_annoator = measurements_per_annoator.filter(pl.col("IsaFluck").is_not_null() & pl.col("rileywolcheski").is_not_null() & pl.col("ishachinniah").is_not_null())
    measurements_per_annoator
    return (measurements_per_annoator,)


@app.cell
def _(gaussian_kde, measurements_per_annoator, np, plt):
    fluck_data = measurements_per_annoator["IsaFluck"].to_numpy()
    isha_data = measurements_per_annoator["ishachinniah"].to_numpy()
    riley_data = measurements_per_annoator["rileywolcheski"].to_numpy()

    label_map = {"IsaFluck": fluck_data, "ishachinniah": isha_data, "rileywolcheski": riley_data}
    comparison = [["IsaFluck", "ishachinniah"], ["IsaFluck", "rileywolcheski"], ["ishachinniah", "rileywolcheski"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    custom_ticks = np.arange(0.25, 2.0, 0.25)

    for ax, comparison in zip(axes, comparison):
        x = label_map[comparison[0]]
        y = label_map[comparison[1]]
        # 1. Calculate Density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy) # Calculate density for each point
    
        # 2. Sort points so dense ones are plotted on top
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
        scatter = ax.scatter(x, y, c=z, s=15, cmap='viridis', alpha=0.8, edgecolor='none') # 'c=z' maps density to color
        ax.set_xlabel(comparison[0])
        ax.set_ylabel(comparison[1])
        ax.set_xticks(custom_ticks)
        ax.set_yticks(custom_ticks)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.plot([0.15, 1.85], [0.15, 1.85], color='orange', linestyle='--', linewidth=2, label='Perfect Agreement')

    plt.colorbar(scatter, label='Point Density')
    plt.tight_layout()
    plt.show()
    return fluck_data, isha_data, riley_data


@app.cell
def _(fluck_data, isha_data, np, pl, riley_data):
    def get_metrics(y_true, y_pred, annotator_pair):
        rmse = np.sqrt(np.mean(y_true - y_pred) ** 2)

        res = np.sum((y_true - y_pred) ** 2)
        tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (res / tot)
    
        bias = np.mean(y_pred - y_true)

        return {"annoator pair": annotator_pair, "rmse": rmse, "r^2": r2, "bias": bias}

    data = [get_metrics(fluck_data, isha_data, "IshaFluck vs ishachinniah"), get_metrics(fluck_data, riley_data, "IshaFluck vs rileywolcheski"), get_metrics(isha_data, riley_data, "ishachinniah vs rileywolcheski")]


    metrics_df = pl.DataFrame(data)
    metrics_df
    return


if __name__ == "__main__":
    app.run()
