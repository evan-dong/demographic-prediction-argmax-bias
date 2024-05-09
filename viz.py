import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import mpltern
from brokenaxes import brokenaxes
import geopandas
from sklearn.calibration import CalibrationDisplay


## CALIBRATION
def plot_calibration(
    predictions, probs, class_labels, palette=None, title=None, filepath=None
):
    _, ax = plt.subplots()
    for race, colname in enumerate(probs.columns):
        one_rest = (predictions == race) * 1
        disp = CalibrationDisplay.from_predictions(
            one_rest,
            probs[colname],
            n_bins=10,
            ax=ax,
            c=None if palette is None else palette[class_labels[race]],
            label=class_labels[race],
        )
    plt.xlabel("Mean Predicted Probability", fontsize=14)
    plt.ylabel("Actual Fraction of Population", fontsize=14)
    plt.legend(frameon=False, fontsize=12, title_fontsize=14)
    plt.tick_params(axis="both", which="major", rotation=0, labelsize=10)

    sns.despine()

    if title is not None:
        plt.title(title)

    if filepath is not None:
        plt.savefig(filepath, bbox_inches="tight", dpi=500)

    plt.show()


## PARETO
def plot_pareto(
    means,
    points=None,
    curves=None,
    fid_col="Aggregate Posterior Fidelity",
    acc_col="Accuracy",
    palette=None,
    filename=None,
):
    points = [] if points is None else points
    curves = [] if curves is None else curves

    for point in points:
        values = means.loc[point]
        plt.plot(
            values[fid_col],
            values[acc_col],
            "o",
            label=point,
            c=None if palette is None else palette[point],
        )

    for curve in curves:
        curve_samples = means[means.index.str.contains(curve, regex=False)]
        curve_samples = curve_samples.sort_index()

        plt.plot(
            curve_samples[fid_col],
            curve_samples[acc_col],
            "-o",
            label=curve,
            c=None if palette is None else palette[curve],
        )

    axlabel_size = 14
    tick_size = 10
    legend_size = 12

    plt.xlabel(fid_col, fontsize=axlabel_size)
    plt.ylabel("Accuracy", fontsize=axlabel_size)
    plt.legend(frameon=False, fontsize=legend_size)
    sns.despine()
    plt.tick_params(
        axis="both", which="major", rotation=0, labelsize=tick_size
    )  # labelsize=6,

    if filename is not None:
        plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.show()


def plot_broken_pareto(
    means,
    points: list[list[str]],
    curves: list[list[str]],
    fid_col="Ground Truth Fidelity",
    acc_col="Accuracy",
    breakwidth=0.2,
    palette=None,
    legend="best",
    filename=None,
):
    if len(points) != len(curves):
        raise ValueError("points and curves must be the same length")

    padding = 0.05

    diffs = []
    for point_list, curve_list in zip(points, curves):
        columns = point_list + [
            curve_value
            for curve in curve_list
            for curve_value in means.index[means.index.str.contains(curve, regex=False)]
        ]
        values = means.loc[columns]
        maximum = values[acc_col].max(axis=None)
        minimum = values[acc_col].min(axis=None)
        diff = maximum - minimum
        assert diff >= 0
        diffs.append((minimum - (diff * padding), maximum + (diff * padding)))

    diffs.sort(key=lambda x: x[0])

    all_curves = [curve_value for curve in curves for curve_value in curve]
    all_points = [point for point_group in points for point in point_group]

    bax = brokenaxes(ylims=diffs, hspace=breakwidth)

    axlabel_size = 14
    tick_size = 10
    legend_size = 12

    for point in all_points:
        values = means.loc[point]
        bax.plot(
            values[fid_col],
            values[acc_col],
            "o",
            label=point,
            c=None if palette is None else palette[point],
        )

    for curve in all_curves:
        curve_samples = means[means.index.str.contains(curve, regex=False)]
        curve_samples = curve_samples.sort_index()

        bax.plot(
            curve_samples[fid_col],
            curve_samples[acc_col],
            "-o",
            label=curve,
            c=None if palette is None else palette[curve],
        )
    if legend is not None:
        bax.legend(frameon=False, fontsize=legend_size, loc=legend)

    bax.set_xlabel(fid_col, fontsize=axlabel_size, labelpad=20)
    bax.set_ylabel(acc_col, fontsize=axlabel_size, labelpad=40)

    if filename is not None:
        plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.show()


## BARPLOTS
def basic_biasplot(df, filename=None, bar_order=None, palette=None):
    plt.figure(figsize=(10, 16))

    g = sns.catplot(
        data=df.reset_index().melt(id_vars="Race"),
        kind="bar",
        x="Race",
        y="value",
        hue="Method",
        legend=True,
        order=bar_order,
        palette=palette,
    )

    axlabel_size = 14
    tick_size = 10
    legend_size = 12

    g.despine()
    g.set_axis_labels(
        "", "Bias (Difference from Voter Population)", fontsize=axlabel_size
    )
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, 1.3),
        ncol=2,
        fontsize=legend_size,
        frameon=False,
    )

    plt.tick_params(axis="x", which="major", rotation=20, labelsize=tick_size)

    if filename is not None:
        plt.savefig(filename, dpi=500, bbox_inches="tight")


def broken_axis_biasplot(
    df,
    bottom_break=None,
    top_break=None,
    breakwidth=0.1,
    palette=None,
    bar_order=None,
    annotations=False,
    filename=None,
):
    break_fraction = 1 / 20

    ## keep only the used rows
    df = df.reindex(bar_order)

    sublplot_size_ratio = 0.1

    ## dynamic break calculation, assuming at least 4 values
    _, second_least = np.sort(np.array(df), axis=None)[:2]
    second_most, _ = np.sort(np.array(df), axis=None)[-2:]

    center_scale = second_most - second_least

    bottom_break = (
        second_least - center_scale * break_fraction
        if bottom_break is None
        else bottom_break
    )
    top_break = (
        second_most + center_scale * break_fraction if top_break is None else top_break
    )

    assert bottom_break < top_break  ## consistency check

    fig, axes = plt.subplots(
        ncols=1,
        nrows=3,
        sharex=False,
        height_ratios=(sublplot_size_ratio, 1, sublplot_size_ratio),
        gridspec_kw={"hspace": breakwidth},
    )
    (ax_top, ax_center, ax_bottom) = axes
    for ax in axes:
        sns.barplot(
            data=df.reset_index().melt(id_vars="Race"),
            x="Race",
            y="value",
            hue="Method",
            order=bar_order,
            palette=palette,
            ax=ax,
        )

    ax_top.set_ylim(bottom=top_break)
    ax_center.set_ylim(bottom_break, top_break)
    ax_bottom.set_ylim(top=bottom_break)

    sns.despine(ax=ax_bottom)
    sns.despine(ax=ax_center, bottom=True)
    sns.despine(ax=ax_top, bottom=True)

    d = 0.01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-2 * d, +2 * d), **kwargs)  # top-left diagonal

    kwargs.update(transform=ax_center.transAxes)  # switch to the center axes
    ax_center.plot(
        (-d, +d),
        (1 - 2 * d * sublplot_size_ratio, 1 + 2 * d * sublplot_size_ratio),
        **kwargs,
    )
    ax_center.plot(
        (-d, +d), (-2 * d * sublplot_size_ratio, +2 * d * sublplot_size_ratio), **kwargs
    )

    kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
    ax_bottom.plot((-d, +d), (1 - 2 * d, 1 + 2 * d), **kwargs)  # bottom-left diagonal

    axlabel_size = 14
    tick_size = 10
    legend_size = 12
    annotation_size = 12

    legend_ncols = 2
    ax_top.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.4,
            (0.55 + 0.27 * np.ceil(df.shape[1] / legend_ncols))
            / sublplot_size_ratio
            / 3,
        ),
        ncol=legend_ncols,
        fontsize=legend_size,
        frameon=False,
    )

    plt.tick_params(axis="x", which="major", rotation=20, labelsize=tick_size)

    # remove other legends
    ax_bottom.legend_.remove()
    ax_center.legend_.remove()
    ax_center.set_xticks([])
    ax_top.set(ylabel=None, xlabel="", xticks=[])
    ax_bottom.set(ylabel=None, xlabel="")
    ax_center.set(ylabel=None, xlabel="", xticks=[])

    ax_center.set_ylabel(
        "Bias (Difference from Voter Population)", fontsize=axlabel_size
    )

    ## specifically,
    if annotations:
        disc_diff_x = 0.15
        disc_diff_y = 0.13
        disc_width = 1.45
        ax_center.annotate(
            "Discretization Bias",
            xy=(disc_diff_x, disc_diff_y),
            xytext=(disc_diff_x + 0.3, disc_diff_y - 0.032),
            xycoords="axes fraction",
            fontsize=annotation_size,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square", fc="white", color="k"),
            arrowprops=dict(
                arrowstyle=f"-[, widthB={disc_width}, lengthB=0.5", lw=1.0, color="k"
            ),
        )

        cal_diff_x = 0.22
        cal_diff_y = 0.34
        cal_width = 2.1

        ax_center.annotate(
            "Miscalibration Error",
            xy=(cal_diff_x, cal_diff_y),
            xytext=(cal_diff_x + 0.3, cal_diff_y - 0.03),
            xycoords="axes fraction",
            fontsize=annotation_size,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square", fc="white", color="k"),
            arrowprops=dict(
                arrowstyle=f"-[, widthB={cal_width}, lengthB=0.5", lw=1.0, color="k"
            ),
        )

    if filename is not None:
        plt.savefig(filename, dpi=500, bbox_inches="tight")

    plt.show()


## SIMPLEX
def plot_simplex(
    probs,
    preds,
    text_labels,
    legend_labels,
    legend: bool = True,
    palette=None,
    filename: str | None = None,
):
    text_size = 16
    tick_size = 12

    ax = plt.subplot(projection="ternary", ternary_sum=1.0)

    plt.grid()
    t = text_labels[0]
    l = text_labels[1]
    r = text_labels[2]
    ax.set_tlabel(text_labels[0], fontsize=text_size)
    ax.set_llabel(text_labels[1], fontsize=text_size)
    ax.set_rlabel(text_labels[2], fontsize=text_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    dotsize = 3

    colormap = {}
    ## Generate the labels (this genuinely seems the easiest way to do this)
    for label in legend_labels:
        points = ax.scatter(
            [],
            [],
            [],
            label=label,
            s=dotsize,
            c=None if palette is None else palette[label],
        )
        colormap[label] = (
            points.get_facecolors().ravel() if palette is None else palette[label]
        )

    if legend:
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(1.1, 1.1),
            markerscale=5,
            fontsize=text_size,
            frameon=False,
        )

    point_colors = np.array([colormap[val] for val in preds])
    probs = np.array(probs)
    points = ax.scatter(
        probs[:, 0], probs[:, 1], probs[:, 2], color=point_colors, s=dotsize
    )

    if filename is not None:
        plt.savefig(filename, dpi=500, bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


## MAPS
def draw_bias_map(df: geopandas.GeoDataFrame, columns: list[str], filename=None):
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(12, 12))
    fig.tight_layout()

    abs_max = df[columns].abs().max(axis=None)
    limit_value = abs_max
    norm = matplotlib.colors.Normalize(vmin=-limit_value, vmax=limit_value, clip=True)
    for ax, colname in zip(axes, columns):
        df.plot(
            column=colname,
            cmap="bwr",
            norm=norm,
            ax=ax,
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.set_title(colname, fontsize=16)
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    sns.despine(left=True, bottom=True)

    ## colorbar
    clb = plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="bwr"), ax=axes, aspect=40
    )
    clb.set_label(label="Caucasian Population Bias", size=20)
    clb.ax.tick_params(labelsize=14)

    if filename is not None:
        plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.show()


def draw_pop_map(df, colname: str, filename=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    fig.tight_layout()
    abs_max = df[colname].abs().max(axis=None)
    limit_value = abs_max

    df.plot(column=colname, cmap="Reds", ax=ax, edgecolor="gray", linewidth=0.5)
    ax.set_title(colname, fontsize=16)
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    sns.despine(left=True, bottom=True)

    ## colorbar
    clb = plt.colorbar(
        matplotlib.cm.ScalarMappable(cmap="Reds"), ax=ax, aspect=40, location="left"
    )
    clb.set_label(label="Caucasian Population", size=20)
    clb.ax.tick_params(labelsize=14)

    if filename is not None:
        plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.show()
