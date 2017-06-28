#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: plumber_plots.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: Plumber plots and variants
"""

import os
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def hls_palette(n_models):
    return sns.color_palette("hls", n_models + 1)


def group_metrics(metric):
    """Return the group of a metric"""
    if metric in ['nme', 'mbe', 'sd_diff', 'corr']:
        return "common"
    elif metric in ['extreme_5', 'extreme_95']:
        return 'extremes'
    elif metric in ['skewness', 'kurtosis', 'overlap']:
        return 'distribution'
    else:
        return None


def p_facetted_plumber_plot(df, x='variable', y='rank', xfacet=None, yfacet=None, **kwargs):
    """Plots a rank df in a facetted grid """

    n_models = len(df.name.unique())

    groups = ['name', x]

    if yfacet is not None:
        groups.append(yfacet)
        yfacets = list(df[yfacet].unique())
        n_yfacet = len(yfacets)
    else:
        yfacets = [None]
        n_yfacet = 1

    if xfacet is not None:
        groups.append(xfacet)
        xfacets = list(df[xfacet].unique())
        n_xfacet = len(xfacets)
    else:
        xfacets = [None]
        n_xfacet = 1

    # create subplt grid according to facets
    fig, axes = plt.subplots(nrows=n_yfacet, ncols=n_xfacet, sharex=True, sharey='row', squeeze=False)

    for i, yfacet_val in enumerate(yfacets):
        for j, xfacet_val in enumerate(xfacets):
            ax = axes[i, j]

            # subset DF according to facets
            if yfacet is not None and xfacet is not None:
                subset_df = df[(df[yfacet] == yfacet_val) & (df[xfacet] == xfacet_val)]
            elif yfacet is not None:
                subset_df = df[df[yfacet] == yfacet_val]
            elif xfacet is not None:
                subset_df = df[df[xfacet] == xfacet_val]

            p_plumber_plot(subset_df, xfacet_val, '', group=x, ax=ax, legend=False, **kwargs)

            ax.legend().set_visible(False)
            ax.xaxis.label.set_visible(False)

            if j == 0 and yfacet is not None:
                ax.set_ylabel(yfacet_val)

    fig.legend(loc='lower center', *ax.get_legend_handles_labels(), ncol=n_models, fontsize='small')

    return fig, axes


def p_plumber_plot(df, model_set, metric_set, palette=None, ax=None,
                   group='variable', **kwargs):
    n_sites = len(df.site.unique())
    n_models = len(df.name.unique())

    df = df.copy()
    df['name'] = df['name'].astype('category')
    mean_df = df.groupby([group, 'name'], sort=False)['rank'].mean().reset_index()

    mean_df_wide = (mean_df.pivot(index=group, columns='name', values='rank')
                           .ix[mean_df[group].unique(), ])

    if palette is None:
        palette = hls_palette(n_models)

    n_group = len(df[group].unique())

    axes = mean_df_wide.plot(marker='.', markersize=10, color=palette,
                             xlim=(-0.2, n_group - 0.8), figsize=(8, 8), ax=ax,
                             **kwargs)

    if "legend" in kwargs and kwargs["legend"]:
        axes.legend(ncol=3)

    axes.set_xticks(range(n_group))
    axes.set_xticklabels(mean_df_wide.index)

    if ax is not None:
        ax.set_title(model_set)
    else:
        plt.title('PLUMBER plot: {ms}, {m} metrics at {s} sites'.format(ms=model_set, m=metric_set, s=n_sites))

    return axes.figure, axes


def p_plumber_plot_horizontal(df, model_set, metric_set, palette=None, ax=None):
    n_sites = len(df.site.unique())
    names = df.name.unique()
    n_models = len(names)

    df = df.copy()
    df['name'] = df['name'].astype('category')
    mean_df = df.groupby(['variable', 'name'], sort=False)['rank'].mean().reset_index()

    mean_df_wide = mean_df.pivot(index='name', columns='variable', values='rank')

    if palette is None:
        palette = hls_palette(n_models)

    axes = mean_df_wide.plot(marker='.', markersize=10, color=palette,
                             xlim=(-0.2, n_models - 0.8), figsize=(12, 5), ax=ax)
    axes.set_xticks(range(n_models))
    axes.set_xticklabels(labels=names, rotation=-90)
    axes.legend(ncol=3)

    if ax is not None:
        ax.set_title(model_set)
    else:
        plt.title('PLUMBER plot: {ms}, {m} metrics at {s} sites'.format(ms=model_set, m=metric_set, s=n_sites))

    return axes.figure, axes


def plot_plumber_plot(df, model_set, metric_set, palette=None, outdir='plots/PLUMBER_plots'):
    p_plumber_plot(df, model_set, metric_set, palette)

    outdir = '{d}/{ms}'.format(d=outdir, ms=model_set)
    os.makedirs(outdir, exist_ok=True)
    path = '{d}/PLUMBER_plot_{mset}_{ms}.png'.format(d=outdir, mset=model_set, ms=metric_set)
    plt.savefig(path)


def plot_rank_histograms(df, model_set, metric_set, palette=None, outdir='plots/PLUMBER_plots'):

    n_sites = len(df.site.unique())
    n_models = len(df.name.unique())

    df = df.copy()
    df['name'] = df['name'].astype('category')

    count_df = (df[['rank', 'name', 'variable', 'value']]
                .groupby(['rank', 'variable', 'name'], sort=False)
                .agg('count')
                .fillna(0)
                .reset_index()
                .rename(columns={'value': 'count'}))

    if palette is None:
        palette = hls_palette(n_models)

    g = sns.factorplot(y="rank", x="count", col="variable", hue="name", data=count_df,
                       palette=palette, orient='h', legend=False)
    g.axes[0, 0].invert_yaxis()
    g.fig.legend(loc='lower center', *g.axes[0, 0].get_legend_handles_labels(), ncol=10)

    plt.suptitle('Rank counts: {ms}, {m} metrics at {s} sites'.format(ms=model_set, m=metric_set, s=n_sites))
    g.fig.set_figheight(g.fig.get_figheight() + 2)

    outdir = '{d}/{ms}'.format(d=outdir, ms=model_set)
    os.makedirs(outdir, exist_ok=True)
    path = '{d}/Rank_histograms_{mset}_{ms}.png'.format(d=outdir, mset=model_set, ms=metric_set)
    plt.savefig(path)

    return g.fig, g.fig.axes


def plot_lipson_plot(df, model_set, metric_set, key_models=None,
                     outdir='plots/Multimodel_ranks_bars'):
    """TODO: Docstring for function.
    """
    n_sites = len(df.site.unique())
    # n_models = len(df.name.unique())

    df = df.copy()
    # df['name'] = df['name'].astype('category')

    mean_df = (df.set_index(['name', 'site', 'metric', 'variable'])['rank']  # .unstack(['site'])
                 .groupby(level=['variable', 'name'], sort=False)
                 .mean()
                 .reset_index())

    outdir = '{d}/{ms}'.format(d=outdir, ms=model_set)

    for i, v in enumerate(['NEE', 'Qle', 'Qh']):
        plot_df = mean_df.copy()
        # ranks the ranked data so that the plots can be ordered correctly
        plot_df['rank_order'] = np.repeat(
            ss.rankdata(plot_df.ix[plot_df['variable'] == v, 'rank']),
            3, axis=-1)
        plot_df.sort_values(['rank_order', 'variable'], inplace=True)
        plot_df.reset_index(drop=True, inplace=True)
        # plot_df['name'] = plot_df['name'].astype('category')

        if key_models is not None:
            colours = ['red' if m in key_models else 'grey' for m in plot_df['name'].unique()]
            g = sns.factorplot(data=plot_df, x='name', y='rank', row='variable', kind='bar',
                               order=plot_df['name'].unique(), margin_titles=True,
                               size=5, aspect=4, palette=sns.color_palette(colours))
        else:
            g = sns.factorplot(data=plot_df, x='name', y='rank', row='variable', kind='bar',
                               order=plot_df['name'].unique(), margin_titles=True,
                               size=5, aspect=4)
        g.set_xticklabels(rotation=90)
        g.fig.suptitle("Rank averages for {ms}, over sites and metrics, sorted by {v} performance, at {s} sites".format(ms=model_set, v=v, s=n_sites))
        g.fig.tight_layout(rect=(0.01, 0.05, 0.97, 0.97))

        file_path = 'Rank_bars_{mset}_{ms}_{v}_sort.pdf'
        file_path = file_path.format(d=outdir, mset=model_set, ms=metric_set, v=v)

        os.makedirs(outdir, exist_ok=True)
        path = '{d}/{f}'.format(d=outdir, f=file_path)
        plt.savefig(path)

    return g.fig, g.fig.axes


def plot_pca_plot(df, model_set, metric_set, palette=None, outdir='plots/PCA_plots'):
    n_sites = len(df.site.unique())
    n_models = len(df.name.unique())

    df = df.copy()
    df['name'] = df['name'].astype('category')

    wide_df = pd.pivot_table(df, index=['name', 'site'],
                             columns=['variable', 'metric'], values='value')

    if palette is None:
        colours = sns.color_palette("Spectral", n_models)
    else:
        colours = palette
    c_dict = dict(zip(df.name.unique(), colours))

    pipe = make_pipeline(StandardScaler(), PCA())

    y_pca = pipe.fit_transform(wide_df)

    fig, ax = plt.subplots()

    # plot_df = pd.DataFrame(data=y_pca[:, 0:2], columns=['PC1', 'PC2'],
    #                        index=wide_df.index.get_level_values(0))

    ax.scatter(y_pca[:, 0], y_pca[:, 1],
               c=[c_dict[m] for m in wide_df.index.get_level_values(0)])

    for variable in df.variable.unique():
        m_vector = np.float64((wide_df.columns.get_level_values(0) == variable).reshape(1, -1))
        m_pca = pipe.transform(m_vector)[0, 0:2]
        plt.plot([0, m_pca[0]], [0, m_pca[1]], 'g-', lw=1)
        ax.annotate(variable, xy=m_pca + np.array([0.1, 0.1]), color='green')

    for metric in df.metric.unique():
        m_vector = np.float64((wide_df.columns.get_level_values(1) == metric).reshape(1, -1))
        m_pca = pipe.transform(m_vector)[0, 0:2]
        plt.plot([0, m_pca[0]], [0, m_pca[1]], 'r-', lw=1)
        ax.annotate(metric, xy=m_pca + np.array([-1, 0.1]), color='red')

    # TODO: Fix to use colours/model names
    fig.legend(loc='lower center', *ax.get_legend_handles_labels(), ncol=10)

    plt.suptitle('PCA plot: {ms}, {m} metrics at {s} sites'.format(ms=model_set, m=metric_set, s=n_sites))
    fig.set_figheight(fig.get_figheight() + 2)

    outdir = '{d}/{ms}'.format(d=outdir, ms=model_set)
    os.makedirs(outdir, exist_ok=True)
    path = '{d}/PCA_plot_{mset}_{ms}.png'.format(d=outdir, mset=model_set, ms=metric_set)
    plt.savefig(path)

    return fig, fig.axes
