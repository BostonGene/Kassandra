import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error


cells_p = {'B_cells': '#558ce0',
 'CD4_T_cells': '#28a35c',
 'CD8_T_cells': '#58d3bb',
 'Dendritic_cells': '#eaabcc',
 'Endothelium': '#F6783E',
 'Fibroblasts': '#a3451a',
 'Macrophages': '#d689b1',
 'Monocytes': '#ad4f80',
 'NK_cells': '#61cc5b',
 'Neutrophils': '#FCB586',
 'T_cells': '#808000',
 'Other': '#999999',
 'Immune_general': '#4f80ad',
 'Monocytic_cells': '#61babf',
 'Lymphocytes': '#5e9e34',
 'Plasma_B_cells': '#29589e',
 'Non_plasma_B_cells': '#248ce0',
 'Granulocytes': '#FCB511',
 'Basophils': '#fa7005',
 'Eosinophils': '#aaB586',
 'Naive_CD4_T_cells': '#bf8f0f',
 'Memory_CD4_T_cells': '#0f8f0f',
 'Memory_B_cells': '#55afe0',
 'Naive_CD8_T_cells': '#8FBC8F',
 'Memory_CD8_T_cells': '#8FBCFF',
 'Naive_B_cells': '#558cff'}

def ccc_func(x,y):
    """ Concordance Correlation Coefficient """
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc  


def print_fitted_line(x_values, y_values, ax=None, linewidth=1, line_color='black', figsize=(6, 6)):
    """
    The function draws a straight line based on linear regression on x_values and y_values.
    :param x_values: pandas Series or numpy array
    :param y_values: pandas Series or numpy array
    :param ax: matplotlib.axes
    :param linewidth: width of the line
    :param line_color: color of the line
    :param figsize: figsize if ax=None
    :returns: ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    fit_coefs = np.polyfit(x_values, y_values, deg=1)
    fit_values = np.sort(x_values)
    ax.plot(fit_values, fit_coefs[0] * fit_values + fit_coefs[1], linewidth=linewidth, color=line_color)

    return ax


def print_cell(predicted_values, true_values, ax=None, pallete=None, single_color='#999999',
               predicted_name='Predicted percentage of cells, %',
               true_name='Real percentage of cells, %', title=None, corr_title=True, corr_rounding=3,
               figsize=(6, 6), s=60, title_font=20, labels_font=18, ticks_size=17, xlim=None, ylim=None,
               corr_line=True, linewidth=1, line_color='black', pad=15, min_xlim=10, min_ylim=10, labelpad=None):
    """
    The function draws a scatterplot with colors from pallete, with correlation in title and
    straight line based on linear regression on x_values and y_values if needed.
    :param predicted_values: pandas Series
    :param true_values: pandas Series
    :param ax: matplotlib.axes
    :param pallete: dict with colors, keys - some or all names from predicted_values and true_values index
    :param single_color: what color to use if there is no palette or some names are missed
    :param predicted_name: xlabel for plot
    :param true_name: ylabel for plot
    :param title: title for plot, will be combined with ', corr = ' if corr_title=True
    :param corr_title: whether to calculate Pearson correlation and print it with title
    :param corr_rounding: precision in decimal digits for correlation if corr_title=True
    :param figsize: figsize if ax=None
    :param s: scalar or array_like, shape (n, ), marker size for scatter
    :param title_font: title size
    :param labels_font: xlabel and ylabel size
    :param ticks_size: tick values size
    :param xlim: x limits, if None xlim will be (0, 1.2 * max(predicted_values))
    :param ylim: y limits, if None ylim will be (0, 1.2 * max(true_values))
    :param corr_line: whether to draw a straight line based on linear regression on x_values and y_values
    :param linewidth: width of the fitted line
    :param line_color: color of the fitted line
    :param pad: distance for titles from picture
    :param min_xlim: minimal range (max picture value) for x
    :param min_ylim: minimal range (max picture value) for y
    :returns: ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ind_points = true_values.dropna().index.intersection(predicted_values.dropna().index)
    ax.grid(b=False)
    predicted_values = predicted_values.loc[ind_points].astype(float)
    true_values = true_values.loc[ind_points].astype(float)
    if corr_title:
        corrcoef, pval = pearsonr(predicted_values, true_values)
        corrcoef = str(round(corrcoef, corr_rounding))
        pval = str(round(pval, 3))
        if title is not None:
            ax.set_title('{title}, corr = {corr}\np = {p}'.format(title=title,
                                                                  corr=corrcoef,
                                                                  p=pval),
                        size=title_font, pad=pad)
        else:
            ax.set_title('Corr = {corr}\np = {p}'.format(corr=corrcoef,
                                                         p=pval),
                        size=title_font, pad=pad)
    elif title is not None:
        ax.set_title(title, size=title_font, pad=pad)
    ax.set_xlabel(predicted_name, size=labels_font, labelpad = labelpad)
    ax.set_ylabel(true_name, size=labels_font, labelpad = labelpad)
    ax.tick_params(labelsize=ticks_size)

    if xlim is None:
        ax.set_xlim(0, max(1.2 * max(predicted_values), min_xlim))
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim(0, max(1.2 * max(true_values), min_ylim))
    else:
        ax.set_ylim(ylim)

    if pallete is not None:
        colors = [pallete[point] if point in pallete else single_color for point in ind_points]
    else:
        colors = single_color

    ax.scatter(predicted_values, true_values, s=s, color=colors)

    if corr_line:
        print_fitted_line(predicted_values, true_values, ax=ax, linewidth=linewidth, line_color=line_color)

    return ax


def print_all_cells_in_one(predicted_values, true_values, ax=None, pallete=None, single_color='#999999',
                           colors_by='index', predicted_name='Predicted percentage of cells, %',
                           true_name='Real percentage of cells, %', 
                           title=None, mae_title=True, corr_title=True, ccc_title=True,
                           corr_rounding=3, figsize=(8, 8), s=50, title_font=20,
                           labels_font=20, ticks_size=17, xlim=None, ylim=None,
                           corr_line=True, linewidth=1, line_color='black', pad=15, min_xlim=10, min_ylim=10):
    """
    The function draws a scatterplot for all cell types with colors from pallete, with correlation in title and
    straight line based on linear regression if needed.
    :param predicted_values: pandas Dataframe
    :param true_values: pandas Dataframe
    :param ax: matplotlib.axes
    :param pallete: dict with colors, keys - some or all names from predicted_values and true_values
    :param single_color: what color to use if there is no palette or some names are missed
    :param colors_by: which names will be used for colors from pallete, 'index' or 'columns'
    :param predicted_name: xlabel for plot
    :param true_name: ylabel for plot
    :param title: title for plot, will be combined with ', corr = ' if corr_title=True
    :param corr_title: whether to calculate Pearson correlation and print it with title
    :param mae_title: whether to calculate MAE and print it with title
    :param ccc_title: whether to calculate Concordance correlation and print it with title
    :param corr_rounding: precision in decimal digits for correlation if corr_title=True
    :param figsize: figsize if ax=None
    :param s: scalar or array_like, shape (n, ), marker size for scatter
    :param title_font: title size
    :param labels_font: xlabel and ylabel size
    :param ticks_size: tick values size
    :param xlim: x limits, if None xlim will be (0, 1.2 * max(predicted_values))
    :param ylim: y limits, if None ylim will be (0, 1.2 * max(true_values))
    :param corr_line: whether to draw a straight line based on linear regression on x_values and y_values
    :param linewidth: width of the fitted line
    :param line_color: color of the fitted line
    :param pad: distance for titles from picture
    :param min_xlim: minimal range (max picture value) for x
    :param min_ylim: minimal range (max picture value) for y
    :returns: ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ind_names = predicted_values.index.intersection(true_values.index)
    col_names = predicted_values.columns.intersection(true_values.columns)
    predicted_values = predicted_values.loc[ind_names, col_names]
    true_values = true_values.loc[ind_names, col_names]
    ravel_predicted = pd.Series(predicted_values.values.ravel()).dropna()
    ravel_true = pd.Series(true_values.values.ravel()).dropna()
    ravel_ind = ravel_predicted.index.intersection(ravel_true.index)
    ravel_predicted = ravel_predicted.loc[ravel_ind].astype(float)
    ravel_true = ravel_true.loc[ravel_ind].astype(float)
    if xlim is None:
        xlim = (0, max(1.2 * max(ravel_predicted), min_xlim))
    if ylim is None:
        ylim = (0, max(1.2 * max(ravel_true), min_ylim))
    
    if not title:
        title = ''
    if corr_title:
        corrcoef, pval = pearsonr(ravel_predicted.loc[ravel_ind], ravel_true.loc[ravel_ind])
        corr=str(round(corrcoef, corr_rounding))
        title += f'\n corr = {corr} p = {pval:.2e}'
    if mae_title:
        mae = round(mean_absolute_error(ravel_predicted.loc[ravel_ind], ravel_true.loc[ravel_ind]), 3)
        title += f' mae={mae}%'
    if ccc_title:
        ccc = ccc_func(ravel_predicted.loc[ravel_ind], ravel_true.loc[ravel_ind])
        ccc = str(round(ccc, corr_rounding))
        title += f'\n CCC = {ccc}'
    
    ax.set_title(title, size=title_font, pad=pad)

    for col in col_names:
        if colors_by == 'index':
            ax_pallete = pallete
            ax_single_color = single_color
        elif col in pallete:
            ax_pallete = None
            ax_single_color = pallete[col]
        else:
            ax_pallete = None
            ax_single_color = single_color

        print_cell(predicted_values[col], true_values[col], ax=ax, pallete=ax_pallete, single_color=ax_single_color,
                   predicted_name=predicted_name, true_name=true_name, corr_title=False, s=s, labels_font=labels_font,
                   ticks_size=ticks_size, xlim=xlim, ylim=ylim, corr_line=False)
    if corr_line:
        print_fitted_line(ravel_predicted, ravel_true, ax=ax, linewidth=linewidth, line_color=line_color)

    return ax

def print_cell_matras(predicted_values, true_values, subplot_ncols=3, order=None, figsize=(16, 16),
                      adjust_figsize=True, pallete=None, single_color='#999999', colors_by='columns',
                      predicted_name='Predicted percentage of cells, %', true_name='Real percentage of cells, %',
                      title='Correlation of predicted and real percentage of cells',
                      top=0.91, wspace=0.4, hspace=0.4,
                      fontsize_title=20, corr_title=True, s=60, sub_title_font=20, labels_font=16, ticks_size=15,
                      xlim=None, ylim=None, corr_line=True, linewidth=1, line_color='black', pad=15,
                      min_xlim=10, min_ylim=10, labelpad=None, show_sub_titles=True):
    """
    The function draws a grid of scatterplots for each cell type with colors from pallete,
    with correlation in titles and straight lines based on linear regression if needed.
    :param predicted_values: pandas Dataframe
    :param true_values: pandas Dataframe
    :param subplot_ncols: number of subplots in width direction, number of subplots in height direction
                          will be calculated based on data
    :param order: left to right top to down subplots order if parameter contains some index names.
                  If some names are not in parameter, they will be added after, sorted alphabetically.
                  If order=None, names will be sorted alphabetically.
    :param figsize: figsize
    :param adjust_figsize: whether to adjust figsize for square subplots
    :param pallete: dict with colors, keys - some or all names from predicted_values and true_values
    :param single_color: what color to use if there is no palette or some names are missed
    :param colors_by: which names will be used for colors from pallete, 'index' or 'columns'
    :param predicted_name: xlabel for subplots
    :param true_name: ylabel for subplots
    :param title: title for figure
    :param top: top for fig.subplots_adjust(), the top of the subplots of the figure
    :param wspace: wspace for fig.subplots_adjust(), the amount of width reserved for space between subplots,
                   expressed as a fraction of the average axis width
    :param hspace: hspace for fig.subplots_adjust(), the amount of height reserved for space between subplots,
                   expressed as a fraction of the average axis height
    :param fontsize_title: fontsize for title for figure
    :param corr_title: whether to calculate Pearson correlation and print it with subplot titles
    :param sub_title_font: fontsize for subplot titles
    :param labels_font: xlabel and ylabel size for subplots
    :param ticks_size: tick values size for subplots
    :param xlim: x limits for each subplot, if None xlim will be (0, 1.2 * max(predicted_values))
    :param ylim: y limits for each subplot, if None ylim will be (0, 1.2 * max(true_values))
    :param corr_line: whether to draw a straight line based on linear regression
                      on x_values and y_values for sublots
    :param linewidth: width of the fitted line for sublots
    :param line_color: color of the fitted line for sublots
    :param pad: distance for titles from picture
    :param min_xlim: minimal range (max picture value) for x
    :param min_ylim: minimal range (max picture value) for y
    :param labelpad: distance for axes labels from axes
    :param show_sub_titles: whether to show titles of subplots
    :returns: axs
    """
    ind_names = predicted_values.index.intersection(true_values.index)
    col_names = predicted_values.columns.intersection(true_values.columns)
    predicted_values = predicted_values.loc[ind_names, col_names]
    true_values = true_values.loc[ind_names, col_names]

    if len(ind_names) < subplot_ncols:
        num_ncols = len(ind_names)
    else:
        num_ncols = subplot_ncols

    num_nrows = (len(ind_names) - 1) // subplot_ncols + 1
    if adjust_figsize:
        figsize = (figsize[0], figsize[0] * num_nrows / num_ncols)

    fig, axs = plt.subplots(num_nrows, num_ncols, figsize=figsize)
    fig.suptitle(title, fontsize=fontsize_title)
    fig.tight_layout()
    fig.subplots_adjust(top=top, wspace=wspace, hspace=hspace)

    if order is not None:
        ordered_names = pd.Index(order).intersection(ind_names)
        ordered_names = ordered_names.append(pd.Index(ind_names).difference(order).sort_values())
    else:
        ordered_names = ind_names.sort_values()

    for ax, cell in zip(axs.flat, ordered_names):
        if colors_by == 'index' and cell in pallete:
            ax_single_color = pallete[cell]
            ax_pallete = None
        elif colors_by == 'index':
            ax_single_color = single_color
            ax_pallete = None
        else:
            ax_single_color = single_color
            ax_pallete = pallete
        if show_sub_titles:
            sub_title = cell
        else:
            sub_title = ''
        print_cell(predicted_values.loc[cell], true_values.loc[cell], ax=ax, pallete=ax_pallete,
                   single_color=ax_single_color, predicted_name=predicted_name, true_name=true_name, title=sub_title,
                   corr_title=corr_title, s=s, title_font=sub_title_font, labels_font=labels_font,
                   ticks_size=ticks_size, xlim=xlim, ylim=ylim, corr_line=corr_line, linewidth=linewidth,
                   line_color=line_color, pad=pad, min_xlim=min_xlim, min_ylim=min_ylim, labelpad=labelpad)

    return axs
