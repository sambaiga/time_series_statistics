import altair as alt
import seaborn as sns
alt.themes.enable("opaque")
alt.data_transformers.disable_max_rows()
colors=['#4e79a7', '#f28e2b', '#e15759', '#f69a48', '#00c0bf','#fdcd49','#8da798','#a19368','#525252','#a6761d','#7035b7','#cf166e']


def plot_correlation(ax, corr_df, cmap = sns.diverging_palette(240, 10, as_cmap=True)):
    """
    Plots a heatmap of the correlation matrix.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot the heatmap.
        corr_df (pandas.DataFrame): DataFrame containing the correlation data with three columns: two for the pairs of items and one for the correlation values.
        cmap (matplotlib.colors.Colormap, optional): Colormap to use for the heatmap. Default is a diverging palette from Seaborn.

    Returns:
        matplotlib.axes.Axes: The Axes object with the heatmap.
    """
    columns= list(corr_df.columns)
    corr=corr_df.pivot(index=columns[1],
                       columns=columns[0], 
                       values=columns[-1])
    ax=sns.heatmap(corr.T,  linewidths=.5, cmap=cmap, center=0, annot=True, fmt=".1g")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")
    ax.set_title("")
    ax.set_ylabel("")
    ax.set_xlabel("")
    return  ax

def scatter_plot(data, variables, targets, hue_col=None, n_sample=1000, random_state=111):
    
    """
    Creates a scatter plot matrix using Altair.

    Parameters:
        data (pandas.DataFrame): The data to plot.
        variables (list of str): List of column names to be used as variables for the x-axis.
        targets (list of str): List of column names to be used as targets for the y-axis.
        hue_col (str, optional): Column name for the color encoding. Default is None.
        n_sample (int, optional): Number of samples to draw from the data for plotting. Default is 1000.
        random_state (int, optional): Seed for random sampling. Default is 111.

    Returns:
        alt.Chart: The Altair chart object with the scatter plot matrix.
    """
    data = data.sample(n=n_sample,random_state=random_state) if n_sample is not None else data
    chart=alt.Chart(data)
    if hue_col is not None:
        chart=chart.mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color=f'{hue_col}:N')
    else:
        chart=chart.mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'))
    
    chart=chart.properties(
        width=150,
        height=150
    ).repeat(
        row=[targets],
        column=variables
    ).configure_axis(
            grid=False,
            labelFontSize=12,
            titleFontSize=12
        ).configure_view(
            strokeOpacity=0
        )
    return chart

def visualise_timeseries_altair(data,  y_col, figure_path=None, y_label='Power (kW)'):
    """
    Visualizes time series data using Altair.

    Parameters:
    data (pandas.DataFrame): The data to plot, with a datetime index and the columns to be plotted.
    y_col (list of str): List of column names to plot on the y-axis.
    figure_path (str, optional): Path to save the figure. If None, the figure is not saved. Default is None.
    y_label (str, optional): Label for the y-axis. Default is 'Power (kW)'.
    colors (list of str, optional): List of colors for the lines. Default is ['blue', 'red', 'green', 'purple'].

    Returns:
    alt.Chart: The Altair chart object with the time series plot.
    """
    chart = alt.Chart(data.reset_index()).mark_point().encode(
            x=alt.X('timestamp:T', axis=alt.Axis(title='Date')),
            y = alt.X(f'{y_col[0]}:Q', title=y_label),
            color=alt.value(colors[0])
        )
    if len(y_col)>1:
        for i in range(1, len(y_col)):
            chart+=alt.Chart(data.reset_index()).mark_point().encode(
            x=alt.X('timestamp:T', axis=alt.Axis(title='Date')),
            y = alt.X(f'{y_col[i]}:Q', title=y_label),
            color=alt.value(colors[i])
        )
    chart=chart.configure_axis(
        grid=False,
        labelFontSize=12,
        titleFontSize=12
    ).configure_view(
        strokeOpacity=0
    ).properties(width=900,
                    height=100
    )
    
    return chart



def plot_kde_(ax, data, x_col, hue_col, label):
    #sns.kdeplot(data, x=x_col, ax=ax, hue=hue_col,  palette='tab20')
    sns.histplot(data, x=x_col, ax=ax, hue=hue_col,  palette='tab20', kde=True)
    ax.autoscale()
    ax.set_xlabel(label)
    return ax

def plot_cdf_(ax, data, x_col, hue_col, label):
    sns.kdeplot(data, x=x_col, ax=ax, hue=hue_col, cumulative=True, common_norm=False, common_grid=False,  palette='tab20')
    ax.autoscale()
    ax.set_xlabel(label)
    ax.set_ylim(0, 1)
    return ax