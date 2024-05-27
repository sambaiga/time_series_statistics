import altair as alt
import seaborn as sns
alt.themes.enable("opaque")
alt.data_transformers.disable_max_rows()



def plot_correlation(ax, corr_df, cmap = sns.diverging_palette(240, 10, as_cmap=True)):
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
    chart=alt.Chart(data.sample(n=1000,random_state=random_state))
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