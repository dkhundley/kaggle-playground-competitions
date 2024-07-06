# Importing the necessary Python libraries
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



# Setting the hex code color values
hex_colors = ['#03045E', '#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8']



def set_color_theme(hex_colors = hex_colors):
    '''
    Sets the Seaborn color theme
    
    Inputs:
        - hex_colors (list): A list of hexidecimal values representing each individual color of the color palette
    
    Returns:
        - N/A
    '''
    
    # Setting the color palette
    sns.set_palette(palette = hex_colors)



def display_binary_counts(binary_feat):
    '''
    Displays a bar count chart and pie chart associated to a binary input feature
    
    Inputs:
        - binary_feat (Polars Series): The binary feature from the Polars DataFrame
        
    Returns:
        - N/A (Displays a bar chart)
    '''
    
    # Getting the counts of each binary class
    binary_counts = binary_feat.value_counts().sort(by = 'count', descending = True)
    
    # Extracting the name of the feature
    feat_name = str(binary_counts.columns[0])
    
    # Creating a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))

    # Displaying a bar chart of the binary count distribution
    ax1.bar(x = binary_counts[feat_name],
            height = binary_counts['count'])
    
    # Annotating the bar chart with information
    ax1.set_title(f'{feat_name} Count Distribution', fontsize = 16)
    ax1.set_xlabel(feat_name)
    ax1.set_ylabel('Count')
    for i, count in enumerate(binary_counts['count']):
        ax1.text(x = i, y = count, s = str(count), ha = 'center', va = 'bottom')
        
    # Displaying a pie chart of the binary count distribution
    ax2.pie(binary_counts['count'],
            labels = binary_counts[feat_name],
            autopct = '%1.1f%%',
            startangle = 90)
    
    # Annotating the pie chart with information
    ax2.set_title(f'{feat_name} Pie Chart', fontsize = 16)
    
    # Adding a circle to the middle of the pie chart to make it look more like a donut
    ax2.add_artist(plt.Circle(xy = (0, 0), radius = 0.70, fc = 'white'))
    
    # Ensuring that the pie chart is circular
    ax2.axis('equal')

    # Displaying the visuals
    plt.tight_layout()
    plt.show()



def display_continuous_distributions(continuous_feat):
    '''
    Displays a variety of visuals associated to a continuous input feature
    
    Inputs:
        - continuous_feat (Polars Series): The continuous feature to build visualizations around
        
    Returns:
        - N/A (Only displays visuals)
    '''
    # Extracting the feature name
    feat_name = continuous_feat.name
    
    # Setting up the figure and subplots
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 20))
    fig.suptitle(f'Distribution Analysis: {feat_name}', fontsize = 16)
    
    # Defining a histogram
    axs[0, 0].hist(continuous_feat, bins = 20)
    axs[0, 0].set_title('Histogram')
    axs[0, 0].set_xlabel(feat_name)
    axs[0, 0].set_ylabel('Frequency')
    
    # Defining a box plot
    axs[0, 1].boxplot(continuous_feat)
    axs[0, 1].set_title('Box Plot')
    axs[0, 1].set_ylabel(feat_name)
    
    # Defining a KDE Plot
    sns.kdeplot(continuous_feat, fill = True, ax = axs[1, 0])
    axs[1, 0].set_title('KDE Plot')
    axs[1, 0].set_xlabel(feat_name)
    axs[1, 0].set_ylabel('Density')
    
    # Defining the summary statistics
    summary_stats = continuous_feat.describe()
    summary_text = '\n'.join([f'{stat}: {value:.2f}' for stat, value in summary_stats.iter_rows()])
    axs[1, 1].text(0.1, 0.5, summary_text, fontsize = 12, va = 'center')
    axs[1, 1].set_title('Summary Statistics')
    axs[1, 1].axis('off')
    
    # Displaying the visuals
    plt.tight_layout()
    plt.show()



def display_categorical_visualizations(cat_feat, top_n = 10):
    '''
    Displays a variety of visuals associated to a categorical input feature
    
    Inputs:
        - cat_feat (Polars Series): The categorical input feature to be analyzed
        - top_n (int): The number of top categorical values we want to display for certain visuals
        
    Returns:
        - N/A (Only displays visuals)
    '''
    
    # Extracting the name of the feature
    feat_name = cat_feat.name
    
    # Setting up the figure and subplots
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
    fig.suptitle(f'Categorical Feature Analysis: {feat_name}', fontsize = 16)
    
    # Defining the bar plot
    cat_feat_counts = cat_feat.value_counts().sort(by = 'count', descending = True).to_pandas()
    sns.barplot(x = cat_feat_counts[feat_name], y = cat_feat_counts['count'], ax = axs[0])
    axs[0].set_title('Frequency of Categories')
    axs[0].set_xlabel(feat_name)
    axs[0].set_ylabel('Frequency')
    
    # Defining the pie chart
    top_categories = cat_feat.value_counts().head(top_n)
    other_categories = cat_feat.value_counts()['count'].sum() - top_categories['count'].sum()
    pie_data = pl.DataFrame({
        feat_name: top_categories[feat_name].append(pl.Series([float('nan')], dtype=pl.Float64)),
        'counts': top_categories['count'].append(pl.Series([other_categories], dtype=pl.UInt32))
    })
    pie_data = pie_data.with_columns([
        pl.when(pl.col(feat_name).is_nan())
        .then(pl.lit('Other'))
        .otherwise(pl.col(feat_name).cast(pl.Utf8))
        .alias(feat_name)
    ])
    axs[1].pie(pie_data['counts'], labels=pie_data[feat_name], autopct='%1.1f%%')
    axs[1].set_title(f'Top Categories')

    # Displaying the visuals
    plt.tight_layout()
    plt.show()
    
     # Defining the treemap
    treemap = px.treemap(cat_feat_counts, path = [feat_name], values = 'count')
    treemap.update_layout(title = f'Treemap Distribution for {feat_name}')
    treemap.show()