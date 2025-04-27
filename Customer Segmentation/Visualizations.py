import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import squarify
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram

from Functions import *

# Correlation Heatmap
def corr_heatmap(correlation):
    """
    This function generates and displays a heatmap of the Spearman's correlation matrix.

    Parameters:
        correlation(pandas.DataFrame): A DataFrame containing the correlation matrix to be visualized.
        
    Returns:
        None: The function displays a heatmap of the correlation matrix and does not return any value.
    """
    plt.figure(figsize = (15, 15))
    ax = sns.heatmap(data = correlation, annot = True, mask = np.triu(np.ones_like(correlation)), cmap = 'RdPu', fmt = '.1')
    ax.set_title("Spearman's Correlation Heatmap", fontdict = {'fontsize': 15})
    plt.show()

# Pie Chart    
def pie_chart(data, variable_to_plot, legend):
    """
    This function generates and displays a pie chart for a specified variable in the dataset.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        variable_to_plot(str): The name of the column for which the pie chart will be created.
        legend(list): A list of labels to be used in the legend for the pie chart.
        
    Returns:
        None: The function displays a pie chart and does not return any value.
    """
    plt.figure(figsize = (15, 15))
    plt.pie(counts(data, variable_to_plot), labels = None, colors = ['palevioletred', 'darkmagenta'], autopct = '%1.1f%%')
    plt.legend(legend, loc = "center left", bbox_to_anchor = (1, 0, 0.5, 1))
    plt.show()
    
# Bar Chart
def bar_chart(data, title, variable_to_plot = None, buying_preferences = False):
    """
    This function generates and displays a bar chart for the given data.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        title(str): The title for the bar chart.
        variable_to_plot(str, optional): The name of the column for which the bar chart will be created. This parameter 
                                        is ignored if `buying_preferences` is True. Defaults to None.
        buying_preferences(bool, optional): If True, the function will plot the bar chart for buying preferences 
                                            (categories and their counts). Default to False.
                                            
    Returns:
        None: The function displays a bar chart and does not return any value.
    """
    fig = go.Figure()
    
    if buying_preferences:
        for category, value in zip(data.index, data['count']):
            fig.add_trace(go.Bar(x = [category], y = [value], name = category, marker_color = 'palevioletred'))
        fig.update_layout(title = title, xaxis_title = 'Category', yaxis_title = 'Count', showlegend = False)
        
    else:    
        fig = px.bar(counts(data, variable_to_plot), x = counts(data, variable_to_plot).index, y = counts(data, variable_to_plot).values, 
                    labels = {'x': f'{variable_to_plot}', 'y': 'Count'},
                    title = title, color_discrete_sequence = ['palevioletred'])
    fig.show()

# Line Chart
def line_chart(data, variable_to_plot, title, gender_distinction = True):
    """
    This function generates and displays a line chart for the given variable, optionally distinguishing by gender.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        variable_to_plot(str): The name of the column for which the line chart will be created.
        title(str): The title for the line chart.
        gender_distinction(bool, optional): If True, the function will plot separate lines for male and female customers.
                                             Defaults to True.
                                             
    Returns:
        None: The function displays a line chart and does not return any value.
    """    
    total_customers = counts(data, variable_to_plot)
    total_males = counts(data[data['customer_gender'] == 'male'], variable_to_plot)
    total_females = counts(data[data['customer_gender'] == 'female'], variable_to_plot)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x = total_customers.index, y = total_customers.values, mode = 'lines+markers', line = dict(color = 'black', width = 2), name = 'Total Customers'))
    
    if gender_distinction:
        fig.add_trace(go.Scatter(x = total_males.index, y = total_males.values, fill = 'tozeroy', line_color = 'darkmagenta', fillcolor = 'rgba(139, 0, 139, 0.5)', name = 'Male'))
        fig.add_trace(go.Scatter(x = total_females.index, y = total_females.values, fill = 'tozeroy', line_color = 'palevioletred', fillcolor = 'rgba(255, 182, 193, 0.5)', name = 'Female'))

    fig.update_layout(
        title = title,
        xaxis_title = variable_to_plot,
        yaxis_title = 'Count',
        xaxis = dict(tickmode = 'array', tickvals = list(range(24)), ticktext = list(range(24))),
        showlegend = True)

    fig.show()

# Histogram
def histogram(data, variable_to_plot, title):
    """
    This function generates and displays a histogram for a specified variable, grouped by gender.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        variable_to_plot(str): The name of the column for which the histogram will be created.
        title(str): The title for the histogram.
        
    Returns:
        None: The function displays a histogram and does not return any value.
    """
    fig = px.histogram(data_frame = data, x = variable_to_plot, color = 'customer_gender', title = 'education', barmode = 'group', color_discrete_map = {'male': 'darkmagenta', 'female': 'palevioletred'})
    fig.update_layout(title = title)
    fig.show()
    
# Treemap
def treemap(data):
    """
    This function generates and displays a treemap of the total money spent by category.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns. Columns representing spending 
                                should have names starting with 'spend_'.
                                
    Returns:
        None: The function displays a treemap and does not return any value.
    """ 
    plt.figure(figsize = (30, 15))
    
    cmap = plt.get_cmap('RdPu')
    color_indices = np.linspace(0, cmap.N, 10, endpoint = False, dtype = int)
    colors = [cmap(i) for i in color_indices]

    non_zero_data = total_money_spent_by_category(data)[total_money_spent_by_category(data)['Total Money Spent'] != 0]
    squarify.plot(sizes = non_zero_data['Total Money Spent'], label = non_zero_data.index, color = colors, alpha = 0.8)
        
    plt.axis('off')

    plt.show()
    
# Scatterplot
def scatterplot(data, column_1, column_2 = None, list_of_columns = None, grid = False):
    """
    This function generates and displays scatterplots to show correlations between specified columns in the dataset.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        column_1(str): The name of the column to be used as the x-axis in the scatterplots.
        column_2(str, optional): The name of the column to be used as the y-axis in a single scatterplot. 
                                Ignored if `grid` is True. Defaults to None.
        list_of_columns(list, optional): A list of column names to be used as the y-axis in multiple scatterplots.
                                        Used only if `grid` is True. Defaults to None.
        grid(boolean, optional): If True, generates a grid of scatterplots using `list_of_columns` as y-axes. 
                               If False, generates a single scatterplot using `column_1` and `column_2`. 
                               Defaults to False.

    Returns:
        None: The function displays scatterplots and does not return any value.
    """
    if grid:
        fig, axes = plt.subplots(3, 3, figsize = (30, 30))
        
        axes = axes.flatten() if len(list_of_columns) > 3 else [axes]

        for ax, column in zip(axes, list_of_columns):
            sns.regplot(x = column_1, y = column, data = data, ax = ax, scatter_kws = {'color': 'palevioletred'}, line_kws = {'color': 'darkmagenta'})
            ax.set_xlabel(f'{column_1}')
            ax.set_ylabel(f'{column}')
            ax.set_title(f'Correlation between {column_1} and {column}')
    else: 
        sns.regplot(x = column_1, y = column_2, data = data, scatter_kws = {'color': 'palevioletred'}, line_kws = {'color': 'darkmagenta'})
        
    plt.tight_layout()
    plt.show()

# Inertia and Silhoutte 
def plot_inertia_and_silhouette(data, k_min = 2, k_max = 15):
    """
    This function generates and displays plots for inertia and silhouette scores over a range of cluster numbers for K-means clustering.

    Parameters:
        data(pandas.DataFrame): The dataset to be clustered.
        k_min(int, optional): The minimum number of clusters to evaluate. Defaults to 2.
        k_max(int, optional): The maximum number of clusters to evaluate. Defaults to 15.

    Returns:
        None: The function displays plots for inertia and silhouette scores and does not return any value
    """
    inertia_values = []
    silhouette_values = []

    k_clusters = range(k_min, k_max + 1)

    for k in k_clusters:
        kmeans = KMeans(n_clusters = k, random_state = 0).fit(data)
        inertia_values.append(kmeans.inertia_)  
        kmeans.predict(data)
        silhouette_values.append(silhouette_score(data, kmeans.labels_, metric = 'euclidean'))  

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(k_clusters, inertia_values, marker = 'o')  
    ax1.set_title('Inertia Plot')
    ax2.plot(k_clusters, silhouette_values, marker = 'o')  
    ax2.set_title('Silhouette Score Plot')
    plt.show()

# Silhouette
def silhoette(data, column) -> None:
    
    cluster_labels = data[column]

    number_clusters = len(cluster_labels.unique())

    silhouette_average = silhouette_score(data, cluster_labels)

    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    fig, ax = plt.subplots()
    fig.set_size_inches(7, 4)

    y_lower = 10
    for i in range(number_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.get_cmap("Spectral")(float(i) / number_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor = color, edgecolor = color, alpha = 0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    ax.axvline(x = silhouette_average, color = "black", linestyle = "--")

    ax.set_yticks([])
    ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_title("Silhouette plot for {} clusters".format(number_clusters))

    plt.show()

    print("Silhouette score for {} clusters: {:.4f}".format(number_clusters, silhouette_average))

# Dendrogram
def create_dendrogram(data, method, threshold = None):
    """
    This function generates and displays a dendrogram for hierarchical clustering using the specified linkage method.

    Parameters:
        data(pandas.DataFrame): The dataset to be clustered.
        method(str): The linkage criterion to use. This can be 'ward', 'complete', 'average', or 'single'.
        threshold(float, optional): The threshold to draw a horizontal line in the dendrogram for visualizing the clustering 
                                    cut-off. Default is None.

    Returns:
        None: The function displays a dendrogram and does not return any value.
    """
    clustering_model = AgglomerativeClustering(linkage = method, distance_threshold = 0, n_clusters = None).fit(data)

    fig, ax = plt.subplots()
    plt.title('Hierarchical Clustering Dendrogram')

    sample_counts = np.zeros(clustering_model.children_.shape[0])
    total_samples = len(clustering_model.labels_)

    for index, nodes in enumerate(clustering_model.children_):
        count = 0
        for node in nodes:
            if node < total_samples:
                count += 1 
            else:
                count += sample_counts[node - total_samples]
        sample_counts[index] = count

    linkage_matrix = np.column_stack([clustering_model.children_, clustering_model.distances_, sample_counts]).astype(float)

    dendrogram(linkage_matrix, truncate_mode = 'level', p = 50)

    if threshold is not None:
        plt.axhline(y = threshold, color = 'black', linestyle = '--')

    plt.show()

# UMAP
def UMAP(transformation, targets):
    """
    This function generates and displays a UMAP (Uniform Manifold Approximation and Projection) plot for visualizing 
    high-dimensional data.

    Parameters:
        transformation(numpy.ndarray): The transformed data obtained from UMAP.
        targets(array-like): The target labels or categories corresponding to each data point.

    Returns:
        None: The function displays the UMAP plot and does not return any value.
    """
    plt.figure(figsize = (15, 12))
    
    labels, targets_categorical = np.unique(targets, return_inverse = True)

    cmap = plt.cm.get_cmap("Spectral")
    norm = plt.Normalize(vmin = 0, vmax = len(labels) - 1)
    plt.scatter(transformation[:, 0], transformation[:, 1], c = targets_categorical, cmap = cmap, norm = norm)

    handles = [plt.scatter([], [], c = cmap(norm(i)), label = label) for i, label in enumerate(labels)]
    plt.legend(handles = handles, title = 'Clusters')

    # Display the plot
    plt.show()

# Pie Chart to visualize the cluster colors
def cluster_colors_pie_chart(colors, labels):
    
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    
    pie_chart = ax.pie([1] * len(colors), colors = colors, labels = labels, startangle = 90, wedgeprops = {'edgecolor': 'black'})
    plt.axis('equal')
    
    plt.show()

# Histograms for each cluster
def cluster_histograms(data, columns_to_include, colors):
    for column in columns_to_include:
        plt.figure(figsize = (12, 8))
        
        for cluster, color in colors.items():
            data_to_plot = data[data['cluster'] == cluster][column]
            sns.histplot(data = data_to_plot, color = color, label = cluster, kde = True, alpha = 0.7, edgecolor = color)

        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title = 'Cluster')
        plt.tight_layout()
        plt.show()

# Bar Plots for each cluster
def cluster_barplots(data, columns_to_include, colors):
    for column in columns_to_include:
        plt.figure(figsize = (30, 8))

        for cluster, color in colors.items():
            data_to_plot = data[data['cluster'] == cluster][column]
            cluster_value_counts = data_to_plot.value_counts()
            for category, count in cluster_value_counts.items():
                plt.bar(str(category) + ' Cluster ' + str(cluster), count, color = color, label = category)

        plt.title(f'Bar Plot of {column} by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average ' + column)
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.show()

# Cluster Sizes
def cluster_sizes(data_1, data_2):
    cluster_sizes = data_1['cluster'].value_counts()
    karen_sizes = data_2.value_counts()

    colors_dictionary = create_cluster_colors_dictionary(['#9e0142', '#d8434e', '#fdbf6f', '#feeea2', '#a2d9a4', '#47a0b3', '#5e4fa2', 'grey'], ['Big Families', 'Big Spenders', 'Drunkards', 'Gamer Community', 'Pet Lovers', 'Savings Squad', 'Veggies Society', 'Fishy Pals'])
    colors = [colors_dictionary[cluster] for cluster in cluster_sizes.index]

    plt.figure(figsize = (14, 6))

    bars_1 = plt.bar(cluster_sizes.index, cluster_sizes, color = colors, edgecolor = 'black')
    bars_2 = plt.bar(karen_sizes.index, karen_sizes, color = 'magenta', label = 'Karens', edgecolor = 'black')
    
    for bar in bars_1:
        y_value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_value, round(y_value, 1), va = 'bottom', ha = 'center', color = 'black', fontweight = 'bold')

    plt.legend()
    plt.show()
