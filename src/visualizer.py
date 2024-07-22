import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def create_histogram(self, data, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=column, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        return plt.gcf()

    def create_scatter_plot(self, data, x_column, y_column):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_column, y=y_column)
        plt.title(f'Scatter Plot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        return plt.gcf()

    def create_line_chart(self, data, x_column, y_column):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x=x_column, y=y_column)
        plt.title(f'Line Chart of {y_column} over {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        return plt.gcf()

    def create_correlation_heatmap(self, data):
        plt.figure(figsize=(12, 10))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        return plt.gcf()

    def create_box_plot(self, data, column):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, y=column)
        plt.title(f'Box Plot of {column}')
        plt.ylabel(column)
        return plt.gcf()

    def create_pair_plot(self, data):
        plt.figure(figsize=(12, 10))
        sns.pairplot(data)
        plt.tight_layout()
        return plt.gcf()