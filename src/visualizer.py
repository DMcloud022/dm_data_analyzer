import plotly.express as px
import plotly.graph_objects as go

class Visualizer:
    def __init__(self, config, logger, error_handler):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler

    def create_histogram(self, data, column):
        fig = px.histogram(data, x=column)
        return fig

    def create_scatter_plot(self, data, x_column, y_column):
        fig = px.scatter(data, x=x_column, y=y_column)
        return fig

    def create_line_chart(self, data, x_column, y_column):
        fig = px.line(data, x=x_column, y=y_column)
        return fig

    def create_correlation_heatmap(self, data):
        fig = px.imshow(data.corr())
        return fig

    def create_box_plot(self, data, column):
        fig = px.box(data, y=column)
        return fig

    def create_pair_plot(self, data):
        fig = px.scatter_matrix(data)
        return fig

    def create_bar_chart(self, data, x_column, y_column):
        fig = px.bar(data, x=x_column, y=y_column)
        return fig

    def create_pie_chart(self, data, names, values):
        fig = px.pie(data, names=names, values=values)
        return fig

    def create_distribution_plot(self, data, column):
        fig = px.histogram(data, x=column, marginal="box")
        return fig