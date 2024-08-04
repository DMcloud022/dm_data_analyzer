import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional, List, Union

class Visualizer:
    def __init__(self, config: Dict[str, Any], logger: Any, error_handler: Any):
        self.config = config
        self.logger = logger
        self.error_handler = error_handler

    def create_histogram(self, data: pd.DataFrame, column: str, bins: Optional[int] = None, 
                         color: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        try:
            fig = px.histogram(data, x=column, color=color, nbins=bins, title=title)
            fig.update_layout(bargap=0.1)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_histogram: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_scatter_plot(self, data: pd.DataFrame, x_column: str, y_column: str, 
                            color: Optional[str] = None, size: Optional[str] = None,
                            title: Optional[str] = None) -> go.Figure:
        try:
            fig = px.scatter(data, x=x_column, y=y_column, color=color, size=size, title=title)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_scatter_plot: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_line_chart(self, data: pd.DataFrame, x_column: str, y_column: Union[str, List[str]], 
                          color: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        try:
            fig = px.line(data, x=x_column, y=y_column, color=color, title=title)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_line_chart: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_correlation_heatmap(self, data: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
        try:
            corr = data.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title=title)
            fig.update_layout(width=800, height=800)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_correlation_heatmap: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_box_plot(self, data: pd.DataFrame, column: str, group: Optional[str] = None, 
                        title: Optional[str] = None) -> go.Figure:
        try:
            fig = px.box(data, y=column, x=group, title=title)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_box_plot: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_pair_plot(self, data: pd.DataFrame, dimensions: List[str], 
                         color: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        try:
            fig = px.scatter_matrix(data, dimensions=dimensions, color=color, title=title)
            fig.update_layout(width=1000, height=1000)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_pair_plot: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_bar_chart(self, data: pd.DataFrame, x_column: str, y_column: str, 
                         color: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        try:
            fig = px.bar(data, x=x_column, y=y_column, color=color, title=title)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_bar_chart: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_pie_chart(self, data: pd.DataFrame, names: str, values: str, 
                         title: Optional[str] = None) -> go.Figure:
        try:
            fig = px.pie(data, names=names, values=values, title=title)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_pie_chart: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_distribution_plot(self, data: pd.DataFrame, column: str, 
                                 color: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        try:
            fig = px.histogram(data, x=column, color=color, marginal="box", title=title)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_distribution_plot: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)

    def create_multi_plot(self, data: pd.DataFrame, plot_specs: List[Dict[str, Any]], 
                          rows: int, cols: int, title: Optional[str] = None) -> go.Figure:
        try:
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=[spec.get('title') for spec in plot_specs])
            
            for i, spec in enumerate(plot_specs):
                row = i // cols + 1
                col = i % cols + 1
                plot_type = spec['type']
                
                if plot_type == 'scatter':
                    trace = go.Scatter(x=data[spec['x']], y=data[spec['y']], mode='markers', name=spec.get('name'))
                elif plot_type == 'line':
                    trace = go.Scatter(x=data[spec['x']], y=data[spec['y']], mode='lines', name=spec.get('name'))
                elif plot_type == 'bar':
                    trace = go.Bar(x=data[spec['x']], y=data[spec['y']], name=spec.get('name'))
                elif plot_type == 'box':
                    trace = go.Box(y=data[spec['y']], name=spec.get('name'))
                else:
                    raise ValueError(f"Unsupported plot type: {plot_type}")
                
                fig.add_trace(trace, row=row, col=col)
            
            fig.update_layout(height=300*rows, width=400*cols, title_text=title)
            return fig
        except Exception as e:
            self.logger.log_error(f"Error in create_multi_plot: {str(e)}")
            raise self.error_handler.handle_visualization_error(e)