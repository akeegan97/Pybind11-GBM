#!/usr/bin/env python
"""Plotting utilities for GBM GUI application."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class GBMPlotter:
    """Handles all plotting functionality for the GBM application."""
    
    def __init__(self, notebook):
        """Initialize plotter.
        
        Args:
            notebook: Parent notebook widget
        """
        self.notebook = notebook
    
    def plot_actual_prices(self, data, parent_frame):
        """Plot historical price data from CSV.
        
        Args:
            data: DataFrame with 'Date' and 'Close' columns
            parent_frame: Frame to place the plot in
        """
        # Clear any existing plot
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(10, 6), dpi=100)
        fig.patch.set_facecolor("black")
        plot = fig.add_subplot(111)
        
        plot.plot(data['Date'], data['Close'], color="green", linestyle='-', linewidth=2)
        plot.set_facecolor("black")
        plot.set_xlabel('Date', color='white', fontsize=12)
        plot.set_ylabel('Close Price ($)', color='white', fontsize=12)
        plot.set_title('Historical Stock Prices', color='white', fontsize=14, fontweight='bold')
        plot.tick_params(colors='white', labelsize=10)
        plot.grid(True, alpha=0.3, color='white')
        
        # Rotate date labels for better readability
        fig.autofmt_xdate()
        
        # Ensure tight layout to show labels
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def plot_gbm_predictions(self, walks, data, end_date, steps, parent_frame):
        """Plot GBM prediction walks alongside actual prices.
        
        Args:
            walks: Array/list of simulated price walks (each walk is a list of prices)
            data: DataFrame with actual price data
            end_date: End date of training period
            steps: Number of prediction steps
            parent_frame: Frame to place the plot in
        """
        # Clear any existing plot
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(10, 6), dpi=100)
        fig.patch.set_facecolor("black")
        plot = fig.add_subplot(111)
        
        # Convert walks to numpy array if it's a list
        walks_array = np.array(walks) if not isinstance(walks, np.ndarray) else walks
        
        # Get actual prices for comparison
        import pandas as pd
        end_date_obj = pd.to_datetime(end_date)
        start_index = data.index[data['Date'] == end_date_obj][0]
        real_prices = data.iloc[start_index:start_index + int(steps) + 1]  # +1 to include start
        
        # Plot settings
        plot.set_facecolor("black")
        
        # Create x-axis based on actual walk length (not steps)
        walk_length = len(walks_array[0]) if len(walks_array) > 0 else int(steps)
        x_values = range(walk_length)  # Use step indices instead of dates
        
        # Plot GBM walks (green with transparency)
        num_walks_to_plot = min(len(walks_array), 100)  # Limit for performance
        for i in range(num_walks_to_plot):
            walk = walks_array[i]
            plot.plot(x_values, walk, color='green', 
                     alpha=0.2, linewidth=0.8)
        
        # Overlay actual prices on top (white, thicker) - use same x-axis length
        if len(real_prices) > 0:
            # Match the length - either truncate real prices or pad
            actual_x = range(min(len(real_prices), walk_length))
            actual_y = real_prices['Close'].values[:len(actual_x)]
            plot.plot(actual_x, actual_y, 
                     label='Actual Price', color='white', 
                     linewidth=2.5, zorder=1000)
        
        plot.set_xlabel('Time Steps', color='white', fontsize=12)
        plot.set_ylabel('Price ($)', color='white', fontsize=12)
        plot.set_title(f'GBM Predictions ({num_walks_to_plot} paths shown) vs Actual Price', 
                      color='white', fontsize=14, fontweight='bold')
        plot.tick_params(colors='white', labelsize=10)
        plot.grid(True, alpha=0.3, color='white')
        plot.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=10)
        
        # Ensure tight layout
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
