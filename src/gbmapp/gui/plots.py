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
        
        fig = Figure(figsize=(8, 6), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot(data['Date'], data['Close'], color="green", linestyle='-')
        plot.set_facecolor("black")
        plot.set_xlabel('Date', color='white')
        plot.set_ylabel('Close Price', color='white')
        plot.set_title('Historical Prices', color='white')
        plot.tick_params(colors='white')
        plot.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def plot_gbm_predictions(self, walks, data, end_date, steps, parent_frame):
        """Plot GBM prediction walks alongside actual prices.
        
        Args:
            walks: Array of simulated price walks
            data: DataFrame with actual price data
            end_date: End date of training period
            steps: Number of prediction steps
            parent_frame: Frame to place the plot in
        """
        # Clear any existing plot
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        fig, plot = plt.subplots(figsize=(8, 6))
        
        # Get actual prices for comparison
        start_index = data.index[data['Date'] == end_date][0]
        real_prices = data.iloc[start_index:start_index + int(steps)]
        
        # Plot settings
        plot.set_facecolor("black")
        fig.patch.set_facecolor("black")
        
        # Plot subset of walks (max 100 for performance)
        for walk in walks:
            plot.plot(real_prices['Date'], walk, color='green', 
                     alpha=0.1, linewidth=0.5)
        
        # Overlay actual prices on top
        plot.plot(real_prices['Date'], real_prices['Close'], 
                 label='Real Prices', color='white', 
                 linewidth=2, zorder=len(walks) + 1)
        
        plot.set_xlabel('Date', color='white')
        plot.set_ylabel('Price', color='white')
        plot.set_title('GBM Predictions vs Actual', color='white')
        plot.tick_params(colors='white')
        plot.grid(True, alpha=0.3)
        plot.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
