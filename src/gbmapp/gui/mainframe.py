#!/usr/bin/env python
"""GBM Prediction GUI Application."""

import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkcalendar import Calendar

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from pylogic import utilities
import simulation
class GBMApp:
    """Main GUI application for GBM stock price predictions."""
    
    # Color scheme constants
    BG_COLOR = "#000000"
    FG_COLOR = "#05ff3f"
    FONT = ("Times New Roman", 12)
    
    def __init__(self, master):
        self.master = master
        master.config(bg=self.BG_COLOR)
        
        # Configure styles
        self.style = ttk.Style(master)
        self.style.configure("Custom.TFrame", background=self.BG_COLOR, foreground=self.FG_COLOR)
        self.style.configure("Button.TButton", font=self.FONT, background=self.BG_COLOR, foreground=self.FG_COLOR)


        self.StartDateFrame = ttk.Frame(master)
        self.EndDateFrame = ttk.Frame(master)
        self.PredictionFrame = ttk.Frame(master)

        # Initialize data attributes
        self.m_Data = None
        self.m_StartDate = ""
        self.m_EndDate = ""
        self.m_Steps = ""
        self.m_Paths = 100_000_000
        
        # Create widgets
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # File selection widgets
        self.m_ButtonLabel = ttk.Label(self.master, text="Select CSV file:")
        self.m_FileBrowseButton = ttk.Button(self.master, text="Browse", command=self.browse_file)
        
        # Start date widgets
        self.m_StartDateLabel = ttk.Label(self.StartDateFrame, text="Start Date of Training")
        self.m_StartDateVerifyLabel = ttk.Label(self.StartDateFrame, text="Is Date Valid: ")
        self.m_StartTrainingDate = Calendar(
            self.master, selectmode='day', 
            background=self.BG_COLOR, foreground=self.FG_COLOR, font=self.FONT
        )
        self.m_StartTrainingDate.bind("<<CalendarSelected>>", self._update_start_date_valid)
        
        # End date widgets
        self.m_EndDateLabel = ttk.Label(self.EndDateFrame, text="End Date of Training")
        self.m_EndDateVerifyLabel = ttk.Label(self.EndDateFrame, text="Is Date Valid: ")
        self.m_EndTrainingDate = Calendar(
            self.master, selectmode='day',
            background=self.BG_COLOR, foreground=self.FG_COLOR, font=self.FONT
        )
        self.m_EndTrainingDate.bind("<<CalendarSelected>>", self._combined_call_for_binding)
        
        # Prediction days widgets
        self.m_PredictionDaysLabel = ttk.Label(self.PredictionFrame, text="Prediction Days")
        self.m_PredictionStepsValid = ttk.Label(self.PredictionFrame, text="Prediction Days Valid: ")
        self.m_PredictionDaysSpinBox = ttk.Spinbox(self.master, from_=0, to=0)
        for event in ["<KeyRelease>", "<<Increment>>", "<<Decrement>>", "<FocusOut>"]:
            self.m_PredictionDaysSpinBox.bind(event, self._update_steps)
        
        # Calculation button
        self.m_StartCalculationButton = ttk.Button(
            self.master, text="Start Prediction", state="disabled", command=self._calculate
        )
        
        # Data label
        self.m_CSVDataLabel = ttk.Label(self.master, text="Data Name: ")

    def _setup_layout(self):
        """Configure the grid layout and widget placement."""
        self.master.title("GBM Predictions")
        self.master.minsize(1200, 600)
        
        # Configure grid weights for resizing
        self.master.columnconfigure(0, weight=0)
        self.master.columnconfigure(1, weight=0)
        self.master.columnconfigure(2, weight=1)
        self.master.columnconfigure(3, weight=1)
        self.master.rowconfigure(0, weight=0)
        self.master.rowconfigure(1, weight=1)
        self.master.rowconfigure(2, weight=0)
        self.master.rowconfigure(3, weight=0)
        self.master.rowconfigure(4, weight=0)
        
        # Configure frame styles
        self.StartDateFrame.config(style="Custom.TFrame")
        self.EndDateFrame.config(style="Custom.TFrame")
        self.PredictionFrame.config(style="Custom.TFrame")
        
        # File selection row
        self._configure_label(self.m_ButtonLabel, column=0, row=0)
        self.m_FileBrowseButton.grid(column=1, row=0, padx=5, pady=5, sticky="w")
        self.m_FileBrowseButton.config(style="Button.TButton")
        
        # Start date frame
        self.StartDateFrame.grid(column=0, row=1, padx=5, pady=5, sticky="nw")
        self._configure_label(self.m_StartDateLabel, column=0, row=0)
        self._configure_label(self.m_StartDateVerifyLabel, column=0, row=1)
        self.m_StartTrainingDate.grid(column=1, row=1, padx=5, pady=5, sticky="nw")
        
        # End date frame
        self.EndDateFrame.grid(column=0, row=2, padx=5, pady=5, sticky="nw")
        self._configure_label(self.m_EndDateLabel, column=0, row=0)
        self._configure_label(self.m_EndDateVerifyLabel, column=0, row=1)
        self.m_EndTrainingDate.grid(column=1, row=2, padx=5, pady=5, sticky="nw")
        
        # Prediction frame
        self.PredictionFrame.grid(column=0, row=3, padx=5, pady=5, sticky="nw")
        self._configure_label(self.m_PredictionDaysLabel, column=0, row=0)
        self._configure_label(self.m_PredictionStepsValid, column=0, row=1)
        self.m_PredictionDaysSpinBox.grid(column=1, row=3, padx=5, pady=5, sticky="w")
        self.m_PredictionDaysSpinBox.config(font=self.FONT, background=self.BG_COLOR, foreground=self.FG_COLOR)
        
        # Calculation button
        self.m_StartCalculationButton.grid(column=1, row=4, padx=5, pady=10, sticky="w")
        self.m_StartCalculationButton.config(style="Button.TButton")
        
        # Data label
        self._configure_label(self.m_CSVDataLabel, column=2, row=0)
    
    def _configure_label(self, label, column, row):
        """Helper method to configure label appearance and placement."""
        label.grid(column=column, row=row, padx=5, pady=2, sticky="w")
        label.config(font=self.FONT, background=self.BG_COLOR, foreground=self.FG_COLOR)

    def _update_start_date_valid(self, event=None):
        """Validate and update the start date selection."""
        if self.m_Data is None:
            return
        
        self.m_StartDate = self.m_StartTrainingDate.get_date()
        is_valid = utilities.VerifyStartDate(self.m_Data, self.m_StartDate)
        self.m_StartDateVerifyLabel.config(text="Date Is Valid" if is_valid else "Date Not Valid")
    
    def _combined_call_for_binding(self, event=None):
        """Combined event handler for end date changes."""
        self._update_end_date_valid(event)
        self._update_prediction_range(event)
        self._toggle_start_calculation_button()
    
    def _update_end_date_valid(self, event=None):
        """Validate and update the end date selection."""
        if self.m_Data is None:
            return
        
        self.m_EndDate = self.m_EndTrainingDate.get_date()
        is_valid = utilities.VerifyEndDate(self.m_Data, self.m_EndDate)
        self.m_EndDateVerifyLabel.config(text="Date Is Valid" if is_valid else "Date Not Valid")
    
    def _update_prediction_range(self, event=None):
        """Update the valid range for prediction days based on end date."""
        if self.m_Data is None:
            return
        
        top_of_range = utilities.GetEndRange(self.m_Data, self.m_EndDate)
        if top_of_range is not None:
            self.m_PredictionDaysSpinBox.config(to=top_of_range)
    
    def _toggle_start_calculation_button(self):
        """Enable or disable the calculation button based on input validity."""
        if self.m_Data is None or not utilities.CheckStartEndDate(self.m_StartDate, self.m_EndDate):
            self.m_StartCalculationButton.config(state="disabled")
        else:
            self.m_StartCalculationButton.config(state="normal")
            self.m_Steps = self.m_PredictionDaysSpinBox.get()
    
    def _update_steps(self, event=None):
        """Update the number of prediction steps."""
        self.m_Steps = self.m_PredictionDaysSpinBox.get()

    def browse_file(self):
        """Open file dialog and load CSV data."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        
        try:
            self.file_path = file_path
            self.m_Data = utilities.ReadCsvData(file_path)
            
            if self.m_Data is not None:
                # Update UI
                file_name = os.path.basename(file_path)
                self.m_CSVDataLabel.config(text=f"Data Name: {file_name}")
                
                # Plot the data
                self._plot_actual_prices()
                
                # Trigger validation for dates if already selected
                self._update_start_date_valid()
                self._update_end_date_valid()
                self._update_prediction_range()
                self._toggle_start_calculation_button()
                
                print(f"Successfully loaded {len(self.m_Data)} rows from {file_name}")
            else:
                self.m_CSVDataLabel.config(text="Data Name: Failed to load")
                print(f"Error: Failed to load CSV file {file_path}")
        except Exception as e:
            self.m_CSVDataLabel.config(text="Data Name: Error")
            print(f"Error loading file: {e}")
    
    def _plot_actual_prices(self):
        """Plot the historical price data from the loaded CSV."""
        # Clear any existing plot
        for widget in self.master.grid_slaves(column=2, row=1):
            widget.destroy()
        
        fig = Figure(figsize=(5, 5), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot(self.m_Data['Date'], self.m_Data['Close'], color="green", linestyle='-')
        plot.set_facecolor("black")
        plot.set_xlabel('Date')
        plot.set_ylabel('Close')
        plot.set_title('Actual Prices')
        plot.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().grid(column=2, row=1, rowspan=3, sticky="nsew", padx=5, pady=5)
    
    def _plot_gbm_predictions(self, walks, end_date):
        """Plot GBM prediction walks alongside actual prices."""
        # Clear any existing plot
        for widget in self.master.grid_slaves(column=3, row=1):
            widget.destroy()
        
        fig, plot = plt.subplots(figsize=(8, 6))
        
        # Get actual prices for comparison
        start_index = self.m_Data.index[self.m_Data['Date'] == end_date][0]
        real_prices = self.m_Data.iloc[start_index:start_index + int(self.m_Steps)]
        
        # Plot settings
        plot.set_facecolor("black")
        fig.patch.set_facecolor("black")
        
        # Plot subset of walks (max 100 for performance to reduce lag)
        subset_walks = walks[:min(100, len(walks))]
        for walk in subset_walks:
            plot.plot(real_prices['Date'], walk, alpha=0.2, linewidth=0.5)
        
        # Overlay actual prices on top
        plot.plot(real_prices['Date'], real_prices['Close'], 
                 label='Real Prices', color='white', linewidth=2, zorder=len(walks) + 1)
        
        plot.set_xlabel('Date', color='white')
        plot.set_ylabel('Price', color='white')
        plot.set_title('GBM Predictions', color='white')
        plot.tick_params(colors='white')
        plot.grid(True, alpha=0.3)
        plot.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().grid(column=3, row=1, rowspan=3, sticky="nsew", padx=5, pady=5)

    def _calculate(self):
        """Calculate and display GBM predictions."""
        # Calculate statistics from training data
        stats = utilities.CalculateStatistics(
            self.m_Data, self.m_StartDate, self.m_EndDate, self.m_Steps
        )
        print(f"Statistics - μ: {stats.normalizedMu}, σ²: {stats.normalizedVariance}, "
              f"σ: {stats.normalizedDeviation}")
        
        # Get starting price and actual future price for comparison
        start_index = self.m_Data.index[self.m_Data['Date'] == self.m_EndDate][0]
        end_index = start_index + int(self.m_Steps)
        compared_data = self.m_Data.iloc[start_index:end_index]
        real_price = compared_data['Close'].iloc[-1]
        starting_price = self.m_Data.loc[start_index - 1, 'Close']
        
        # Run multi-threaded simulation
        start_time = time.perf_counter()
        walks, average_price = simulation.SimulateGBMMultiThreaded(
            starting_price, stats.normalizedMu, stats.normalizedVariance,
            stats.normalizedDeviation, int(self.m_Steps), self.m_Paths
        )
        elapsed = time.perf_counter() - start_time
        print(f"C++ MultiThreaded version: {elapsed:.4f} seconds")
        
        # Run intrinsic multi-threaded simulation
        start_time = time.perf_counter()
        walks, average_price = simulation.SimulateGBMIntrinsicMT(
            starting_price, stats.normalizedMu, stats.normalizedVariance,
            stats.normalizedDeviation, int(self.m_Steps), self.m_Paths
        )
        elapsed = time.perf_counter() - start_time
        print(f"C++ MultiThreaded Intrinsic version: {elapsed:.4f} seconds")
        
        # Display results
        np_paths = np.array(walks)
        self._plot_gbm_predictions(np_paths, self.m_EndDate)
        
        print(f"Average predicted price: ${average_price:.2f}")
        print(f"Actual price: ${real_price:.2f}")
        print(f"Prediction accuracy ratio: {average_price/real_price:.4f}")



def main():
    """Initialize and run the GBM application."""
    root = tk.Tk()
    app = GBMApp(root)
    
    def on_close():
        """Clean up resources before closing."""
        plt.close('all')
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()