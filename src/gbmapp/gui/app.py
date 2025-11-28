#!/usr/bin/env python
"""Main GBM GUI Application."""

import os
import tkinter as tk
from tkinter import ttk

from gbmapp.core.models import SimConfig, SimResult
from gbmapp.core.service import GBMService
from .theme import GBMTheme
from .widgets import ConfigPanel
from .plots import GBMPlotter


class GBMMainFrame:
    """Main application frame for GBM stock price predictions."""
    
    def __init__(self, master):
        """Initialize the main application frame.
        
        Args:
            master: Root tkinter window
        """
        self.master = master
        master.config(bg=GBMTheme.BG_COLOR)
        
        # Configure styles
        self.style = GBMTheme.configure_styles(master)
        
        # Initialize data attributes
        self.m_Data = None
        self.sim_config = None
        self.sim_result = None
        
        # Create main layout
        self._create_layout()
        
        # Initialize plotter
        self.plotter = GBMPlotter(self.notebook)
        
        # Create widgets
        self._create_widgets()
        self._setup_tabs()
    
    def _create_layout(self):
        """Create the main two-pane layout."""
        self.master.title("GBM Predictions")
        self.master.minsize(1400, 800)
        
        # Configure main grid
        self.master.columnconfigure(0, weight=0)  # Left pane (config) - fixed width
        self.master.columnconfigure(1, weight=1)  # Right pane (notebook) - expandable
        self.master.rowconfigure(0, weight=1)
        
        # Left pane - Configuration panel
        self.left_pane = ttk.Frame(self.master, style="Custom.TFrame")
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Right pane - Tabbed notebook
        self.notebook = ttk.Notebook(self.master)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Configuration panel
        self.config_panel = ConfigPanel(
            self.left_pane,
            on_load_data=self._on_data_loaded,
            on_run_simulation=self._run_simulation
        )
    
    def _setup_tabs(self):
        """Create and configure notebook tabs."""
        # Tab 1: Historical Prices
        self.tab_historical = ttk.Frame(self.notebook, style="Custom.TFrame")
        self.notebook.add(self.tab_historical, text="Historical Prices")
        
        # Tab 2: GBM Predictions
        self.tab_predictions = ttk.Frame(self.notebook, style="Custom.TFrame")
        self.notebook.add(self.tab_predictions, text="GBM Predictions")
        
        # Tab 3: Results
        self.tab_results = ttk.Frame(self.notebook, style="Custom.TFrame")
        self.notebook.add(self.tab_results, text="Results")
        
        # Configure tab frames
        for tab in [self.tab_historical, self.tab_predictions, self.tab_results]:
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
    
    def _on_data_loaded(self, data, file_name):
        """Callback when data is loaded from CSV.
        
        Args:
            data: Loaded pandas DataFrame
            file_name: Name of the loaded file
        """
        self.m_Data = data
        
        # Plot historical prices in first tab
        self.plotter.plot_actual_prices(self.m_Data, self.tab_historical)
        
        # Switch to historical prices tab
        self.notebook.select(self.tab_historical)
    
    def _run_simulation(self, config: SimConfig):
        """Run GBM simulation with given configuration.
        
        Args:
            config: SimConfig dataclass with simulation parameters
        """
        if self.m_Data is None:
            print("Error: No data loaded")
            return
        
        try:
            # Use service layer to run simulation
            self.sim_config = config
            self.sim_result = GBMService.run_simulation(config, self.m_Data)
            
            # Update UI
            self._display_predictions()
            self._display_results()
            
        except Exception as e:
            print(f"Simulation error: {e}")

    def _display_predictions(self):
        """Display GBM prediction plots in predictions tab."""
        if self.sim_result is None:
            return
        if self.sim_config is None:
            return
        self.plotter.plot_gbm_predictions(
            self.sim_result.display_paths,
            self.m_Data,
            self.sim_config.end_date,
            self.sim_config.steps,
            self.tab_predictions
        )
        
        # Switch to predictions tab
        self.notebook.select(self.tab_predictions)
    
    def _display_results(self):
        """Display simulation results in results tab."""
        if self.sim_result is None:
            return
        if self.sim_config is None:
            return
        
        # Clear existing results
        for widget in self.tab_results.winfo_children():
            widget.destroy()
        
        # Create results display
        results_frame = ttk.Frame(self.tab_results, style="Custom.TFrame")
        results_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Title
        title = ttk.Label(
            results_frame, 
            text="Simulation Results",
            font=("Times New Roman", 16, "bold")
        )
        GBMTheme.configure_label(title, 0, 0)
        
        # Results data
        results_data = [
            ("Engine Used:", self.sim_result.engine_used),
            ("Execution Time:", f"{self.sim_result.elapsed_time:.4f} seconds"),
            ("Total Paths:", f"{self.sim_config.paths:,}"),
            ("Displayed Paths:", f"{self.sim_config.display_paths}"),
            ("Prediction Steps:", f"{self.sim_config.steps}"),
            ("", ""),  # Spacer
            ("Average Predicted Price:", f"${self.sim_result.avg_price:.2f}"),
        ]
        
        # Add actual price and accuracy if available
        if self.sim_result.real_price is not None and self.sim_result.ratio is not None:
            results_data.extend([
                ("Actual Price:", f"${self.sim_result.real_price:.2f}"),
                ("Accuracy Ratio:", f"{self.sim_result.ratio:.4f}"),
                ("Prediction Error:", f"{abs(1 - self.sim_result.ratio) * 100:.2f}%"),
            ])
        else:
            results_data.append(
                ("Actual Price:", "N/A (predicting beyond available data)")
            )
        
        row = 1
        for label_text, value_text in results_data:
            if label_text:  # Skip spacers
                label = ttk.Label(results_frame, text=label_text)
                value = ttk.Label(results_frame, text=value_text)
                GBMTheme.configure_label(label, 0, row)
                GBMTheme.configure_label(value, 1, row)
            row += 1


def main():
    """Initialize and run the GBM application."""
    root = tk.Tk()
    app = GBMMainFrame(root)
    
    def on_close():
        """Clean up resources before closing."""
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
