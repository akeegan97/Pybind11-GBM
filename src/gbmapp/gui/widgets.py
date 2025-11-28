#!/usr/bin/env python
"""Custom widgets for GBM GUI application."""

import os
import tkinter as tk
from tkinter import filedialog, ttk
from tkcalendar import Calendar

from .theme import GBMTheme
from gbmapp.core.models import SimConfig
from gbmapp.core.service import GBMService

class DateSelector:
    """Date selection widget with validation."""
    
    def __init__(self, master, label_text, validation_callback):
        """Initialize date selector.
        
        Args:
            master: Parent tkinter widget
            label_text: Label to display above calendar
            validation_callback: Callback function when date changes
        """
        self.frame = ttk.Frame(master)
        self.frame.config(style="Custom.TFrame")
        
        self.label = ttk.Label(self.frame, text=label_text)
        self.verify_label = ttk.Label(self.frame, text="Is Date Valid: ")
        
        self.calendar = Calendar(
            master,
            selectmode='day',
            background=GBMTheme.BG_COLOR,
            foreground=GBMTheme.FG_COLOR,
            font=GBMTheme.FONT
        )
        
        self.validation_callback = validation_callback
        
    def bind_event(self, callback):
        """Bind calendar selection event.
        
        Args:
            callback: Function to call on date selection
        """
        self.calendar.bind("<<CalendarSelected>>", callback)
    
    def get_date(self):
        """Get the selected date.
        
        Returns:
            Selected date from calendar
        """
        return self.calendar.get_date()
    
    def update_validation_status(self, is_valid):
        """Update the validation label.
        
        Args:
            is_valid: Boolean indicating if date is valid
        """
        self.verify_label.config(
            text="Date Is Valid" if is_valid else "Date Not Valid"
        )
    
    def grid_frame(self, column, row):
        """Position the frame in grid.
        
        Args:
            column: Grid column
            row: Grid row
        """
        self.frame.grid(column=column, row=row, padx=5, pady=5, sticky="nw")
        
    def grid_widgets(self, calendar_row):
        """Position widgets within frame and parent.
        
        Args:
            calendar_row: Row position for calendar in parent
        """
        GBMTheme.configure_label(self.label, column=0, row=0)
        GBMTheme.configure_label(self.verify_label, column=0, row=1)
        self.calendar.grid(column=1, row=calendar_row, padx=5, pady=5, sticky="nw")

class PredictionDaysSelector:
    """Widget for selecting number of prediction days."""
    
    def __init__(self, master, update_callback):
        """Initialize prediction days selector.
        
        Args:
            master: Parent tkinter widget
            update_callback: Callback when value changes
        """
        self.frame = ttk.Frame(master)
        self.frame.config(style="Custom.TFrame")
        
        self.label = ttk.Label(self.frame, text="Prediction Days")
        self.valid_label = ttk.Label(self.frame, text="Prediction Days Valid: ")
        
        self.spinbox = ttk.Spinbox(master, from_=0, to=0)
        self.spinbox.config(
            font=GBMTheme.FONT,
            background=GBMTheme.BG_COLOR,
            foreground=GBMTheme.FG_COLOR
        )
        
        self.update_callback = update_callback
        
    def bind_events(self, callback):
        """Bind events to spinbox.
        
        Args:
            callback: Function to call on value change
        """
        for event in ["<KeyRelease>", "<<Increment>>", "<<Decrement>>", "<FocusOut>"]:
            self.spinbox.bind(event, callback)
    
    def get_value(self):
        """Get the current spinbox value.
        
        Returns:
            Current value as string
        """
        return self.spinbox.get()
    
    def set_range(self, max_value):
        """Set the valid range for prediction days.
        
        Args:
            max_value: Maximum allowed value
        """
        self.spinbox.config(**{'to': max_value})
    
    def grid_frame(self, column, row):
        """Position the frame in grid.
        
        Args:
            column: Grid column
            row: Grid row
        """
        self.frame.grid(column=column, row=row, padx=5, pady=5, sticky="nw")
    
    def grid_widgets(self, spinbox_row):
        """Position widgets within frame and parent.
        
        Args:
            spinbox_row: Row position for spinbox in parent
        """
        GBMTheme.configure_label(self.label, column=0, row=0)
        GBMTheme.configure_label(self.valid_label, column=0, row=1)
        self.spinbox.grid(column=1, row=spinbox_row, padx=5, pady=5, sticky="w")

class FileBrowser:
    """File browser widget for CSV selection."""
    
    def __init__(self, master, browse_callback):
        """Initialize file browser.
        
        Args:
            master: Parent tkinter widget
            browse_callback: Callback when file is selected
        """
        self.label = ttk.Label(master, text="Select CSV file:")
        self.button = ttk.Button(master, text="Browse", command=browse_callback)
        self.button.config(style="Button.TButton")
        
    def browse_file(self):
        """Open file dialog and return selected file path.
        
        Returns:
            Selected file path or None
        """
        return filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    def grid_widgets(self):
        """Position widgets in grid."""
        GBMTheme.configure_label(self.label, column=0, row=0)
        self.button.grid(column=1, row=0, padx=5, pady=5, sticky="w")


class ConfigPanel:
    """Configuration panel for simulation parameters."""
    
    def __init__(self, parent, on_load_data, on_run_simulation):
        """Initialize configuration panel.
        
        Args:
            parent: Parent widget
            on_load_data: Callback when data is loaded (data, file_name)
            on_run_simulation: Callback to run simulation (SimConfig)
        """
        self.parent = parent
        self.on_load_data = on_load_data
        self.on_run_simulation = on_run_simulation
        
        self.data = None
        
        # Create scrollable frame
        self.canvas = tk.Canvas(parent, bg=GBMTheme.BG_COLOR, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style="Custom.TFrame")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self._create_widgets()
        self._layout_widgets()
    
    def _create_widgets(self):
        """Create all configuration widgets."""
        row = 0
        
        # Title
        title = ttk.Label(
            self.scrollable_frame,
            text="Configuration",
            font=("Times New Roman", 14, "bold")
        )
        GBMTheme.configure_label(title, 0, row)
        row += 1
        
        # File loading section
        separator1 = ttk.Separator(self.scrollable_frame, orient='horizontal')
        separator1.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        file_label = ttk.Label(self.scrollable_frame, text="Data File")
        GBMTheme.configure_label(file_label, 0, row)
        row += 1
        
        self.file_button = ttk.Button(
            self.scrollable_frame,
            text="Browse CSV...",
            command=self._load_file,
            style="Button.TButton"
        )
        self.file_button.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        row += 1
        
        self.file_name_label = ttk.Label(self.scrollable_frame, text="No file loaded")
        GBMTheme.configure_label(self.file_name_label, 0, row)
        row += 1
        
        # Date range section
        separator2 = ttk.Separator(self.scrollable_frame, orient='horizontal')
        separator2.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        date_label = ttk.Label(self.scrollable_frame, text="Training Period")
        GBMTheme.configure_label(date_label, 0, row)
        row += 1
        
        start_label = ttk.Label(self.scrollable_frame, text="Start Date:")
        GBMTheme.configure_label(start_label, 0, row)
        row += 1
        
        self.start_date_entry = ttk.Entry(self.scrollable_frame)
        self.start_date_entry.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        self.start_date_entry.config(font=GBMTheme.FONT)
        row += 1
        
        end_label = ttk.Label(self.scrollable_frame, text="End Date:")
        GBMTheme.configure_label(end_label, 0, row)
        row += 1
        
        self.end_date_entry = ttk.Entry(self.scrollable_frame)
        self.end_date_entry.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        self.end_date_entry.config(font=GBMTheme.FONT)
        row += 1
        
        # Simulation parameters section
        separator3 = ttk.Separator(self.scrollable_frame, orient='horizontal')
        separator3.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        params_label = ttk.Label(self.scrollable_frame, text="Simulation Parameters")
        GBMTheme.configure_label(params_label, 0, row)
        row += 1
        
        steps_label = ttk.Label(self.scrollable_frame, text="Prediction Steps:")
        GBMTheme.configure_label(steps_label, 0, row)
        row += 1
        
        self.steps_spinbox = ttk.Spinbox(self.scrollable_frame, from_=1, to=1000, increment=1)
        self.steps_spinbox.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        self.steps_spinbox.set(30)
        row += 1
        
        paths_label = ttk.Label(self.scrollable_frame, text="Number of Paths:")
        GBMTheme.configure_label(paths_label, 0, row)
        row += 1
        
        self.paths_spinbox = ttk.Spinbox(
            self.scrollable_frame,
            from_=1000,
            to=1000000000,
            increment=1000
        )
        self.paths_spinbox.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        self.paths_spinbox.set(100000000)
        row += 1
        
        display_label = ttk.Label(self.scrollable_frame, text="Display Paths:")
        GBMTheme.configure_label(display_label, 0, row)
        row += 1
        
        self.display_paths_spinbox = ttk.Spinbox(
            self.scrollable_frame,
            from_=1,
            to=500,
            increment=10
        )
        self.display_paths_spinbox.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        self.display_paths_spinbox.set(100)
        row += 1
        
        engine_label = ttk.Label(self.scrollable_frame, text="Engine:")
        GBMTheme.configure_label(engine_label, 0, row)
        row += 1
        
        # Get available engines from service
        available_engines = GBMService.get_available_engines()
        if not available_engines:
            available_engines = ["AUTO"]  # Fallback
        
        self.engine_combo = ttk.Combobox(
            self.scrollable_frame,
            values=available_engines,
            state="readonly"
        )
        self.engine_combo.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        self.engine_combo.set(available_engines[0])  # Set to first available
        row += 1
        
        # Run button
        separator4 = ttk.Separator(self.scrollable_frame, orient='horizontal')
        separator4.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        self.run_button = ttk.Button(
            self.scrollable_frame,
            text="Run Simulation",
            command=self._run_simulation,
            state="disabled",
            style="Button.TButton"
        )
        self.run_button.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
    
    def _layout_widgets(self):
        """Configure widget layout."""
        self.scrollable_frame.columnconfigure(0, weight=1)
        self.scrollable_frame.columnconfigure(1, weight=1)
    
    def _load_file(self):
        """Load CSV file."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        
        try:
            # Use service layer to load data
            self.data = GBMService.load_data(file_path)
            
            file_name = os.path.basename(file_path)
            self.file_name_label.config(text=f"Loaded: {file_name}")
            
            # Enable run button
            self.run_button.config(state="normal")
            
            # Callback to parent
            self.on_load_data(self.data, file_name)
            
        except Exception as e:
            self.file_name_label.config(text=f"Error: {str(e)}")
    
    def _run_simulation(self):
        """Gather configuration and run simulation."""
        if self.data is None:
            return
        
        try:
            config = SimConfig(
                start_date=self.start_date_entry.get(),
                end_date=self.end_date_entry.get(),
                steps=int(self.steps_spinbox.get()),
                paths=int(self.paths_spinbox.get()),
                engine=self.engine_combo.get(),
                display_paths=int(self.display_paths_spinbox.get())
            )
            
            # Callback to parent with config
            self.on_run_simulation(config)
            
        except Exception as e:
            print(f"Configuration error: {e}")
