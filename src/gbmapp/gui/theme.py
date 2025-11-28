#!/usr/bin/env python
"""Theme and styling configuration for GBM GUI."""

from tkinter import ttk


class GBMTheme:
    """Theme configuration for the GBM application."""
    
    # Color scheme constants
    BG_COLOR = "#000000"
    FG_COLOR = "#05ff3f"
    FONT = ("Times New Roman", 12)
    
    @staticmethod
    def configure_styles(master):
        """Configure ttk styles for the application.
        
        Args:
            master: The root tkinter window
            
        Returns:
            ttk.Style: Configured style object
        """
        style = ttk.Style(master)
        style.configure(
            "Custom.TFrame", 
            background=GBMTheme.BG_COLOR, 
            foreground=GBMTheme.FG_COLOR
        )
        style.configure(
            "Button.TButton", 
            font=GBMTheme.FONT, 
            background=GBMTheme.BG_COLOR, 
            foreground=GBMTheme.FG_COLOR
        )
        return style
    
    @staticmethod
    def configure_label(label, column, row):
        """Configure a label with theme settings.
        
        Args:
            label: The ttk.Label to configure
            column: Grid column position
            row: Grid row position
        """
        label.grid(column=column, row=row, padx=5, pady=2, sticky="w")
        label.config(
            font=GBMTheme.FONT, 
            background=GBMTheme.BG_COLOR, 
            foreground=GBMTheme.FG_COLOR
        )
