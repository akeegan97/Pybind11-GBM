#!/usr/bin/env python
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkcalendar import Calendar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import(FigureCanvasTkAgg, NavigationToolbar2Tk)
from pylogic import utilities
import numpy as np
import simulation
import time
class GBMApp:
    def __init__(self, master):
        self.master = master
        self.style = ttk.Style(master)
        self.style_buttons = ttk.Style(master)
        master.config(bg="#000000")
        
        # Configure the custom style for the StartDateFrame
        self.style.configure("Custom.TFrame", background="#000000", foreground="#05ff3f")
        self.style_buttons.configure("Button.TButton", font=("Times New Roman", 12), background="#000000", foreground="#05ff3f")


        self.StartDateFrame = ttk.Frame(master)
        self.EndDateFrame = ttk.Frame(master)
        self.PredictionFrame = ttk.Frame(master)

        self.m_ButtonLabel = ttk.Label(master, text="Select CSV file:")
        self.m_FileBrowseButton = ttk.Button(master, text="Browse",command=self.browse_file)

        self.m_StartDateLabel = ttk.Label(self.StartDateFrame,text="Start Date of Training")
        self.m_StartDateVerifyLabel = ttk.Label(self.StartDateFrame,text="Is Date Valid: ")

        self.m_StartTrainingDate = Calendar(master, selectmode='day', background='#000000', foreground='#05ff3f', font=('Times New Roman', 12))
        self.m_StartTrainingDate.bind("<<CalendarSelected>>", self.UpdateStartDateValid)

        self.m_EndDateLabel = ttk.Label(self.EndDateFrame, text="End Date of Training")
        self.m_EndDateVerifyLabel = ttk.Label(self.EndDateFrame,text="Is Date Valid: ")

        self.m_EndTrainingDate = Calendar(master, selectmode='day', background='#000000', foreground='#05ff3f', font=('Times New Roman', 12))
        self.m_EndTrainingDate.bind("<<CalendarSelected>>", self.CombinedCallForBinding)

        self.m_PredictionDaysLabel = ttk.Label(self.PredictionFrame,text="Prediction Days")
        self.m_PredictionStepsValid = ttk.Label(self.PredictionFrame,text="Prediction Days Valid: ")
        
        self.m_PredictionDaysSpinBox = ttk.Spinbox(master, from_=0,to=0)
        self.m_PredictionDaysSpinBox.bind("<KeyRelease>", self.UpdateSteps)
        self.m_PredictionDaysSpinBox.bind("<<Increment>>", self.UpdateSteps)
        self.m_PredictionDaysSpinBox.bind("<<Decrement>>", self.UpdateSteps)
        self.m_PredictionDaysSpinBox.bind("<FocusOut>", self.UpdateSteps)
        self.m_StartCalculationButton = ttk.Button(master, text="Start Prediction",state="disabled",command= self.Calculate)

        self.m_CSVDataLabel = ttk.Label(master,text="Data Name: ")
        self.m_Data = None
        self.m_StartDate = ""
        self.m_EndDate = ""
        self.m_Steps = ""
        self.m_Paths = 100_000_000
        self.SetupLayout()

    def SetupLayout(self):
        self.master.title("GBM Predictions")
        self.StartDateFrame.config(style="Custom.TFrame")
        self.PredictionFrame.config(style="Custom.TFrame")
        self.m_ButtonLabel.grid(column=0,row=0)
        self.m_ButtonLabel.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")
        self.m_FileBrowseButton.grid(column=1,row=0)
        self.m_FileBrowseButton.config(style="Button.TButton")

        self.StartDateFrame.grid(column=0,row=1)
        self.m_StartDateLabel.grid(column=0,row=0)
        self.m_StartDateLabel.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")
        self.m_StartDateVerifyLabel.grid(column=0,row=1)
        self.m_StartDateVerifyLabel.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")

        self.EndDateFrame.grid(column=0,row=2)
        self.EndDateFrame.config(style="Custom.TFrame")
        self.m_EndDateLabel.grid(column=0,row=0)
        self.m_EndDateLabel.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")
        self.m_EndDateVerifyLabel.grid(column=0,row=1)
        self.m_EndDateVerifyLabel.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")

        self.PredictionFrame.grid(column=0,row=3)
        self.PredictionFrame.config(style="Custom.TFrame")
        self.m_PredictionDaysLabel.grid(column=0,row=0)
        self.m_PredictionDaysLabel.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")
        self.m_PredictionStepsValid.grid(column=0,row=1)
        self.m_PredictionStepsValid.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")

        self.m_StartTrainingDate.grid(column=1,row=1)
        self.m_EndTrainingDate.grid(column=1,row=2)
        self.m_PredictionDaysSpinBox.grid(column=1,row=3)
        self.m_PredictionDaysSpinBox.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")
        
        self.m_StartCalculationButton.grid(column=1,row=4)
        self.m_StartCalculationButton.config(style="Button.TButton")

        self.m_CSVDataLabel.grid(column=2,row=0)
        self.m_CSVDataLabel.config(font=("Times New Roman",12),background="#000000",foreground="#05ff3f")

    def UpdateStartDateValid(self, event = None):
        if self.m_Data is None:
            return
        self.m_StartDate = self.m_StartTrainingDate.get_date()
        print("FROM UPDATESTARTDATE: ",self.m_StartDate)
        if utilities.VerifyStartDate(self.m_Data, self.m_StartDate):
            self.m_StartDateVerifyLabel.config(text = "Date Is Valid")
        else:
            self.m_StartDateVerifyLabel.config(text = "Date Not Valid")
    
    def CombinedCallForBinding(self, event = None):
        self.UpdateEndDateValid(event)
        self.UpdatePredictionRange(event)
        self.ToggleStartCalculationButton()

    def UpdateEndDateValid(self, event = None):
        if self.m_Data is None:
            return
        self.m_EndDate = self.m_EndTrainingDate.get_date()
        if utilities.VerifyEndDate(self.m_Data,self.m_EndDate):
            self.m_EndDateVerifyLabel.config(text = "Date Is Valid")
            print("FROM UPDATESTARTDATE: ",self.m_EndDate)
        else:
            self.m_EndDateVerifyLabel.config(text = "Date Not Valid")
    
    def UpdatePredictionRange(self, event=None):
        if self.m_Data is None:
            return
        topOfRange = utilities.GetEndRange(self.m_Data, self.m_EndDate)
        if topOfRange is not None:
            self.m_PredictionDaysSpinBox.config(to = topOfRange)

    def ToggleStartCalculationButton(self):
        if self.m_Data is None or not utilities.CheckStartEndDate(self.m_StartDate, self.m_EndDate):
            self.m_StartCalculationButton.config(state="disabled")
        else:
            self.m_StartCalculationButton.config(state="normal")
            self.m_Steps = self.m_PredictionDaysSpinBox.get()

    def UpdateSteps(self, event=None):
        self.m_Steps = self.m_PredictionDaysSpinBox.get()
        print(self.m_Steps)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path = file_path
            self.m_Data = utilities.ReadCsvData(file_path)
            if self.m_Data is not None:
                self.plot()  
                fileName = os.path.basename(file_path)
                self.m_CSVDataLabel.config(text=f"Data Name:{fileName}")

    def plot(self):
        fig = Figure(figsize=(5,5), dpi=100)
        plot1 = fig.add_subplot(111)
        plot1.plot(self.m_Data['Date'], self.m_Data['Close'], color="green", linestyle='-')
        plot1.set_facecolor("black")
        plot1.set_xlabel('Date')
        plot1.set_ylabel('Close')
        plot1.set_title('Actual Prices')
        plot1.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().grid(column= 2,row=1,rowspan=3,sticky="nsew")

    def plotGBM(self, walks, endDate):
        fig, plot1 = plt.subplots(figsize=(8, 6))
        startIndex = self.m_Data.index[self.m_Data['Date'] == endDate][0]
        realPrices = self.m_Data.iloc[startIndex:startIndex + int(self.m_Steps)]
        plot1.plot(realPrices['Date'], realPrices['Close'], label='Real Prices', color='blue')
        plot1.set_facecolor("black")
        subsetOfWalks = walks[:min(500, len(walks))]
        for i, walk in enumerate(subsetOfWalks):
            plot1.plot(realPrices['Date'], walk, label=f'Walk {i+1}')
        plot1.plot(realPrices['Date'],realPrices['Close'],zorder=len(walks)+1,color='white')
        plot1.set_xlabel('Date')
        plot1.set_ylabel('Price')
        plot1.set_title('GBM Predictions')
        plot1.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().grid(column=3, row=1, rowspan=3, sticky="nsew")

    def Calculate(self):
        stats = utilities.CalculateStatistics(self.m_Data, self.m_StartDate, self.m_EndDate, self.m_Steps)
        print(stats.normalizedDeviation,stats.normalizedMu,stats.normalizedVariance)

        startIndex = self.m_Data.index[self.m_Data['Date'] == self.m_EndDate][0]
        endIndex = startIndex + int(self.m_Steps)
        comparedData = self.m_Data.iloc[startIndex:endIndex]
        realPrice = comparedData['Close'].iloc[-1]
        startingPrice = self.m_Data.loc[startIndex - 1, 'Close']
#        start_time = time.perf_counter()
#        paths, simulatedPrice = utilities.GBM(startingPrice,stats.normalizedMu,stats.normalizedVariance,stats.normalizedDeviation,int(self.m_Steps),self.m_Paths)
#        end_time = time.perf_counter()
#        print(f"Python version took {end_time - start_time:.4f} seconds.")
#        self.plotGBM(paths, self.m_EndDate)
#        print(simulatedPrice)
#        print(realPrice)
#        print(simulatedPrice/realPrice)
#
#        start_time = time.perf_counter()
#        walks, averagePrice = simulation.SimulatedGBM(startingPrice,stats.normalizedMu,stats.normalizedVariance,stats.normalizedDeviation,int(self.m_Steps),self.m_Paths)
#        end_time = time.perf_counter()
#        print(f"C++ version took {end_time - start_time:.4f} seconds.")
#        npPaths = np.array(walks)
#        self.plotGBM(npPaths, self.m_EndDate)

        start_time = time.perf_counter()
        walks, averagePrice = simulation.SimulateGBMMultiThreaded(startingPrice,stats.normalizedMu,stats.normalizedVariance,stats.normalizedDeviation,int(self.m_Steps),self.m_Paths)
        end_time = time.perf_counter()
        print(f"C++ MultiThreaded version took {end_time - start_time:.4f} seconds.")


        start_time = time.perf_counter()
        walks, averagePrice = simulation.SimulateGBMIntrinsicMT(startingPrice,stats.normalizedMu,stats.normalizedVariance,stats.normalizedDeviation,int(self.m_Steps),self.m_Paths)
        end_time = time.perf_counter()
        print(f"C++ MultiThreaded Intrinsic version took {end_time - start_time:.4f} seconds.")
        npPaths = np.array(walks)
        self.plotGBM(npPaths, self.m_EndDate)

        print(averagePrice)
        print(realPrice)
        print(averagePrice/realPrice)



def main():
    root = tk.Tk()
    app = GBMApp(root)
    def onClose():
        plt.close('all')
        root.quit()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW",onClose)
    root.mainloop()

if __name__ =="__main__":
    main()