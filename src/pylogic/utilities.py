import pandas as pd
import numpy as np

def ReadCsvData(filePath):
    if VerifyCsvFormat(filePath):     
        data = pd.read_csv(filePath)
        data = pd.DataFrame({
            "Date": data["Date"],
            "Close": data["Close"]
        })
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values(by='Date',ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
    else:
        return None

def VerifyCsvFormat(filePath):
    try:
        data = pd.read_csv(filePath,nrows=10)
    except Exception as e:
        return False
    if 'Date' not in data.columns or 'Close' not in data.columns:
        return False
    if data['Date'].dtype != 'object' or data['Close'].dtype != 'float64':
        return False
    return True


def VerifyStartDate(data, startDate):
    startDate = pd.to_datetime(startDate)
    if(startDate in data['Date'].values):
        return True
    else:
        return False
    
def VerifyEndDate(data, endDate):
    endDate = pd.to_datetime(endDate)
    if(endDate in data['Date'].values):
        return True
    else:
        return False
    
def GetEndRange(data, endDate):
    endDate = pd.to_datetime(endDate)
    try:
        index_of_end_date = data.index[data['Date'] == endDate][0]
        top_of_range = len(data) - index_of_end_date - 1
        return top_of_range
    except IndexError:
        return None

def CheckStartEndDate(startDate, endDate):
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)
    if startDate < endDate:
        return True
    else:
        return False
    
class Statistics:
    def __init__(self, mu, deviation, variance, normalizedMu, normalizedVariance, normalizedDeviation):
        self.trainingMu = mu
        self.trainingDeviation = deviation
        self.trainingVariance = variance
        self.normalizedMu = normalizedMu
        self.normalizedDeviation = normalizedDeviation
        self.normalizedVariance = normalizedVariance
    
def CalculateStatistics(data, startDate, endDate,steps):
    startIndex = data.index[data['Date'] == startDate][0]
    endIndex = data.index[data['Date']==endDate][0]
    trainingData = data.iloc[startIndex:endIndex]
    logReturns = np.log(trainingData['Close']) - np.log(trainingData['Close'].shift(1))
    trainingMu = logReturns.mean()
    trainingDeviation = logReturns.std()
    trainingVariance = trainingDeviation**2
    normalizedMu = trainingMu * int(steps)
    normalizedVariance = trainingVariance * np.sqrt(int(steps))
    normalizedDeviation = np.sqrt(trainingDeviation)
    trainingStats = Statistics(trainingMu, trainingDeviation,trainingVariance, normalizedMu,normalizedVariance, normalizedDeviation)

    return trainingStats

def GBM(startingPrice, normalizedMu, normalizedVar, normalizedDev, steps, paths):
    deltaT = 1 / steps
    simulatedPaths = np.full((paths, steps), startingPrice)
    displayPaths = []
    
    for i in range(simulatedPaths.shape[0]):
        path = [startingPrice]
        
        for j in range(1, simulatedPaths.shape[1]):
            previousPrice = path[-1]
            randomShock = np.random.normal(0, np.sqrt(deltaT))
            predictedPrice = previousPrice * np.exp((normalizedMu - normalizedVar / 2) * deltaT + normalizedDev * randomShock)
            path.append(predictedPrice)
        if i < 50:  
            displayPaths.append(path)
        simulatedPaths[i] = path
    averagePredictedPrice = np.mean(simulatedPaths[:, -1]) 
    
    return displayPaths, averagePredictedPrice


