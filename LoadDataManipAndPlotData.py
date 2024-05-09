import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def load_data(filePath):
    data = pd.read_csv(filePath)
    print(data.head)
    return data

def findMissing(data):
    nullValues = data.isnull().sum()
    return nullValues

def handleMissing(data, method='mean'):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data_filled = data.copy()
    
    for column in numeric_columns:
        if method == 'mean':
            data_filled[column] = data[column].fillna(data[column].mean())
        elif method == 'median':
            data_filled[column] = data[column].fillna(data[column].median())
    
    return data_filled
    
def plotData(data):
    sns.scatterplot(x='SGPA', y='CGPA', data=data)
    plt.title('Scatter Plot')
    plt.show()
    
    sns.histplot(data['SGPA'], bins=20, kde=True)
    plt.title('Histogram')
    plt.show()

    sns.boxplot(x='SGPA', y='CGPA', data=data)
    plt.title('Box Plot')
    plt.show()

    
dataset = load_data('student_records.csv')
missingValues = findMissing(dataset)
print('Missing Values: ')
print(missingValues)

filledDataset = handleMissing(dataset,method='mean')

plotData(filledDataset)