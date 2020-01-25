import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

data = pd.read_csv('data_for_outlier.csv')

value1=data['fixed.acidity']
value2=data['volatile.acidity']
value3=data['citric.acid']
value4=data['residual.sugar']
value5=data['chlorides']
value6=data['free.sulfur.dioxide']
value7=data['total.sulfur.dioxide']
value8=data['density']
value9=data['pH']
value10=data['sulphates']
value11=data['alcohol']

print(data['fixed.acidity'].describe())

box=[value1,value2,value3,value4,value5,value6,value7,value8,value9,value10,value11]
plt.boxplot(box)
plt.show()