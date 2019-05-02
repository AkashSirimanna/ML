import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plot


df = pd.read_csv("kc_house_data.csv")
df = df[['sqft_living','price']]
data = np.array(df)

def summations(data): #variables a and b in the form y = mx + b
	sumPrice = 0
	sumArea = 0
	sumAreaSq = 0
	sumAreaPrice = 0
	rSq = 0

	for area,price in data:
		sumPrice += price
		sumArea = sumArea
		sumAreaSq += pow(area,2)
		sumAreaPrice += area*price
		#R^2 = sum of vertical distance from point to line squared/all points

	b = (sumPrice*sumAreaSq - sumArea*sumAreaPrice)/(len(data)*sumAreaSq - pow(sumArea,2))
	m = (len(data)*sumAreaPrice - sumArea*sumPrice)/(len(data)*sumAreaSq - pow(sumArea,2))
	return m,b

def rSquared(m,b,data):
	numPoints = len(data)
	sumRSq = 0
	for x,y in data:
		regressed = m*x + b
		sumRSq += abs(regressed-y)
	return sumRSq/numPoints


def regression(data,slope,intercept):
	minR = float(rSquared(slope,intercept,data))
	idealB = intercept
	b = intercept
	for counter in range(int(b)):
		counter+= 1
		newM,newB = summations(data)
		tempR = float(rSquared(newM,newB-counter,data))
		if (minR > tempR):
			minR = tempR
			idealB = newB-counter
		if (tempR > minR):
			return idealB


slope = summations(data)[0]
intercept = summations(data)[0]
minPrice = int(df['price'].min())
maxPrice = int(df['price'].max())
print(minPrice,maxPrice)
b = regression(data,slope,intercept)
x = df[['sqft_living']]
y = slope*x + intercept
plot.ylabel('Price')
plot.xlabel('Sqft Living')
plot.scatter(df[['sqft_living']],df[['price']],s=1)
plot.plot(x,y,'r','-')
plot.show()


