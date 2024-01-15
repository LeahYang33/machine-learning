'''
diabetes dataset 是一個糖尿病資料集,主要包括 442 筆資料,10 個屬性值,分別是:age (年齡)、 sex (性別)、
bmi (Body Mass Index 體質指數)、 bp (Average Blood Pressure平均血壓)、s1~s6 (一年後疾病級數指標)。
target 為一年後患疾病的定量指標。
1.建立線性多元迴歸的預測模型,繪出散佈圖來比較預測一年後患疾病的定量指標和實際一年後患疾病的定量指標的結果。

2.建立線性多元迴歸的預測模型,只取 age (年齡)、 sex (性別)、 bmi (Body Mass Index 體質指數)、
bp (Average Blood Pressure 平均血壓) 做為解釋變數,產生模型,並繪出散佈圖來比較預測一年後患疾病的定量指標和
實際一年後患疾病的定量指標的結果。
'''
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

diabetes=datasets.load_diabetes()
#自變數
X=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
#[:,0:4] 表示選擇所有行 (:) 和前四列（0到3）。
X2 = X.iloc[:,0:4]
#應變數
y=pd.DataFrame(diabetes.target,columns=['Quantitative measure'])


lm=LinearRegression()
lm.fit(X,y)

lm2=LinearRegression()
lm2.fit(X2,y)
#線性回歸模型對 X 預測的結果
predicted_quantitative_measure=lm.predict(X)
plt.scatter(y, predicted_quantitative_measure)
plt.xlabel('Quantitative measure')
plt.ylabel('Predicted Quantitative Measure')
plt.title('Quantitative Measure vs Predicted Quantitative Measure')
plt.show()

predicted_quantitative_measure2 = lm2.predict(X2)
plt.scatter(y, predicted_quantitative_measure2)
plt.xlabel('Quantitative Measure')
plt.ylabel('Predicted Quantitative Measure')
plt.title('Quantitative Measure vs Predicted Quantitative Measure')
plt.show()