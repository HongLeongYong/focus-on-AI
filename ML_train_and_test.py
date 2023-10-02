import numpy
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

numpy.random.seed(2)

# 100位顧客 以及 他們的購物資料
# 常態分佈
x = numpy.random.normal(3, 1, 100) # 100個數字 以3為平均值 1為標準差
y = numpy.random.normal(150, 40, 100) / x # 100個數字 以150為平均值 40為標準差 除以x

# 80%的資料用來訓練 20%的資料用來測試
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# numpy.poly1d() 用來建立多項式
# numpy.polyfit() 用來建立多項式模型 - 最小平方多項式擬合
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4)) # 4次方的多項式

# r2_score() 用來評估模型的準確度
r2 = r2_score(train_y, mymodel(train_x)) # 模型準確度
print(r2)

myline = numpy.linspace(0, 6, 100) # 0~6 100個數字

# 顯示圖表
plt.scatter(train_x, train_y) # 顯示訓練資料
plt.plot(myline, mymodel(myline)) # 顯示模型
plt.show()

