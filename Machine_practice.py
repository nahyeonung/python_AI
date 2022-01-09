import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd


import matplotlib as mpl
import numpy as np


# x = np.linspace(0, 20, 100)
# plt.plot(x, np.sin(x))
# plt.show()

df = pd.DataFrame(columns=['x', 'y'])
df.loc[0] = [3,1]
df.loc[1] = [4,1]
df.loc[2] = [3,2]
df.loc[3] = [4,2]
df.loc[4] = [10,5]
df.loc[5] = [10,6]
df.loc[6] = [11,5]
df.loc[7] = [11,6]
df.loc[8] = [15,1]
df.loc[9] = [15,2]
df.loc[10] = [16,1]
df.loc[11] = [16,2]

# print(df.head(20)) #데이터의 맨 처음부터 출력 Default 값 5 

sns.lmplot('x','y', data=df, fit_reg=False, scatter_kws={"s":200}) #seaborn에서 2D차트인 lm차트를 사용했고 데이터포인트 200으로 데이터 표시

plt.title('kmean plt')

plt.xlabel('x')

plt.ylabel('y')

data_points = df.values
kmeans = KMeans(n_clusters=3).fit(data_points)
