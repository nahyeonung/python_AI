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

data_points = df.values #values라는 함수를 사용함으로써 numpy형태의 array로 변경
kmeans = KMeans(n_clusters=3).fit(data_points) #3개의 클러스터(집합) 으로 나눠줌.
kmeans.labels_
kmeans.cluster_centers_ #각 클러스터의 센터들의 값을 보여줌.
df['cluster_id'] = kmeans.labels_
print(df.head(20))

sns.lmplot('x','y', data=df, fit_reg=False, # x-axis, y-axis, data, no line
                    scatter_kws={"s": 150}, # marker size
                    hue="cluster_id")
#title
plt.title('after kmean clustering')
plt.show()