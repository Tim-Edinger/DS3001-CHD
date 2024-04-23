import pandas as pd
import sklearn.preprocessing
from sklearn.cluster import KMeans # Import kmc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

data = pd.read_csv("fhs_train.csv")
data_cln = data.dropna()


# print(data.head())
# print(data.columns)


def maxmin(x):
    u = (x-min(x))/(max(x)-min(x))
    return u

x_train = data_cln.loc[:,['cigsPerDay', 'sysBP', 'glucose', 'age', 'sex', 'prevalentStroke','TenYearCHD']]
x_train = x_train.apply(maxmin)


kmeans = KMeans(n_clusters=8, max_iter=300, n_init = 10, random_state=0)
y = kmeans.fit_predict(x_train)
x_train['clusters'] = y
print(x_train)

pca_num_components = 2

reduced_data = PCA(n_components=pca_num_components).fit_transform(x_train)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=x_train['clusters'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(x_train.groupby('clusters').mean())