#ICA For Subject One Self Paced
import pandas as pd
df_S1_SP = pd.read_csv('S1_SP.csv')

from sklearn.decomposition import FastICA

X1 = df_S1_SP
transformer = FastICA(n_components=5,random_state=0)
X1_transformed = transformer.fit_transform(X1)
X1_transformed.shape




#ICA For Subject One Speed 50
import pandas as pd
df_S1_50 = pd.read_csv('S1_50.csv')

from sklearn.decomposition import FastICA

X2 = df_S1_50
transformer = FastICA(n_components=5,random_state=0)
X2_transformed = transformer.fit_transform(X2)
X2_transformed.shape




#ICA For Subject One Speed 75
import pandas as pd
df_S1_75 = pd.read_csv('S1_75.csv')

from sklearn.decomposition import FastICA

X3 = df_S1_75
transformer = FastICA(n_components=5,random_state=0)
X3_transformed = transformer.fit_transform(X3)
X3_transformed.shape




#ICA For Subject One Speed 100
import pandas as pd
df_S1_100 = pd.read_csv('S1_100.csv')

from sklearn.decomposition import FastICA

X4 = df_S1_100
transformer = FastICA(n_components=5,random_state=0)
X4_transformed = transformer.fit_transform(X4)
X4_transformed.shape




#ICA For Subject One Speed 125
import pandas as pd
df_S1_125 = pd.read_csv('S1_125.csv')

from sklearn.decomposition import FastICA

X5 = df_S1_125
transformer = FastICA(n_components=5,random_state=0)
X5_transformed = transformer.fit_transform(X5)
X5_transformed.shape




#Print transformed data

print(X1_transformed)
print(X2_transformed)
print(X3_transformed)
print(X4_transformed)
print(X5_transformed)











from sklearn.model_selection import train_test_split

X = X1_transformed
y = #setup as speed variable

#split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)