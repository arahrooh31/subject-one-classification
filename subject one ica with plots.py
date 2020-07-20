
#ICA For Subject One Speed 50
import pandas as pd
df_S1_50 = pd.read_csv('S1_50.csv')

from sklearn.decomposition import FastICA

X1 = df_S1_50
transformer = FastICA(n_components=5,random_state=1)
X1_transformed = transformer.fit_transform(X1)
X1_transformed.shape

X1_transformed = pd.DataFrame(X1_transformed)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

fig = plt.figure()
X1_transformed.plot(legend = False)
plt.title('0.50 m/s')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')


#ICA For Subject One Speed 75
import pandas as pd
df_S1_75 = pd.read_csv('S1_75.csv')

from sklearn.decomposition import FastICA

X2 = df_S1_75
transformer = FastICA(n_components=5,random_state=1)
X2_transformed = transformer.fit_transform(X2)
X2_transformed.shape


X2_transformed = pd.DataFrame(X2_transformed)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

fig = plt.figure()
X2_transformed.plot(legend = False)
plt.title('0.75 m/s')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')


#ICA For Subject One Speed 100
import pandas as pd
df_S1_100 = pd.read_csv('S1_100.csv')

from sklearn.decomposition import FastICA

X3 = df_S1_100
transformer = FastICA(n_components=5,random_state=1)
X3_transformed = transformer.fit_transform(X3)
X3_transformed.shape


X3_transformed = pd.DataFrame(X3_transformed)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

fig = plt.figure()
X3_transformed.plot(legend = False)
plt.title('1.0 m/s')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')


#ICA For Subject One Speed 125
import pandas as pd
df_S1_125 = pd.read_csv('S1_125.csv')

from sklearn.decomposition import FastICA

X4 = df_S1_125
transformer = FastICA(n_components=5,random_state=1)
X4_transformed = transformer.fit_transform(X4)
X4_transformed.shape

X4_transformed = pd.DataFrame(X4_transformed)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

fig = plt.figure()
X4_transformed.plot(legend = False)
plt.title('1.25 m/s')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')

#ICA For Subject One Self Paced
import pandas as pd
df_S1_SP = pd.read_csv('S1_SP.csv')

from sklearn.decomposition import FastICA

X5 = df_S1_SP
transformer = FastICA(n_components=5,random_state=1)
X5_transformed = transformer.fit_transform(X5)
X5_transformed.shape

X5_transformed = pd.DataFrame(X5_transformed)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

fig = plt.figure()
X5_transformed.plot(legend = False)
plt.title('Self-Paced')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')
