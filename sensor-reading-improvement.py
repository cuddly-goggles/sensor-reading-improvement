#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install pandas
# !pip install scikit-learn
# !pip install numpy
# !pip install tensorflow
# !pip install matplotlib
# !pip install seaborn


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np


# In[2]:


boxmod_data = pd.read_csv('./data/BoxMod.csv', header=None)
juul_data = pd.read_csv('./data/JUUL.csv', header=None)


# In[3]:


boxmod_data
# boxmod_data.tail()


# In[4]:


juul_data
# juul_data.tail()


# In[5]:


X_boxmod = boxmod_data.iloc[:, 1:]
y_boxmod = boxmod_data.iloc[:, 0] 


# In[6]:


# X_boxmod.head()


# In[7]:


X_juul = juul_data.iloc[:, 1:]
y_juul = juul_data.iloc[:, 0]


# In[8]:


X_boxmod_train, X_boxmod_test, y_boxmod_train, y_boxmod_test = train_test_split(X_boxmod, y_boxmod, test_size=0.3)
X_juul_train, X_juul_test, y_juul_train, y_juul_test = train_test_split(X_juul, y_juul, test_size=0.3)


# In[ ]:





# In[9]:


scaler = StandardScaler()
X_boxmod_train_scaled = scaler.fit_transform(X_boxmod_train)
X_boxmod_test_scaled = scaler.transform(X_boxmod_test)
X_juul_train_scaled = scaler.transform(X_juul_train)
X_juul_test_scaled = scaler.transform(X_juul_test)


# In[10]:


def evaluation(model, X_test_scaled, y_test, dataset_name):
    predictions = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"{dataset_name} - R2 Score:", r2)
    print(f"{dataset_name} - RMSE:", rmse)

def evaluation_cross(model, X_test_scaled, y_test, dataset_name):
    cv_scores = cross_val_score(model, X_test_scaled, y_test, cv=5, scoring='r2')

    # Calculate RMSE for each fold
    rmse_scores = np.sqrt(-cross_val_score(model, X_test_scaled, y_test, cv=5, scoring='neg_mean_squared_error'))
    
    print(f"Average R2 Score for {dataset_name}:", np.mean(cv_scores))
    print(f"Average RMSE for {dataset_name} data:", np.mean(rmse_scores))


# In[ ]:





# In[11]:


model = RandomForestRegressor()


# In[12]:


# Train the model on BoxMod training data
model.fit(X_boxmod_train_scaled, y_boxmod_train)


# In[13]:


evaluation(model, X_boxmod_test_scaled, y_boxmod_test, "BoxMod")


# In[14]:


evaluation_cross(model, X_boxmod_test_scaled, y_boxmod_test, "BoxMod")


# In[ ]:





# In[15]:


evaluation(model, X_juul_test_scaled, y_juul_test, "JUUL")


# In[16]:


evaluation_cross(model, X_juul_test_scaled, y_juul_test, "JUUL")


# In[17]:


# Fine-tuning on JUUL training data
model.fit(X_juul_train_scaled, y_juul_train)


# In[18]:


evaluation(model, X_juul_test_scaled, y_juul_test, "JUUL")


# In[19]:


evaluation_cross(model, X_juul_test_scaled, y_juul_test, "JUUL")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import KFold


# In[21]:


scaler = StandardScaler()
boxmod_scaled = scaler.fit_transform(boxmod_data)
juul_scaled = scaler.transform(juul_data)

X_boxmod_scaled = boxmod_scaled[:, 1:]
y_boxmod_scaled = boxmod_scaled[:, 0]
X_juul_scaled = juul_scaled[:, 1:]
y_juul_scaled = juul_scaled[:, 0]


# In[22]:


# scaler = StandardScaler()
# X_boxmod_scaled = scaler.fit_transform(X_boxmod)
# X_juul_scaled = scaler.transform(X_juul)


# In[23]:


def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + 0.01 * epoch)


# In[24]:


def create_model(X):
    model = Sequential()
    
    # model.add(Dense(32, input_dim=X.shape[1], activation='tanh'))
    # model.add(Dropout(0.2))
    # model.add(Dense(20, activation='tanh'))
    # model.add(Dense(1, activation='linear'))
    
    model.add(Dense(128, input_dim=X.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1, activation='linear'))
    return model


# In[25]:


def perform_kfold_cv(X, y):
    kfold = KFold(5, shuffle=True)
    mse_scores = []
    r2_scores = []
    
    for train, test in kfold.split(X, y):
        # Create the model
        model = create_model(X)

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
        # Fit the model
        model.fit(X[train], y[train], epochs=100, batch_size=32, verbose=0)
    
    
        # Evaluate the model
        y_pred = model.predict(X[test]).flatten()
        mse_scores.append(mean_squared_error(y[test], y_pred))
        r2_scores.append(r2_score(y[test], y_pred))

    
    # Calculate average and standard deviation of MSE
    average_mse = np.mean(mse_scores)
    std_dev_mse = np.std(mse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)

    
    print("Average MSE:", average_mse)
    print("Standard Deviation of MSE:", std_dev_mse)
    print("Average R2:", average_r2)
    print("Standard Deviation of R2:", std_dev_r2)


# In[26]:


perform_kfold_cv(X_boxmod_scaled, y_boxmod_scaled)


# In[27]:


perform_kfold_cv(X_juul_scaled, y_juul_scaled)


# In[ ]:





# In[ ]:





# In[28]:


def train_base_model(X, y):
    model = create_model(X)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    return model


# In[29]:


def perform_transfer_learning_kfold_cv(X_juul, y_juul, base_model):
    kfold = KFold(5, shuffle=True)
    mse_scores = []
    r2_scores = []

    for train, test in kfold.split(X_juul, y_juul):
        # Clone the base model structure and weights
        model = Sequential(base_model.layers[:-1])  # Exclude the last layer
        model.add(Dense(1, activation='linear'))  # Add new output layer
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Fine-tune the model on JUUL data
        model.fit(X_juul[train], y_juul[train], epochs=50, batch_size=32, verbose=0)  # Fewer epochs for fine-tuning

        # Evaluate the model
        y_pred = model.predict(X_juul[test]).flatten()
        mse_scores.append(mean_squared_error(y_juul[test], y_pred))
        r2_scores.append(r2_score(y_juul[test], y_pred))

    average_mse = np.mean(mse_scores)
    std_dev_mse = np.std(mse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)

    print("Average MSE:", average_mse)
    print("Standard Deviation of MSE:", std_dev_mse)
    print("Average R2:", average_r2)
    print("Standard Deviation of R2:", std_dev_r2)


# In[30]:


base_model = train_base_model(X_boxmod_scaled, y_boxmod_scaled)


# In[31]:


perform_transfer_learning_kfold_cv(X_juul_scaled, y_juul_scaled, base_model)


# In[ ]:





# In[ ]:





# In[ ]:





# In[32]:


from sklearn.cluster import KMeans

# Function to perform KMeans clustering and return cluster labels
def cluster_sensors(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data.T)
    return kmeans.labels_

# number of clusters
n_clusters = 5

# Perform clustering on the BoxMod and JUUL datasets
boxmod_clusters = cluster_sensors(boxmod_data.iloc[:, 1:], n_clusters)  # Exclude the ground truth column
juul_clusters = cluster_sensors(juul_data.iloc[:, 1:], n_clusters)       # Exclude the ground truth column

# Add the cluster information as a new feature to the datasets
boxmod_data_with_clusters = boxmod_data.copy()
juul_data_with_clusters = juul_data.copy()

for i, cluster_label in enumerate(boxmod_clusters):
    boxmod_data_with_clusters[f'Sensor_{i+1}_Cluster'] = cluster_label

for i, cluster_label in enumerate(juul_clusters):
    juul_data_with_clusters[f'Sensor_{i+1}_Cluster'] = cluster_label


# In[33]:


boxmod_data_with_clusters.head()


# In[34]:


juul_data_with_clusters.head()


# In[35]:


boxmod_data_with_clusters.columns = boxmod_data_with_clusters.columns.astype(str)
X_boxmod_clusters = boxmod_data_with_clusters.drop(columns=[boxmod_data_with_clusters.columns[0]])  # Drop the ground truth column
y_boxmod_clusters = boxmod_data_with_clusters.iloc[:, 0]  # Ground truth is the first column

# Splitting the BoxMod dataset with clusters into training and testing sets
X_train_boxmod_clusters, X_test_boxmod_clusters, y_train_boxmod_clusters, y_test_boxmod_clusters = train_test_split(
    X_boxmod_clusters, y_boxmod_clusters, test_size=0.2, random_state=42
)

# Initialize and train the Random Forest Regressor on the BoxMod dataset with cluster information
rf_regressor_clusters = RandomForestRegressor()
rf_regressor_clusters.fit(X_train_boxmod_clusters, y_train_boxmod_clusters)

evaluation_cross(rf_regressor_clusters, X_test_boxmod_clusters, y_test_boxmod_clusters, "BoxMod")


# In[ ]:





# In[36]:


from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

# calculate rolling averages with a window size of 3
rolling_avg_boxmod = boxmod_data.iloc[:, 1:].rolling(window=3, min_periods=1).mean()

# Combining the original features with the new statistical features
combined_features_boxmod = pd.concat([boxmod_data.iloc[:, 1:], rolling_avg_boxmod], axis=1)

# Normalizing the combined features
scaler = StandardScaler()
normalized_features_boxmod = scaler.fit_transform(combined_features_boxmod)

# Applying PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of the variance
reduced_features_boxmod = pca.fit_transform(normalized_features_boxmod)

# Splitting the reduced dataset into training and testing sets
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    reduced_features_boxmod, y_boxmod, test_size=0.2, random_state=42
)

# Training a Neural Network (MLPRegressor)
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, early_stopping=True)
mlp_regressor.fit(X_train_reduced, y_train_reduced)

evaluation_cross(mlp_regressor, X_test_reduced, y_test_reduced, "BoxMod")


# In[ ]:





# In[37]:


from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Random Forest and Gradient Boosting models
rf_regressor_reduced = RandomForestRegressor(random_state=42)
gb_regressor_reduced = GradientBoostingRegressor(random_state=42)

# Train the Random Forest model on the reduced BoxMod dataset
rf_regressor_reduced.fit(X_train_reduced, y_train_reduced)

evaluation_cross(rf_regressor_reduced, X_test_reduced, y_test_reduced, "BoxMod")


# Train the Gradient Boosting model on the reduced BoxMod dataset
gb_regressor_reduced.fit(X_train_reduced, y_train_reduced)

evaluation_cross(gb_regressor_reduced, X_test_reduced, y_test_reduced, "BoxMod")


# In[ ]:





# In[38]:


# Number of lags (past time steps) to consider for lag features
n_lags = 3

# Window size for rolling statistics
rolling_window_size = 3

# Creating lag features for the BoxMod dataset
lag_features_boxmod = pd.concat([boxmod_data.iloc[:, 1:].shift(i) for i in range(1, n_lags + 1)], axis=1)

# Creating rolling window features for the BoxMod dataset
rolling_features_boxmod = boxmod_data.iloc[:, 1:].rolling(window=rolling_window_size, min_periods=1).mean()

# Combining the original features with the lag and rolling features
combined_ts_features_boxmod = pd.concat([boxmod_data.iloc[:, 1:], lag_features_boxmod, rolling_features_boxmod], axis=1).dropna()

# New target values corresponding to the rows with lag features
new_y_boxmod = boxmod_data.iloc[n_lags:, 0]

# Splitting the dataset into training and testing sets
X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(
    combined_ts_features_boxmod, new_y_boxmod, test_size=0.2, random_state=42
)

# Training a Random Forest model on the new dataset with time series features
rf_regressor_ts = RandomForestRegressor(random_state=42)
rf_regressor_ts.fit(X_train_ts, y_train_ts)

# Calculating performance metrics
evaluation_cross(rf_regressor_ts, X_test_ts, y_test_ts, "BoxMod")


# In[39]:


lag_features_juul = pd.concat([juul_data.iloc[:, 1:].shift(i) for i in range(1, n_lags + 1)], axis=1)
rolling_features_juul = juul_data.iloc[:, 1:].rolling(window=rolling_window_size, min_periods=1).mean()
combined_ts_features_juul = pd.concat([juul_data.iloc[:, 1:], lag_features_juul, rolling_features_juul], axis=1).dropna()
new_y_juul = juul_data.iloc[n_lags:, 0]

# Calculating performance metrics on Juul dataset
evaluation_cross(rf_regressor_ts, combined_ts_features_juul, new_y_juul, "Juul")


# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[41]:


scaler = StandardScaler()
boxmod_scaled = scaler.fit_transform(boxmod_data)
juul_scaled = scaler.fit_transform(juul_data)


# In[42]:


def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 1:])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


# In[43]:


def perform_kfold_time_cv(X, y, look_back):
    kfold = KFold(5, shuffle=True)
    mse_scores = []
    r2_scores = []
    n_features = 31
    
    for train, test in kfold.split(X, y):
        # Create the model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(look_back, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
    
        # Fit the model
        model.fit(X[train], y[train], epochs=50, batch_size=32, verbose=0)
        
        # Evaluate the model
        y_pred = model.predict(X[test]).flatten()
        mse_scores.append(mean_squared_error(y[test], y_pred))
        r2_scores.append(r2_score(y[test], y_pred))

    
    # Calculate average and standard deviation of MSE
    average_mse = np.mean(mse_scores)
    std_dev_mse = np.std(mse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)

    
    print("Average MSE:", average_mse)
    print("Standard Deviation of MSE:", std_dev_mse)
    print("Average R2:", average_r2)
    print("Standard Deviation of R2:", std_dev_r2)
    return model


# In[44]:


look_back = 10
X, y = create_dataset(boxmod_scaled, look_back)
model_lstm_boxmod = perform_kfold_time_cv(X, y, look_back)


# In[45]:


X_lstm_juul, y_lstm_juul = create_dataset(juul_scaled, look_back)
y_pred_lstm_juul = model_lstm_boxmod.predict(X_lstm_juul).flatten()
mean_squared_error(y_lstm_juul, y_pred_lstm_juul), r2_score(y_lstm_juul, y_pred_lstm_juul)


# In[ ]:





# In[46]:


look_back = 10
X, y = create_dataset(juul_scaled, look_back)
perform_kfold_time_cv(X, y, look_back)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




