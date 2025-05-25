import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

g3_dict_keys = pd.read_excel('/content/drive/MyDrive/ZIP_DATA/G3_Au_data/G3_46I05_DATA_CORRECTED.xlsx', sheet_name = None)
g2_dict_keys = pd.read_excel('/content/drive/MyDrive/ZIP_DATA/G2_data/G2_DATA_CORRECTED.xlsx', sheet_name = None)
g4_dict_keys = pd.read_excel('/content/drive/MyDrive/ZIP_DATA/G4_Channel_data/G4 Channel data for 46I05.xlsx', sheet_name = None)

g3_dict_keys.keys()

g3_1213 = g3_dict_keys['G3_12-13_aU_46I05']
g3_1213.head()

g3_1213.columns

from scipy.spatial import distance_matrix
import seaborn as sns

def spatial_corr_matrix(df, metals, decay):
    coords = df[['Lat', 'Long']]
    dists = distance_matrix(coords, coords)
    weights = np.exp(-decay * dists)  #decay : corresponds to the decrease in weights based on distance.

    corr_matrix = pd.DataFrame(index=metals, columns=metals)

    for i in metals:
        for j in metals:
            if i != j:
                weighted_mean_i = np.average(df[i], weights=weights.sum(axis=0))
                weighted_mean_j = np.average(df[j], weights=weights.sum(axis=0))
                weighted_cov = np.average((df[i]-weighted_mean_i)*(df[j]-weighted_mean_j), weights=weights.sum(axis=0))
                weighted_var_i = np.average((df[i]-weighted_mean_i)**2, weights=weights.sum(axis=0))
                weighted_var_j = np.average((df[j]-weighted_mean_j)**2, weights=weights.sum(axis=0))

                corr = weighted_cov / np.sqrt(weighted_var_i * weighted_var_j)
                corr_matrix.at[i, j] = corr
            else:
                corr_matrix.at[i, j] = 1
    return corr_matrix.astype(float)

metals = ['Pb', 'Ni', 'Cu', 'Zn', 'Au','Ag','Co']


corr_matrix = spatial_corr_matrix(g3_1213, metals, decay=5.0)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Spatially Weighted Correlation Heatmap of Metals')
plt.show()


Xs = g3_1213[['Angle', 'Azimuth', 'Depth From', 'Depth To', 'Sample Length','Cu','Co','Ni','Pb','Zn','Ag']]  # Replace 'Column1', 'Column2', 'Column3' with actual column names
ys = g3_1213['Au']

Xs.head()

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, random_state=42)

model = XGBRegressor(random_state = 42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

ys.value_counts()

feature_importances = model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': Xs.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('XGBoost Feature Importance')
plt.show()
#this backs up the spatial correlation matrix seen earlier.

