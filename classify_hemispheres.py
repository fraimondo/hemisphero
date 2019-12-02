import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

df = pd.read_csv('./BrainValues_HemisphDOC_MA.txt', sep=',',
                 header=None, 
                 names=['Code', 'LH_Occ', 'LC_Temp', 'RH_Occ', 'RH_Temp'])


_codes = ['UWS', 'MCS', 'MA']

df['DOC'] = [_codes[x] for x in df['Code'].values]


features = {
    'Right Hemisphere': ['RH_Occ', 'RH_Temp'],
    'Left Hemisphere': ['LH_Occ', 'LC_Temp']}

df_train = df[df['DOC'].isin(['MCS', 'UWS'])]
df_test = df[~df['DOC'].isin(['MCS', 'UWS'])]

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', C=1, probability=True,
                class_weight='balanced'))
])

fig, axes = plt.subplots(1, len(features), figsize=(14, 6))

for t_ax, (t_h, t_features) in zip(axes, features.items()):
    print(f'Using only {t_h}')
    X_train = df_train[t_features].values 
    y_train = df_train['Code'].values

    X_test = df_test[t_features].values
    y_test = df_test['Code'].values
    for t_feature in t_features:
        auc = roc_auc_score(y_train, df_train[t_feature].values)
        print(f'AUC for {t_feature} = {auc}')
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)

    print(f'Prob {t_h} MCS Pat 1: {probas[0, 1]}')

    t_ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='r', 
                 alpha=0.8, label='Patients in VS/UWS')
    t_ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='b', 
                 alpha=0.8, label='Patients in MCS')
    t_ax.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], c='g', 
                 alpha=0.8, label='Patient MA')
    if t_h == 'Left Hemisphere':
        t_ax.legend()
     
    xlim = t_ax.get_xlim()
    ylim = t_ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    a = t_ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, 
                     linestyles=['-'])
    t_ax.set_title(t_h)
    
fig.savefig('fig_clf.pdf')