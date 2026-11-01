# %% [markdown]
# # Predicting Airline Satisfaction — baseline LightGBM
# Playground S6E10 · métrica: ROC AUC · submissão: `id,satisfaction` (probabilidade).
# 
# No Kaggle: **Add Input → Competition → playground-series-s6e10**, depois *Run All* e *Submit*.

# %%
import glob, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

base = glob.glob('/kaggle/input/**/train.csv', recursive=True)
DATA = base[0].rsplit('/',1)[0] if base else '.'
train = pd.read_csv(f'{DATA}/train.csv', index_col='id')
test  = pd.read_csv(f'{DATA}/test.csv',  index_col='id')
TARGET = 'satisfaction'
y = train.pop(TARGET)
if y.dtype != 'int64' and y.dtype != 'float64':
    y = y.astype(str).str.lower().isin(['true','1','satisfied']).astype(int)
print(train.shape, test.shape, y.mean().round(3))

# %%
def fe(df):
    df = df.copy()
    svc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].between(0,5).all() and df[c].nunique() <= 6]
    df['svc_mean'] = df[svc].mean(axis=1)
    df['svc_min']  = df[svc].min(axis=1)
    df['svc_zero'] = (df[svc] == 0).sum(axis=1)
    if 'Arrival Delay in Minutes' in df:
        df['delay_total'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes'].fillna(0)
    for c in df.select_dtypes(exclude='number').columns:
        df[c] = df[c].astype('category')
    return df

X, Xt = fe(train), fe(test)
for c in X.select_dtypes('category').columns:
    Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)

# %%
params = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              n_estimators=5000, verbose=-1)
oof = np.zeros(len(X)); pred = np.zeros(len(Xt))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for k,(tr,va) in enumerate(skf.split(X,y)):
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], eval_metric='auc',
          callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va] = m.predict_proba(X.iloc[va])[:,1]
    pred += m.predict_proba(Xt)[:,1] / 5
    print(f'fold {k}: AUC {roc_auc_score(y.iloc[va], oof[va]):.5f} ({m.best_iteration_} it)')
print('CV AUC:', round(roc_auc_score(y, oof), 5))

# %%
pd.DataFrame({'id': Xt.index, 'satisfaction': pred}).to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()

# %% [markdown]
# ## Próximos passos
# - Juntar o dataset original (Airline Passenger Satisfaction, ~130k linhas) ao treino.
# - Testar CatBoost/XGBoost e fazer média das probabilidades.
# - Os melhores notebooks públicos estão em ~0,961 de AUC.
