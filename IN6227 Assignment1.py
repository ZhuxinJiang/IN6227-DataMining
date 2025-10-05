#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd

train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

train_df = pd.read_csv(train_url, header=None, names=columns, na_values=' ?')
test_df = pd.read_csv(test_url, header=0, names=columns, na_values=' ?')

missing_train = train_df.isna().sum().to_frame('MissingCount')
missing_train['MissingRate'] = missing_train['MissingCount'] / len(train_df) * 100

missing_test = test_df.isna().sum().to_frame('MissingCount')
missing_test['MissingRate'] = missing_test['MissingCount'] / len(test_df) * 100

print("Training set missing values:")
print(missing_train)

print("\nTest set missing values:")
print(missing_test)

print("\nSample of training data:")
print(train_df.head())

print("\nSample of test data:")
print(test_df.head())


# In[77]:


print("Before dropping missing values:")
print("Training set samples:", len(train_df))
print("Test set samples:", len(test_df))

train_df = train_df.dropna()
test_df = test_df.dropna()

print("\nAfter dropping missing values:")
print("Training set samples:", len(train_df))
print("Test set samples:", len(test_df))


# In[78]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

#删除fnlwgt
train_df = train_df.drop(columns=['fnlwgt'], errors='ignore')
test_df = test_df.drop(columns=['fnlwgt'], errors='ignore')

#分类
nominal_features = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
ordinal_features = ['education']
#interval_features = ['education_num']
# ratio_features = ['capital_gain', 'capital_loss', 'hours_per_week']  # 不归一化


# In[79]:


import pandas as pd

print(train_df['income'].value_counts())
print(test_df['income'].value_counts())


# In[80]:


#标称
le_dict = {}
for col in nominal_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    le_dict[col] = le

#education
education_order = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
    'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors',
    'Masters', 'Prof-school', 'Doctorate'
]
edu_map = {edu: i for i, edu in enumerate(education_order)}

train_df['education'] = train_df['education'].astype(str).str.strip()
test_df['education'] = test_df['education'].astype(str).str.strip()

train_df['education'] = train_df['education'].map(edu_map)
test_df['education'] = test_df['education'].map(edu_map)


#education_num
# scaler = StandardScaler()
# train_df['education_num'] = scaler.fit_transform(train_df[['education_num']])
# test_df['education_num'] = scaler.transform(test_df[['education_num']])


#income
train_df['income'] = train_df['income'].astype(str).str.strip().str.replace('.', '', regex=False)
test_df['income'] = test_df['income'].astype(str).str.strip().str.replace('.', '', regex=False)

income_map = {'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1}
train_df['income'] = train_df['income'].map(income_map)
test_df['income'] = test_df['income'].map(income_map)

print("\nSample of preprocessed training data:")
print(train_df.head())

print("\nSample of preprocessed test data:")
print(test_df.head())


# In[81]:


trd=train_df
ted=test_df


# In[82]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = train_df.drop(columns=['income'])
y_train = train_df['income']
X_test = test_df.drop(columns=['income'])
y_test = test_df['income']


# In[83]:


def evaluate(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }


# In[150]:


import pandas as pd
import numpy as np
from chefboost import Chefboost as chef
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.tree import DecisionTreeClassifier

# 假定 train_df, test_df 已预处理好
inv_map = {0: '<=50K', 1: '>50K'}
train_for_chef = train_df.copy()
test_for_chef  = test_df.copy()
if set(train_df['income'].unique()).issubset({0,1}):
    train_for_chef['income'] = train_for_chef['income'].map(inv_map)
    test_for_chef['income']  = test_for_chef['income'].map(inv_map)

train_for_chef = train_for_chef.rename(columns={'income':'Decision'})
test_for_chef  = test_for_chef.rename(columns={'income':'Decision'})

feature_cols = [c for c in train_for_chef.columns if c != 'Decision']
is_numeric_col = {c: train_for_chef[c].dtype.kind in 'iufc' for c in feature_cols}

def row_to_obj(row):
    obj = []
    for c in feature_cols:
        v = row[c]
        if pd.isna(v):
            obj.append(0 if is_numeric_col[c] else "")
        else:
            obj.append(float(v) if is_numeric_col[c] else str(v))
    return obj

def evaluate_string_preds(y_true_strings, y_pred_strings, positive_label='>50K'):
    y_true = [1 if y==positive_label else 0 for y in y_true_strings]
    y_pred = [1 if y==positive_label else 0 for y in y_pred_strings]
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)    
    }

#ID3
start_time = time.time()
config_id3 = {'algorithm':'ID3'}
model_id3 = chef.fit(train_for_chef, config_id3)
train_time_id3 = time.time() - start_time

start_time = time.time()
preds_id3 = [chef.predict(model_id3, row_to_obj(row)) for _, row in test_for_chef.iterrows()]
predict_time_id3 = time.time() - start_time

perf_id3 = evaluate_string_preds(test_for_chef['Decision'].tolist(), preds_id3)
print("ID3 performance:", perf_id3)
print(f"Training time: {train_time_id3:.2f}s, Prediction time: {predict_time_id3:.2f}s\n")

#C4.5
start_time = time.time()
config_c45 = {'algorithm':'C4.5'}
model_c45 = chef.fit(train_for_chef, config_c45, target_label='Decision', validation_df=test_for_chef)
train_time_c45 = time.time() - start_time

start_time = time.time()
preds_c45 = [chef.predict(model_c45, row_to_obj(row)) for _, row in test_for_chef.iterrows()]
predict_time_c45 = time.time() - start_time

perf_c45 = evaluate_string_preds(test_for_chef['Decision'].tolist(), preds_c45)
print("C4.5 performance:", perf_c45)
print(f"Training time: {train_time_c45:.2f}s, Prediction time: {predict_time_c45:.2f}s\n")


# In[148]:


# 训练 CART(gini)
X_train = train_df.drop(columns=['income'])
y_train = train_df['income']
X_test = test_df.drop(columns=['income'])
y_test = test_df['income']

start_time = time.time()
dt_gini = DecisionTreeClassifier(criterion='gini',random_state=42)
dt_gini.fit(X_train, y_train)
train_time_gini = time.time() - start_time

start_time = time.time()
y_pred_gini = dt_gini.predict(X_test)
predict_time_gini = time.time() - start_time

perf_gini = {
    "accuracy": accuracy_score(y_test, y_pred_gini),
    "precision_weighted": precision_score(y_test, y_pred_gini, average='weighted'),
    "recall_weighted": recall_score(y_test, y_pred_gini, average='weighted'),
    "f1_weighted": f1_score(y_test, y_pred_gini, average='weighted')
}
print("CART (gini) performance:", perf_gini)
print(f"Training time: {train_time_gini:.2f}s, Prediction time: {predict_time_gini:.2f}s\n")


# In[147]:


# 训练 CART(entropy)
start_time = time.time()
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
train_time_entropy = time.time() - start_time

start_time = time.time()
y_pred_entropy = dt_entropy.predict(X_test)
predict_time_entropy = time.time() - start_time

perf_entropy = { 
    "accuracy": accuracy_score(y_test, y_pred_entropy),
    "precision_weighted": precision_score(y_test, y_pred_entropy, average='weighted'),
    "recall_weighted": recall_score(y_test, y_pred_entropy, average='weighted'),
    "f1_weighted": f1_score(y_test, y_pred_entropy, average='weighted')
}
print("ID3-like (entropy) performance:", perf_entropy)
print(f"Training time: {train_time_entropy:.2f}s, Prediction time: {predict_time_entropy:.2f}s\n")


# In[164]:


#entropy grid
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'max_depth': [5,6,7, 8,9, 10, 11,12],
    'min_samples_split': [2, 10, 20, 30],
    'min_samples_leaf': [1, 5, 10, 20,30,40,50,60,70,80,90,100],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

dt_entropy = DecisionTreeClassifier(criterion='entropy',class_weight='balanced', random_state=42)

grid_search = GridSearchCV(
    dt_entropy,
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
print("Best CV F1-score:", grid_search.best_score_)
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)

perf = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='weighted'),
    "recall": recall_score(y_test, y_pred, average='weighted'),
    "f1_score": f1_score(y_test, y_pred, average='weighted')
}

print("Performance on test set with best parameters:", perf)

# 如果你想要详细的分类报告（含每个类别的指标）
print("\nDetailed classification report:")
print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))


# In[180]:


#entropy random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time
import numpy as np

X_train = train_df.drop(columns=['income'])
y_train = train_df['income']
X_test  = test_df.drop(columns=['income'])
y_test  = test_df['income']

#RandomizedSearch
from scipy.stats import randint

param_dist = {
    'max_depth': [None] + list(range(4, 21)),
    'min_samples_split': randint(2, 50),
    'min_samples_leaf': randint(1, 20),
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

rnd = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(criterion='entropy',class_weight='balanced', random_state=42),
    param_distributions=param_dist,
    n_iter=60,     
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

t0 = time.time()
rnd.fit(X_train, y_train)
t1 = time.time()
print(f"RandomizedSearch done in {(t1-t0):.1f} seconds")
print("Best params (RandomizedSearch):", rnd.best_params_)
print("Best CV score (F1):", rnd.best_score_)

best_dt_rnd = rnd.best_estimator_
y_pred_rnd = best_dt_rnd.predict(X_test)
print("Test set performance (RandomizedSearch best):")
print({
    'accuracy': accuracy_score(y_test, y_pred_rnd),
    'precision': precision_score(y_test, y_pred_rnd),
    'recall': recall_score(y_test, y_pred_rnd),
    'f1_score': f1_score(y_test, y_pred_rnd)
})
print("\nClassification report:\n", classification_report(y_test, y_pred_rnd))


# In[177]:


import time
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

t0 = time.time()
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
train_time = time.time() - t0

t1 = time.time()
y_pred_nb = nb_model.predict(X_test_scaled)
predict_time = time.time() - t1

nb_perf = {
    'accuracy': accuracy_score(y_test, y_pred_nb),
    'precision': precision_score(y_test, y_pred_nb, average='weighted'),
    'recall': recall_score(y_test, y_pred_nb, average='weighted'),
    'f1_score': f1_score(y_test, y_pred_nb, average='weighted')
}

print("Naive Bayes Performance:", nb_perf)
print(f"Training time: {train_time:.2f}s, Prediction time: {predict_time:.2f}s")

print("\nDetailed classification report:\n")
print(classification_report(y_test, y_pred_nb, target_names=['<=50K', '>50K']))


# In[174]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb = GaussianNB()

# 网格搜索
param_grid = {
    'var_smoothing': np.logspace(-12, -6, 7) 
}

grid_search = GridSearchCV(
    estimator=gnb,
    param_grid=param_grid,
    scoring=make_scorer(f1_score, average='weighted'), 
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)
print("Best params (GridSearch):", grid_search.best_params_)
print("Best CV F1-score:", grid_search.best_score_)

best_gnb = grid_search.best_estimator_

# 测试集评估
y_pred_gnb = best_gnb.predict(X_test_scaled)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
nb_perf = {
    'accuracy': accuracy_score(y_test, y_pred_gnb),
    'precision': precision_score(y_test, y_pred_gnb, average='weighted'),
    'recall': recall_score(y_test, y_pred_gnb, average='weighted'),
    'f1_score': f1_score(y_test, y_pred_gnb, average='weighted')
}
print("Test set performance with best parameters:", nb_perf)
print("\nDetailed classification report:\n", classification_report(y_test, y_pred_gnb, target_names=['<=50K', '>50K']))

# 随机搜索
from scipy.stats import uniform

param_dist = {'var_smoothing': uniform(loc=1e-12, scale=1e-6)} 

rnd_search = RandomizedSearchCV(
    estimator=gnb,
    param_distributions=param_dist,
    n_iter=30, 
    scoring=make_scorer(f1_score, average='weighted'),
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

rnd_search.fit(X_train_scaled, y_train)
print("Best params (RandomizedSearch):", rnd_search.best_params_)
print("Best CV F1-score:", rnd_search.best_score_)


# In[181]:


import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
)


params = {
    'class_weight': None,
    'max_depth': 15,
    'max_features': None,
    'min_samples_leaf': 3, 
    'min_samples_split': 40
}

def ensure_binary_series(y):
    y_s = pd.Series(y)
    if y_s.dtype.kind in 'iufc':  
        return y_s.astype(int)
    mapping = {}
    for v in y_s.unique():
        if isinstance(v, str) and '>' in v:
            mapping[v] = 1
        elif isinstance(v, str) and v.strip().startswith('>'):
            mapping[v] = 1
        else:
            mapping[v] = 0
    return y_s.map(mapping).astype(int)

y_train_bin = ensure_binary_series(y_train)
y_test_bin  = ensure_binary_series(y_test)

dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42, **params)

t0 = time.time()
dt_entropy.fit(X_train, y_train_bin)
train_time = time.time() - t0


t1 = time.time()
y_pred = dt_entropy.predict(X_test)
y_proba = dt_entropy.predict_proba(X_test)[:, 1]
predict_time = time.time() - t1


perf = {
    "accuracy": accuracy_score(y_test_bin, y_pred),
    "precision_weighted": precision_score(y_test_bin, y_pred, average='weighted', zero_division=0),
    "recall_weighted": recall_score(y_test_bin, y_pred, average='weighted', zero_division=0),
    "f1_weighted": f1_score(y_test_bin, y_pred, average='weighted', zero_division=0)
}

cm = confusion_matrix(y_test_bin, y_pred)
cls_report = classification_report(y_test_bin, y_pred, target_names=["<=50K", ">50K"], zero_division=0)

fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
roc_auc = auc(fpr, tpr)
roc_auc_score_val = roc_auc_score(y_test_bin, y_proba)

print("使用参数:", params)
print("\nCART (entropy) performance (weighted):", perf)
print(f"Training time: {train_time:.2f}s, Prediction time: {predict_time:.2f}s")
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", cls_report)
print(f"ROC AUC: {roc_auc:.4f}  (roc_auc_score: {roc_auc_score_val:.4f})")

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], linestyle='--', lw=1.5, label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — CART (entropy)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


# In[183]:


import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.preprocessing import label_binarize

best_gnb_grid = grid_search.best_estimator_
best_gnb_rnd = rnd_search.best_estimator_

def to_binary(y):
    y_s = np.array(y)
    if y_s.dtype.kind in 'iufc':
        return y_s.astype(int)
    return np.array([1 if (isinstance(v, str) and ('>' in v or v.strip().startswith('>'))) else 0 for v in y_s])

y_test_bin = to_binary(y_test)

def eval_and_plot_roc(model, X_test, y_test_bin, title_suffix):

    t0 = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]  
    pred_time = time.time() - t0

    y_pred = model.predict(X_test)

    perf = {
        'accuracy': accuracy_score(y_test_bin, y_pred),
        'precision_weighted': precision_score(y_test_bin, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test_bin, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test_bin, y_pred, average='weighted', zero_division=0)
    }

    # ROC / AUC
    fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_auc_sklearn = roc_auc_score(y_test_bin, y_proba)

    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], linestyle='--', lw=1.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {title_suffix}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    return perf, roc_auc, pred_time

print("GridSearch 最佳参数:", grid_search.best_params_)
t_start = time.time()
grid_perf, grid_auc, grid_pred_time = eval_and_plot_roc(best_gnb_grid, X_test_scaled, y_test_bin, "GaussianNB (GridSearch)")
print("GridSearch Best model performance:", grid_perf)
print(f"GridSearch AUC: {grid_auc:.4f}, Prediction time: {grid_pred_time:.4f}s\n")
print("详细分类报告（GridSearch 最佳模型）：")
print(classification_report(y_test_bin, best_gnb_grid.predict(X_test_scaled), target_names=['<=50K','>50K'], zero_division=0))

print("RandomizedSearch 最佳参数:", rnd_search.best_params_)
rnd_perf, rnd_auc, rnd_pred_time = eval_and_plot_roc(best_gnb_rnd, X_test_scaled, y_test_bin, "GaussianNB (RandomizedSearch)")
print("RandomizedSearch Best model performance:", rnd_perf)
print(f"RandomizedSearch AUC: {rnd_auc:.4f}, Prediction time: {rnd_pred_time:.4f}s\n")
print("详细分类报告（RandomizedSearch 最佳模型）：")
print(classification_report(y_test_bin, best_gnb_rnd.predict(X_test_scaled), target_names=['<=50K','>50K'], zero_division=0))

print("汇总：")
print(f"Grid AUC: {grid_auc:.4f}    Random AUC: {rnd_auc:.4f}")
print(f"Grid F1 (weighted): {grid_perf['f1_weighted']:.4f}    Random F1 (weighted): {rnd_perf['f1_weighted']:.4f}")


# In[184]:


import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.preprocessing import label_binarize

best_gnb_grid = grid_search.best_estimator_
best_gnb_rnd = rnd_search.best_estimator_

def to_binary(y):
    y_s = np.array(y)
    if y_s.dtype.kind in 'iufc':
        return y_s.astype(int)
    return np.array([1 if (isinstance(v, str) and ('>' in v or v.strip().startswith('>'))) else 0 for v in y_s])

y_test_bin = to_binary(y_test)

def eval_and_plot_roc(model, X_test, y_test_bin, title_suffix):
    t0 = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]  
    pred_time = time.time() - t0

    y_pred = model.predict(X_test)

    perf = {
        'accuracy': accuracy_score(y_test_bin, y_pred),
        'precision_weighted': precision_score(y_test_bin, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test_bin, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test_bin, y_pred, average='weighted', zero_division=0)
    }

    # ROC / AUC
    fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], linestyle='--', lw=1.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {title_suffix}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    return perf, roc_auc, pred_time

# ---------- GridSearch ----------
print("GridSearch 最佳参数:", grid_search.best_params_)
grid_perf, grid_auc, grid_pred_time = eval_and_plot_roc(best_gnb_grid, X_test_scaled, y_test_bin, "GaussianNB (GridSearch)")
print("GridSearch Best model performance:")
for k, v in grid_perf.items():
    print(f"  {k}: {v:.3f}")
print(f"GridSearch AUC: {grid_auc:.3f}, Prediction time: {grid_pred_time:.3f}s\n")
print("详细分类报告（GridSearch 最佳模型）：")
print(classification_report(y_test_bin, best_gnb_grid.predict(X_test_scaled), target_names=['<=50K','>50K'], zero_division=0, digits=3))

# ---------- RandomizedSearch ----------
print("RandomizedSearch 最佳参数:", rnd_search.best_params_)
rnd_perf, rnd_auc, rnd_pred_time = eval_and_plot_roc(best_gnb_rnd, X_test_scaled, y_test_bin, "GaussianNB (RandomizedSearch)")
print("RandomizedSearch Best model performance:")
for k, v in rnd_perf.items():
    print(f"  {k}: {v:.3f}")
print(f"RandomizedSearch AUC: {rnd_auc:.3f}, Prediction time: {rnd_pred_time:.3f}s\n")
print("详细分类报告（RandomizedSearch 最佳模型）：")
print(classification_report(y_test_bin, best_gnb_rnd.predict(X_test_scaled), target_names=['<=50K','>50K'], zero_division=0, digits=3))

# ---------- 汇总 ----------
print("汇总：")
print(f"Grid AUC: {grid_auc:.3f}    Random AUC: {rnd_auc:.3f}")
print(f"Grid F1 (weighted): {grid_perf['f1_weighted']:.3f}    Random F1 (weighted): {rnd_perf['f1_weighted']:.3f}")

