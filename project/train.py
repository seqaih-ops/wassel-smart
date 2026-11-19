"""Wassel Smart — reproducible educational model. All data are synthetic."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, confusion_matrix, brier_score_loss

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUTPUT = ROOT / 'outputs'
OUTPUT.mkdir(exist_ok=True)
RAW_FEATURES = ['Distance_Km', 'Driver_Orders', 'Traffic_Level']
FEATURES = RAW_FEATURES + ['Distance_Traffic', 'Orders_Traffic']
TARGET = 'Delayed'
RANDOM_STATE = 42

# 1. Synthetic data generation: assumptions, not observed delivery records.
# An unobserved random factor represents variation absent from the inputs.
rng = np.random.default_rng(RANDOM_STATE)
n = 3000
distance = rng.uniform(1, 50, n).round(1)
orders = rng.integers(0, 11, n)
traffic = rng.choice([1, 2, 3], n, p=[.35, .4, .25])
log_odds = (-4.8 + .065 * distance + .22 * orders + .55 * (traffic - 1)
            + .025 * distance * (traffic - 1) + .09 * orders * (traffic - 1)
            + rng.normal(0, .6, n))
probability = 1 / (1 + np.exp(-log_odds))
df = pd.DataFrame({'Distance_Km': distance, 'Driver_Orders': orders,
                   'Traffic_Level': traffic, 'Delayed': rng.binomial(1, probability)})
df.to_csv(OUTPUT / 'synthetic_deliveries.csv', index=False)
print('Synthetic records:', len(df), '| Delayed prevalence:', round(df[TARGET].mean(), 3))

# 2. Keep 20% untouched during training, preprocessing, and model tuning.
X_train, X_test, y_train, y_test = train_test_split(
    df[RAW_FEATURES], df[TARGET], test_size=.2, stratify=df[TARGET], random_state=RANDOM_STATE)

def add_delivery_features(data):
    result = data[RAW_FEATURES].copy()
    result['Distance_Traffic'] = result.Distance_Km * result.Traffic_Level
    result['Orders_Traffic'] = result.Driver_Orders * result.Traffic_Level
    return result

def pipeline(classifier):
    return Pipeline([('features', FunctionTransformer(add_delivery_features)),
                     ('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()), ('classifier', classifier)])

# 3. Baselines and the Lab 3 gradient-boosting search, training folds only.
models = {
    'Logistic Regression': pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    'Random Forest': pipeline(RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=8, random_state=42)),
    'Default Gradient Boosting': pipeline(GradientBoostingClassifier(random_state=42))
}
for model in models.values():
    model.fit(X_train, y_train)
search = GridSearchCV(pipeline(GradientBoostingClassifier(random_state=42)),
    {'classifier__n_estimators':[60,120], 'classifier__learning_rate':[.04,.08],
     'classifier__max_depth':[1,2]}, scoring='average_precision',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), error_score='raise')
search.fit(X_train, y_train)
best_model = search.best_estimator_
models['Tuned Gradient Boosting'] = best_model
print('Best training CV AP:', round(search.best_score_, 4), '| Parameters:', search.best_params_)

# 4. Holdout evaluation once; fixed 0.5 threshold, not optimized on test labels.
# The deployed model is the CV-selected Gradient Boosting model, not a test-set winner.
rows=[]
for name, model in models.items():
    p=model.predict_proba(X_test)[:,1]
    pred=(p >= .5).astype(int)
    rows.append({'model':name, 'AP':average_precision_score(y_test,p),
        'ROC_AUC':roc_auc_score(y_test,p), 'F1':f1_score(y_test,pred),
        'accuracy':accuracy_score(y_test,pred), 'Brier':brier_score_loss(y_test,p)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
p=best_model.predict_proba(X_test)[:,1]
metrics={'data_source':'synthetic', 'records':n, 'train_records':len(X_train),
    'test_records':len(X_test), 'prevalence':float(y_test.mean()), 'seed':42,
    'threshold':.5, 'cv_ap':float(search.best_score_), 'best_params':search.best_params_,
    'models':rows, 'confusion_matrix':confusion_matrix(y_test,p>=.5).tolist(),
    'sklearn_version':sklearn.__version__}
(OUTPUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
pd.DataFrame(rows).to_csv(OUTPUT/'model_comparison.csv',index=False)

# 5. Export actual trained trees for identical inference in the browser.
clf=best_model.named_steps['classifier']
scaler=best_model.named_steps['scaler']
prior=clf.init_.class_prior_[1]
trees=[]
for est in clf.estimators_[:,0]:
    t=est.tree_
    trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),
                  'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),
                  'value':t.value[:,0,0].tolist()})
export={'features':FEATURES,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
    'initial_log_odds':float(np.log(prior/(1-prior))), 'learning_rate':clf.learning_rate,
    'trees':trees,'importance':clf.feature_importances_.tolist()}
(OUTPUT/'model.json').write_text(json.dumps(export,separators=(',',':')))
# Fixtures validate the exported evaluator against scikit-learn, not hand-coded scores.
fixture_x=X_test.iloc[:200]
(OUTPUT/'parity_fixtures.json').write_text(json.dumps({'inputs':fixture_x.to_dict('records'),
    'probabilities':best_model.predict_proba(fixture_x)[:,1].tolist()}))
print('Exported model and evaluation artifacts to', OUTPUT)
