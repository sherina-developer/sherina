import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score



demographic = pd.read_csv('data/demographic.csv')
diet = pd.read_csv('data/diet.csv')
examination = pd.read_csv('data/examination.csv')
labs = pd.read_csv('data/labs.csv')
questionnaire = pd.read_csv('data/questionnaire.csv')

print("\nDemographic data shape:", demographic.shape)
print("Diet data shape:", diet.shape)
print("Examination data shape:", examination.shape)
print("Labs data shape:", labs.shape)
print("Questionnaire data shape:", questionnaire.shape)

print("\nDemographic preview")
print(demographic.head())


def safe_select(df, cols):
    available = [c for c in cols if c in df.columns]
    return df[['SEQN'] + available]

meta_demo = safe_select(demographic, ['RIDAGEYR', 'RIAGENDR'])
meta_exam = safe_select(examination, ['BMXBMI'])
meta_labs = safe_select(labs, ['LBXGLU', 'LBXTC'])
meta_sleep_activity = safe_select(questionnaire, ['SLD010H', 'PAQ605'])
meta_diet = safe_select(diet, ['DRABF'])



metahealth = meta_demo.merge(meta_exam, on='SEQN', how='left')
metahealth = metahealth.merge(meta_labs, on='SEQN', how='left')
metahealth = metahealth.merge(meta_sleep_activity, on='SEQN', how='left')
metahealth = metahealth.merge(meta_diet, on='SEQN', how='left')

print("\nMetaHealth dataset shape:", metahealth.shape)
print(metahealth.head())


metahealth = metahealth.rename(columns={
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Gender',
    'BMXBMI': 'BMI',
    'LBXTC': 'Cholesterol',
    'SLD010H': 'Sleep',
    'PAQ605': 'Activity',
    'DRABF': 'Breakfast'
})

print("Columns:", metahealth.columns.tolist())



metahealth['Risk_BMI'] = metahealth['BMI'] > 25
metahealth['Risk_Cholesterol'] = metahealth['Cholesterol'] > 200
metahealth['Risk_Sleep'] = metahealth['Sleep'] < 6
metahealth['Risk_Activity'] = metahealth['Activity'] == 0
metahealth['Risk_Breakfast'] = metahealth['Breakfast'] == 0

metahealth['Metabolic_Risk'] = (
    metahealth['Risk_BMI'] |
    metahealth['Risk_Cholesterol'] |
    metahealth['Risk_Sleep'] |
    metahealth['Risk_Activity'] |
    metahealth['Risk_Breakfast']
).astype(int)

metahealth = metahealth.drop(columns=[
    'Risk_BMI','Risk_Cholesterol','Risk_Sleep','Risk_Activity','Risk_Breakfast'
])

print("\nRisk Distribution")
print(metahealth['Metabolic_Risk'].value_counts())



X = metahealth[['Age','Gender','BMI','Cholesterol','Sleep','Activity','Breakfast']]
y = metahealth['Metabolic_Risk']

X = X.fillna(X.median())



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()

num_cols = ['Age','BMI','Cholesterol','Sleep']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])



logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

y_pred_logreg = logreg.predict(X_test_scaled)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))



tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))



fig, axes = plt.subplots(2,3, figsize=(16,10))
fig.suptitle("MetaHealth Metabolic Risk Analysis")

sns.countplot(x='Metabolic_Risk', data=metahealth, ax=axes[0,0])
axes[0,0].set_title("Risk Distribution")

sns.barplot(x='Metabolic_Risk', y='BMI', data=metahealth, ax=axes[0,1])
axes[0,1].set_title("Average BMI by Risk")

sns.barplot(x='Metabolic_Risk', y='Cholesterol', data=metahealth, ax=axes[0,2])
axes[0,2].set_title("Average Cholesterol")

sns.boxplot(x='Metabolic_Risk', y='BMI', data=metahealth, ax=axes[1,0])
axes[1,0].set_title("BMI Distribution")

sns.boxplot(x='Metabolic_Risk', y='Cholesterol', data=metahealth, ax=axes[1,1])
axes[1,1].set_title("Cholesterol Distribution")

sns.boxplot(x='Metabolic_Risk', y='Sleep', data=metahealth, ax=axes[1,2])
axes[1,2].set_title("Sleep vs Risk")

plt.tight_layout()
plt.show()



param_grid = {
    'C':[0.01,0.1,1,10],
    'penalty':['l2'],
    'solver':['liblinear']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=3
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)



y_pred = best_model.predict(X_test_scaled)

print("\nTuned Logistic Regression Accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



y_prob = best_model.predict_proba(X_test_scaled)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

print("AUC Score:", roc_auc_score(y_test, y_prob))



coefficients = best_model.coef_[0]

feature_importance = pd.Series(coefficients, index=X.columns)

feature_importance.sort_values().plot(kind='barh')

plt.title("Feature Importance")
plt.xlabel("Coefficient Value")

plt.show()



sns.heatmap(metahealth.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()



joblib.dump(best_model,"metabolic_model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("Model saved successfully")