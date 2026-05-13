import pandas as pd, numpy as np, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap

os.makedirs('reports/figures', exist_ok=True)

df = pd.read_csv('data/framingham.csv')
df.drop(['education'], axis=1, inplace=True)
df.rename(columns={'male':'gender'}, inplace=True)

FEATURES = ['age','gender','currentSmoker','cigsPerDay','BPMeds',
            'prevalentStroke','prevalentHyp','diabetes','totChol',
            'sysBP','diaBP','BMI','heartRate','glucose']
X = df[FEATURES]
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test_imp)

# Plot 1: Class Distribution
fig, ax = plt.subplots(figsize=(7,5))
counts = y.value_counts()
ax.bar(['No CHD','CHD Risk'], counts.values,
       color=['#2ecc71','#e74c3c'], edgecolor='white', linewidth=2)
for i,v in enumerate(counts.values):
    ax.text(i, v+30, str(v), ha='center', fontweight='bold', fontsize=13)
ax.set_title('Class Distribution — Severe Imbalance (5.6:1)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('reports/figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('1. class_distribution.png saved')

# Plot 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(12,9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size':8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('2. correlation_heatmap.png saved')

# Plot 3: ROC Curves
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}
fig, ax = plt.subplots(figsize=(9,7))
colors = ['#e74c3c','#3498db','#2ecc71']
for (name, m), color in zip(models.items(), colors):
    m.fit(X_train_sc, y_train_sm)
    y_prob = m.predict_proba(X_test_sc)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
ax.plot([0,1],[0,1],'k--', lw=1.5, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('reports/figures/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('3. roc_curves.png saved')

# Plot 4: SHAP
rf = models['Random Forest']
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_sc[:150])
sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:,:,1]
plt.figure(figsize=(10,7))
shap.summary_plot(sv, X_test_sc[:150], feature_names=FEATURES, show=False)
plt.title('SHAP Feature Impact on CHD Prediction', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('4. shap_summary.png saved')

print('\nALL DONE! Check reports/figures/ folder')