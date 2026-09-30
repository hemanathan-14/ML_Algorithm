import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df1 = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

dataset = dataset.sample(n=1000, random_state=42)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=0, stratify=y
)

kpca = KernelPCA(n_components=5, kernel='sigmoid', gamma=15)
try:
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)
except Exception as e:
    pass

classifier = GradientBoostingClassifier(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nPrecision : {:.4f}".format(precision_score(y_test, y_pred, average='weighted', zero_division=0)))
print("\nRecall : {:.4f}".format(recall_score(y_test, y_pred, average='weighted', zero_division=0)))
print("\nF1 Score : {:.4f}".format(f1_score(y_test, y_pred, average='weighted', zero_division=0)))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
