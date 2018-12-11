import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import seaborn as sns
from scipy.stats import norm

#classifiers
from sklearn.linear_model import LogisticRegression

data_raw = pd.read_csv("creditcard.csv", header=0)

# check class distributions
count_classes = pd.value_counts(data_raw['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Class Distribution of Raw Data")
plt.xlabel("Class")
plt.ylabel("Frequency")
# plt.savefig('class_dstbr', dpi=300)
# plt.show()

# extract the true class labels from the features
X, y = data_raw.iloc[:, 0:29].values, data_raw.iloc[:, 30].values

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

# do the random undersampling

data_raw = data_raw.sample(frac=1)

fraud_data = data_raw.loc[data_raw['Class'] == 1]
non_fraud_data = data_raw.loc[data_raw['Class'] == 0][:492]

data = pd.concat([fraud_data, non_fraud_data])

colors = ["#0101DF", "#DF0101"]
sns.countplot('Class', data=data, palette=colors)
plt.title('Class Distribution of Undersampled Data', fontsize=14)
# plt.savefig('class_dstbr_undersample', dpi=300)
# plt.show()

# Correlation matrix

f, (ax1) = plt.subplots(figsize=(16,16))

sub_sample_corr = data.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title('Correlation Matrix', fontsize=14)
plt.savefig('correlation_matrix', dpi=300)
plt.show()


# visualize distributions

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_fraud_dist = data['V14'].loc[data['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = data['V12'].loc[data['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)


v10_fraud_dist = data['V10'].loc[data['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)
# plt.savefig('normal distribution', dpi=300)
# plt.show()

# remove outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

data_out = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

# visualize data distribution after outlier removal
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_fraud_dist = data_out['V14'].loc[data['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = data_out['V12'].loc[data['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)


v10_fraud_dist = data_out['V10'].loc[data['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)
# plt.savefig('outliers_removed', dpi=300)
# plt.show()

# create new training and test sets
X_new, y_new = data_out.iloc[:, 0:29].values, data_out.iloc[:, 30].values
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.3, random_state=0)


# train models
lreg = LogisticRegression(C=1.0)
lreg.fit(X_train_new, y_train_new)
y_pred_lreg_dev = lreg.predict(X_dev)
y_pred_lreg_test = lreg.predict(X_test)

lreg_1 = LogisticRegression(C=1.0)
lreg_1.fit(X_train, y_train)
y_pred = lreg_1.predict(X_test)
y_pred_dev = lreg_1.predict(X_dev)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_lreg_test)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=300)
# plt.show()

print("Recall score on dev sets with logistic regression (trained with undersampled data):", recall_score(y_dev, y_pred_lreg_dev, average='macro') * 100)
print("Recall score on test sets with logistic regression (trained with undersampled data):", recall_score(y_test, y_pred_lreg_test, average='macro') * 100)

print("Recall score on dev sets with logistic regression (trained with full dataset):", recall_score(y_dev, y_pred_dev, average='macro') * 100)
print("Recall score on test sets with logistic regression (trained with full dataset):", recall_score(y_test, y_pred, average='macro') * 100)
