# Credit Risk Analysis

## Project Overview

Credit risk is the possibility of a loss resulting from a borrower's failure to repay a loan or meet contractual obligations. To determine credit risk, it requires the creditors to evaluate customers based on their credit score. As a result from this, there are classification imbalances with credit risk because good loans outnumber riskier loans. We are tasked to build a classification model using machine learning statistical algorithms to make predictions on the credit risk of a client. In our analysis, we will be using the credit card credit dataset from LendingClub, a peer-to-peer lending services company. We will utilize different machine learning techniques such as `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, `SMOTEENN`, `BalancedRandomForestClassifier`, and `EasyEnsembleClassifier` to train and evaluate data to build a recommendation for the best machine learning model to use for credit risk predictions.

## Resources

+ Analysis Software: `Python 3.10`, `Jupyter Notebook 6.4.12`
+ Data Source: `LoanStats_2019Q1.csv`

## Results

### Resampling

In each analysis with the resampling models, we used the resampled data to train a logistic regression model and calculated the balanced accuracy score from `sklearn.metrics`, printed the confusion matrix, and generated a classification report from `imbalanced-learn`.

#### Naive Random Oversampling

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. Oversampling addresses class imbalance by duplicating or mimicking existing data. 

Balanced Accuracy Score:
```
0.663188044716539
```

Confusion Matrix:


Classification Report: 


The Naive Random Oversampling model accurately predicts credit risk 66.3% of the time. Additionally, the precision of the model for high risk is 0.01 and low risk is 1.00. In other words, when it predicts that a client is high risk, it is correct 1% and when it predicts that a client is low risk, it is correct 100% of the time. The recall in our model 0.75 for high risk and 0.57 for low risk. This means that it correctly identifies 75% of all high risk and 57% for all low risk.

#### SMOTE Oversampling

The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

Balanced Accuracy Score:
```
0.6621894942066704
```

Confusion Matrix:


Classification Report: 


The SMOTE Oversampling model accurately predicts credit risk 66.2% of the time. Additionally, the precision of the model for high risk is 0.01 and low risk is 1.00. In other words, when it predicts that a client is high risk, it is correct 1% and when it predicts that a client is low risk, it is correct 100% of the time. The recall in our model 0.63 for high risk and 0.69 for low risk. This means that it correctly identifies 63% of all high risk and 69% for all low risk.

#### Cluster Centroids (Undersampling)

Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

Balanced Accuracy Score:
```
0.5447339051023905
```

Confusion Matrix:


Classification Report: 


The Cluster Centroids model accurately predicts credit risk 54.4% of the time. Additionally, the precision of the model for high risk is 0.01 and low risk is 1.00. In other words, when it predicts that a client is high risk, it is correct 1% and when it predicts that a client is low risk, it is correct 100% of the time. The recall in our model 0.69 for high risk and 0.40 for low risk. This means that it correctly identifies 69% of all high risk and 40% for all low risk.

#### SMOTEENN (Combination Sampling)

SMOTEENN is an approach to resampling that combines aspects of both oversampling and undersampling.

Balanced Accuracy Score:
```
0.644711676499736
```

Confusion Matrix:


Classification Report: 


The SMOTEENN model accurately predicts credit risk 64.5% of the time. Additionally, the precision of the model for high risk is 0.01 and low risk is 1.00. In other words, when it predicts that a client is high risk, it is correct 1% and when it predicts that a client is low risk, it is correct 100% of the time. The recall in our model 0.72 for high risk and 0.57 for low risk. This means that it correctly identifies 72% of all high risk and 57% for all low risk.

### Ensemble Learners

In each analysis with the ensemble models, we trained the model using training data and calculated the balanced accuracy score from `sklearn.metrics`, printed the confusion matrix, and generated a classification report from `imbalanced-learn`.

#### Balanced Random Forest Classifier

Instead of having a single, complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.

Balanced Accuracy Score:
```
0.7885466545953005
```

Confusion Matrix:


Classification Report: 


The SMOTEENN model accurately predicts credit risk 78.9% of the time. Additionally, the precision of the model for high risk is 0.03 and low risk is 1.00. In other words, when it predicts that a client is high risk, it is correct 3% and when it predicts that a client is low risk, it is correct 100% of the time. The recall in our model 0.70 for high risk and 0.87 for low risk. This means that it correctly identifies 70% of all high risk and 87% for all low risk.

#### AdaBoost Classifier

In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized.

Balanced Accuracy Score:
```
0.9316600714093861
```

Confusion Matrix:


Classification Report: 


The SMOTEENN model accurately predicts credit risk 93.2% of the time. Additionally, the precision of the model for high risk is 0.09 and low risk is 1.00. In other words, when it predicts that a client is high risk, it is correct 9% and when it predicts that a client is low risk, it is correct 100% of the time. The recall in our model 0.92 for high risk and 0.94 for low risk. This means that it correctly identifies 92% of all high risk and 94% for all low risk.

## Summary





