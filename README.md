# Imbalanced Classification

## Employee Attrition

This work was part of a one month PoC for an Employee Attrition Analytics project at Honeywell International. I presented this notebook at a Honeywell internal data science meetup group and received permission to post it publicly. I would like to thank Matt Pettis (Managing Data Scientist), Nabil Ahmed (Solution Architect), Kartik Raval (Data SME), and Jason Fraklin (Business SME). Without their mentorship and contributions, this project would not have been possible.

This project consists of two notebooks:

[_Imbalanced Classification with mlr_](Imbalanced_Classification_with_mlr.md)  
    This notebook presents a reference implementation of an imbalanced data problem, namely predicting employee attrition. We'll use [`mlr`](https://mlr-org.github.io/mlr/index.html), a package designed to provide an infrastructure for Machine Learning in R. Additionally, this notebook considers post-modeling steps such as interpretability and productionisation.

[_Vanilla vs SMOTE_](VanillaVsSMOTE.md)  
     In this notebook, we investigate whether SMOTE actually improves model performance. We compare vanilla (non-SMOTE) verses SMOTE flavored models using **logistic regression**, **decision trees**, and **randomForest**. We also consider how tuning model operating thresholds and tuning SMOTE parameters impact the results.