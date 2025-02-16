# Federated Learning for Health Outcomes Predictions in a Multicentric Sample of Hospitals

## Abstract

We evaluated the performance of federated learning (FL) strategies in predicting
COVID-19 mortality among hospitalized patients, in a multicenter longitudinal sample
of 21 Brazilian hospitals that included 17,022 patients and 22 predictors. We tested two
scenarios based on a horizontal structure of FL. In the first scenario, we analyzed two
standard models, a deep Multi-Layer Perceptron neural network (MLP), and a Logistic
Regression (LR). In the second scenario, we developed a federated learning approach
inspired by the Random Forest algorithm. We then compared the predictive
performance of the FL models to that of local training data from each hospital. The
results indicated that the FL models had a higher average AUC-ROC, resulting in a
better predictive performance than models trained using only local data. The average
gain for the FL models was 7.21% for LR, 12.79% for MLP, and 6.48% for the Random
Forest approach. The study also observed that the FL models had a more significant
predictive gain in hospitals with a smaller number of patient data.

## Author summary
The authors, coming from diverse fields of expertise, present a comprehensive analysis of
federated learning—a decentralized machine learning approach where multiple
institutions collaboratively train a model while keeping data localized to ensure privacy
and security. This study evaluates the effectiveness of federated learning architectures
that aggregate model parameters through averaging in predicting COVID-19 mortality.
By applying this methodology across 21 hospitals throughout Brazil, the study
investigates its utility across various patient volume contexts and assesses its predictive
performance. Additionally, the authors explore federated models based on decision trees
and propose the development of a self-scalable random forest algorithm to enhance
predictive capabilities and adaptability. The findings suggest that federated learning
holds promise as a powerful solution for predictive challenges in healthcare settings,
fostering both innovation and data security.

## CONCLUSION

The use of federated learning methodology demonstrated significant improvements over local learning across all tested models. Despite the heterogeneous distribution of data across Brazilian regions, federated learning effectively captured patterns and contributed to predictive gains in COVID-19 mortality throughout most of the training process. The most notable improvements were observed in hospitals with smaller patient populations.

The decision tree-based federated learning model proposed in our study achieved an average AUC-ROC of 0.8022, with low variability among the evaluated hospitals (Coefficient of Variation: 7.30%). Overall, the federated learning approach enhanced the performance of algorithms in predicting COVID-19 mortality for hospitalized patients, particularly in a country marked by significant healthcare inequalities.
