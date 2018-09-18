## Machine-Learning-Classifiers

### 1. Naive Bayes Classifier and Logistic Regression: 
  - ### Dataset : http://archive.ics.uci.edu/ml/datasets/banknote+authentication
  - Compare the two approaches on the bank note authentication dataset, for each row the first four columns are the feature values and the      last column is the class label. Implemented a Gaussian Nave Bayes classifier and a logistic regression classifier. Did NOT use        existing functions or packages which can provide you the Naive Bayes Classifier/Logistic Regression class or fit/predict function. Used three-fold cross-validation to split the data and train/test your models.
   - Plotted a learning curve: the accuracy vs. the size of the training set. Plotted 6 points for the curve, using `.01 .02 .05 .1 .625 1` RANDOM fractions of training set and testing on the full test set each time. Plotted both the Naive Bayes and logistic regression learning curves on the same figure.
   
### 2. K-Nearest Neighbor 
- ### Dataset : http://yann.lecun.com/exdb/mnist/
- Evaluated the KNN classifier implemented by you on the famous MNIST data set where each example is a hand written digit. Each example
includes 28x28 grey-scale pixel values as features and a categorical class label out of 0-9. Implemented a KNN classifier with euclidean distance from scratch without using existing class or functions. 
- Plotted curves for training and test errors: the training/test error (which is equal to 1.0-accuracy) vs. the value of K. Plot 11 points for the curve, using `K = 1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99`. Plot the error curves for training error and test error in the same figure.

### 3. K-Means
- ### Dataset : http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/ 
-  Given a data set consisting of 4 examples a, b, c, d in two-dimensional space, assigned the 4 examples into 2 clusters using K-Means algorithm with Euclidean distance. To initialize the algorithm, a and c are in a cluster, b and d are in the other cluster. Implemented K-Means algorithm until convergence, including each cluster centroid and the cluster membership of each example after each iteration. 
- The data set contains 11 columns, separated by comma. The first column is the example id which should be ignored. The second to tenth columns are the 9 features, based on which you run K-means algorithm. The last column is the class label. Implemented K-Means algorithm to perform clustering on this dataset with `K = 2, 3, 4, 5, 6, 7, 8` .

### License

This project is licensed under the MIT License - see the [LICENSE.md] (https://github.com/Bharathgc/Machine-Learning-Classifiers/blob/master/LICENSE) file for details
