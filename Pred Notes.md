# Overview

Data science = set of **principles** to guide extraction of knowledge from data  
$\quad$ Machine learning = the science & art of programming computers so they can learn from data.  
$\quad$ Machine learning is an application of data science 
$\quad$$\quad$ Data mining = the actual **extraction** using technology  
$\quad$$\quad$ Data mining is the process of coming up with a model based on historical data  
$\quad$$\quad$ Data mining is hypothesis generation, stats is hypothesis testing  
$\quad$$\quad$ Data mining is an application of machine learning

**Data mining tasks** 
1) Classification - which set of class does this person belong to?  
2) Regression - How many hours will this person use our service?   
3) Similarity matching - These firms bought from us. Who else is likely to?  
4) Clustering - What segments do our customers fall into?  
5) Co-occurence (market basket analysis) - For each segment, what are commonly purchased together?  
6) Profiling - What is the typical behavior of this segment?  
7) Link prediction - You and x share 10 friends. She likes this person, you prob will too     
8) Data reduction - Dropping unnecessary info thats clouding our insights  
9) Causal modeling - What influences x?  


**Data mining process** 
1) Business understanding  
2) Data understanding - how was it collected? for free? biases?    
3) Data prep  
4) Modeling  
5) Data prep again (possibly)  
6) Modeling again (possibly)  
7) Evaluate model  
8) Deploy model

**Types of data mining models**  
1) **Classification** (Categorical) 
2) **Regression** (Numeric data) 


Hyperparameter = something that we set (eg max depth of the tree)  
Parameter = learns from the training data, does not need our intervention

Creating model = induction  
Using model = deduction

1) Accuracy = total correct / total  
2) Precision (pos) = true positives / total predicted positives  
$\quad$ Are you just flagging everything as positive?  
3) Recall (pos) = true positives / total actual positives  
$\quad$ Are you grabbing the ones that actually were positive?

---

### ML Process

1) Split into    
`Train` 70% - run each k here   
`Valid` 10% - select k with best result - this is the hyperparameter we will use  
`Test` 20% - final test   
2) Create a model using the `training` set with each of the possible hyperparameters  
3) Run the model on the `validation` set  
4) Select hyperparamter that led to best `validation` set result  
5) Re-run the model on the `Train` + `Valid`  
6) Evaluate using the `test` set

**Training set** = examples that the system uses to learn 
**Training instance** = each training example 
**Attribute** = feature = data type (e.g., “Mileage”) 
**Feature** generally means an attribute + value (e.g., “Mileage = 15,000”)

$\quad$ Xtrain --> non target columns, attributes, features 
$\quad$ Ytrain --> target columns, attributes, features 
$\quad$ Xtest --> new non target data (should come from same distribution) 
$\quad$ Ytest --> our prediction, compare to ground truth



**ML useful for...** 
1) **Data mining** = Applying ML to dig into large data to discover patterns that were not immediately apparent. 
2) Fluctuating environments that may change over time 
3) Problems that require tons of rules

---

### **Types of ML Systems (not exclusive)** - pg 20-40

1a) **Supervised** - training data includes the desired solutions (ie labels) 
Classification - trained with many examples, class would be spam/not spam 
Regression - predict numeric value given a set of predictors. 
k-Nearest Neighbors 
Logistic Regression 
Support Vector Machines (SVMs) 
Decision Trees and Random Forests 
Neural networks


1b) **Unsupervised** - training data does not include the desired solutions 
**Clustering** (k-Means, Hierarchical Cluster Analysis (HCA), Expectation Maximization) 
**Visualization & dimensionality reduction** (Principal Component Analysis (PCA), Kernel PCA, Locally-Linear Embedding (LLE), t-distributed Stochastic Neighbor Embedding (t-SNE)) 
Dimensionality reduction = simplify without losing too much info 
Feature extraction = merge multico into one feature that represents the info 
Anomaly detection 
**Association rules** (Apriori, Eclat) 


1c) **Semisupervised** - partially labeled training data (usually a lot of unlabeled data + some labeled data) 
Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms.
Deep belief networks (DBNs), restricted Boltzmann machines (RBMs)

1d) **Reinforcement learning**
The learning system (ie agent) observes environment, performs actions, and get rewards. 
Learns by itself what is the best strategy (ie policy) to max reward

---

**2) Batch learning | online - can they learn on the fly?** 

2a) **Batch learning** (offline learning) - system is trained and then put into production 
If you want an update, you need to train a new version with the old and new data 
Takes alot of time and computing resources

2b) **Online learning** 
Train the system incrementally by feeding it data instances sequentially in mini batches 
Fast and cheap, learns on the fly 
Good when data is continuous flow 
**Learning rate** = how fast system should adapt to changing data 
High learning rate = rapidly adapt to new data & quickly forget the old data  
If bad data is fed to system, performance will decline. Need to monitor closely

An out-of-core learning algorithm chops the data into mini-batches and uses online learning techniques to learn from these mini-batches

---

**3) Instance-based | model-based learning - how do they generalize? just compare new data to old? or detect patterns and predict?**

3a)  **Instance based** - learn by heart and then extrapolate to similar cases 
measure of similarity = how close something is to the provided example

3b) **Model based** - build model, use model to make predictions 
Utility function and cost function 
Inference = predictions on new cases

---

### Main Challenges of ML (pg 41 - 52)

**1) Bad data** 
a) Bad quantity - typically need thousands of examples 
b) Bad quality - too much missing info, poorly collected 
c) Nonrepresentative - training needs to cover cases similar to new cases 
d) Feautures - select most useful features, combine features to produce better ones

**2) Overfitting / overgeneralizing** - fix by.. 
a) Gather more data 
b) Reduce noise by fixing errors and removing outliers 
c) Regularization - Simplify by picking fewer parameters (linear instead of poly), reduce number of columns used 
$\quad$ Hyperparameter = parameter of learning algorithm to determine amount of regularization

**3) Underfitting** - fix by.. 
a) Add parameters 
b) Feed better features (relevant info) 
c) Reduce constraints (reduce regularization hyperparameter)

---

### Fine tuning (pg 52 - 53)


1) Split your data into two sets: training set & test set. (Usually 80% of the data for training and hold out 20% for testing) 
1b) Use cross validation to split up training set 
2) Run multiple models with different hyperparameter for regularization on training set 
3) Select model with best performance on validation set 
4) Find the **generalization error** on the test set. Tells you how well your model will perform on instances it has never seen before. 
$\quad$ If training error (mistakes on training) is low but generalization error (mistakes on testing) is high, we have overfit

---

### Exercises
12) A model has one or more model parameters that determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances. A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularization to apply).

13) Model-based learning algorithms search for an optimal value for the model parameters such that the model will generalize well to new instances. We usually train such systems by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized. To make predictions, we feed the new instance’s features into the model’s prediction function, using the parameter values found by the learning algorithm.

17) A validation set is used to compare models. It makes it possible to select the best model and tune the hyperparameters.

18) If you tune hyperparameters using the test set, you risk overfitting the test set, and the generalization error you measure will be optimistic (you may launch a model that performs worse than you expect).

19) Cross-validation is a technique that makes it possible to compare models (for model selection and hyperparameter tuning) without the need for a separate validation set. This saves precious training data.







# Classification

[Code](http://localhost:8888/notebooks/1%20Predictive/2%20-%20Classification%20(Trees%2C%20kNN%2C%20Logistic)/Comparing%20Classifiers.ipynb)

<img src=https://i.imgur.com/rljQgL9.jpg width="900" height="300" align="left">

---

## Overview

**One of main tasks of data mining = find and select most informative variables**  
Supervised segmentation = segment population into groups that differ in some way from each other   

Purity measures = tell us how pure our groups are after we segment  
Information gain = evaluates effectiveness of new segment vs old in getting pure groups   
For each additional "child" we add, need to run a chisquared test to make sure improvement is stat significant


**Regression with Decision Trees**  
Instead of minimizing impurity like we were doing in classification above, we are now looking to minimize mse (mean squared error)  


---



## Purity measures - How good are our groups?

**1) Gini impurity (pure = 0, impure = .5) - Uses squares**    

1. Looks at only the "final group" from each tree  
2. We provide the char of the new observation  
3. For each final group, calculates $\quad$ $number \: with \: the \: characteristics \: / \: total \: number \: in \: group$  
4. Squares each of the ratios from above  
5. Takes $1 \: - \: each \: squared \: ratio$

**2) Entropy (pure = 0, impure = .5) - Uses logs**    

1. Calculates this ratio for each group:   
   $\quad$ $Ratio \: = \: number \: with \: the \: characteristics \: / \: total \: number \: in \: group$  
2. For each group, calculates   
   $\quad$ $Each \: group \: = \: -1 \: * \: ratio \: * \: log(ratio)$  
3. Sums each group from above

Gini impurity is slightly faster  
Gini impurity tends to isolate the most frequent class in its own branch  
Entropy tends to produce slightly more balanced trees


---

## Regularization Hyperparameters - Fine tuning results


**Parametric** = predetermined number of parameters, limited degrees of freedom, limited risk of overfitting  
**Nonparametric** = not constrained, will likely overfit  
$\quad$ Need to use regularization while training to resolve this  

---

## Performance Evaluation - Confusion Matrix


<img src=https://i.imgur.com/GqFRVwK.png width="400" height="340" align="left">



**F measurement** - combines precision and recall  
Looking at pos only --- $F \: = 2 * \frac{Precision \: * \: Recall}   {Precision \: + \: Recall}$

F measurement will be 1 when precision and recall are 1 (ideal)  
F measurement will be 0 when precision and recall are 0  


1) Accuracy = total correct / total  
2) Precision (pos) = true positives / total predicted positives  
3) Recall (pos) = true positives / total actual positives



```python

```


```python

```

---

### Naive Bayes

`Bayes Theorem`  
Intuition - For each row, given their chars, what is the prob that they fall into each group within our target column?  
$\quad$ This logic is used in logistic regression, trees, k-nn

**Prob of being in group** (run for each of the columns, uses normal or log normal etc dist for numeric columns)  
= prop of my attribute given that they were in the group  
\* prop of dataset in the group  
/ prop of dataset with my attribute

More robust for large number of variables than logistic regression  
Logistic regression will be better when there is high correlation between 2 columns

<img src=https://i.imgur.com/pgtG11S.png width="400" height="340" align="left">


```python

```



# Classification Evaluation

### Single Measures of Model Performance

**Limitations of accuracy**  
1\. Unbalanced class distributions (99% of companies succeeding)  
$\quad$ For each class, should also look at precise, recall, f  
2\. Unequal costs and benefits for each group (medical diagnosis)


**1) Matthews Correlation Coef** - Between -1 and +1 (perfect)  
**2) Kappa Stat - compares to random** - Always <= 1

### ROC Curve - used to compare models against each, not to select the best threshold

We dont want to just see how many we got correct or wrong, we want see our performance based on different thresholds. 
We can get a confusion matrix for each different threshold setting

We want to see which model tends to perform best no matter what threshold we use. (ROC curve)
Then, we can use the best threshold setting

For each confusion matrix
Calc TP and FP (calcing down the column, our pred vs ground truth)

ROC curve only cares about looking down the labels for pred vs ground truth, doesn't care about unbalanced data (ie label distribution)

Roc curve - each dot represents a different threshold
For each threshold setting, we use the TP and FP to plot a dot 
FP on the xaxis, TP on the yaxis
Point (0,0) on the curve occurs when threshold is 1 (we would always predict as going to the - group) (very picky)
Point (1,1) on the curve occurs when threshold is 0 (we would always predict as going to the + group)

Worst case --> go from (0,0) to (1,0) to (1,1)  
Best case --> go from (0,0) to (0,1) to (1,1)  
Random case --> diagonal line (this will not change even if we have unbalanced data)

### AUC - used when we need a single number so summarize performance or when we dont know anything about operating conditions

If the models aren't purely better or worse than one another, we calculate the area under the curve of each model.
We select the model with the greatest area under the curve
Should only be looking at the ROC curve though to compare models


### 2. Precision Recall Curve - Sensitive to unbalanced data

Remember:  
Precision: looks at the ones we predicted (true pos / total pred pos)  
Reall: looks at the ground truth (true pos / total pos)

As our rate of our fraction of truth increases, how many are we predicting that are wrong?

We want the model with the greatest area under the curve

Best case --> go from (0,1) to (1,1) to the ratio of + group / - group (top heavy stops sooner)  
Worst case --> go from (0,0) to the ratio of + group / - group (top heavy stops sooner)  
Random case --> flat line, will again be the ratio of + group / - group (top heavy stops sooner)



### 3. Cumulative response curve

### 4. Lift curve - improvement over random guess - compare model to model

x = percent of our dataset (approaches total)  
y = correct

### 5. Validation curve - visually showing why a selection of hyperparameters is best - helps use see if we are over or under fitting



## Limits of Accuracy Measures

Need to consider cost/benefit of making an incorrect prediction (misdiagnosis)

How - Build out a cost matrix - matches size of confusion matrix
Set top left and bottom right as rev
Set top right and bottom left as cost

Multiply out. We are looking to find the max combo.

We set the cost/benefit matrix based on domain knowledge.
We can then fine tune the threshold to find the max. These are the decisions we should make



# Regression

## Max: Gradient Descent

Shown in code guide

## Min: Regularization

1. (L1) Lasso removes the unneeded columns - may be eratic
   1. Won't work well when more cols than rows 
   2. When cols are correlated, removes one of the cols at random
2. (L2) Ridge - doesn't do feature selection
3. (L1 + L2) Elastic net
   1. Use this when some cols are correlated and we want to do feature selection  



## Evaluation

### Measures -  `Total SSE`, `MAE`, `MAPE`, `RMSE`, `SMAPE`  

MAPE when percent of error matters 
RMSE when want to penalize large errors

Even errors: RMSE = MAE 
Small variance errors: RMSE > MAE by a little 
Large variance errors: RMSE > MAE by a lot

`SMAPE` (when we want the target to be 0) 
Goes from 0 to 200% 
Apply when we are comparing average error of different models 
Does not apply when we are looking at each observation

---

### Charts - `Lift` and `Learning Curve`

Ranked by their predicted number, comparing to the average  

<img src=https://i.imgur.com/np7zERs.png width="400" height="340" align="left">



## Generalized Linear Model

`Linear` when -inf to +inf 
Gamma when y always pos 
Inv Gause when y always pos

`Poisson` when when only integers in target
(Not good when variance is larger than mean)

`Negative binomial` when only integers in target
(Better when variance is larger than mean)





# Midterm

### Notes from Song

https://canvas.umn.edu/courses/139500/pages/week-7

**SVM**

1. (SVM) rbg kernel large gamma = far away neighbors don't really matter
2. (SVM) large C = tight street


**Eval**

3. Nested CV - outer is normal CV to eval, inner is Gridsearch to tune 
4. Measurements for unbalanced - MCC, Kappa, Lift curve, Prec-Rec curve, (not accuracy, ROC)
5. ROC curve - TP rate / FP rate  
   Lift curve - Lift over random  / percent of instances (decreasing by score)
6. MCC range from -1 to 1


**Reg**

7. Regularization

 - L1 does feature selection auto, not stable when part of feature is highly correlated
 - L2 is stable when part of feature is highly correlated
 - Elastic net will generate different coefs for uncorr vs corr


**Naive Bayes**

- Prior dist = percent of positives, percent of negatives

---

**Classification** 

2. SVM with rbf - Lots of white where predictions won't be good (when points are on the street)
3. Naive Bayes - when numerical data, we can generate a "normal curve" type of shape
4. Logistic or SVM with linear kernel - linear decision boundary
5. KNN - Very curved, not a real "pattern" to the curve
6. Tree - orthoganol





---

### Training Models (Ch 4)

1. If we have millions of obs, should use Stoch or mini-batch
2. Gradient descent struggles if we don't scale. Normal equation doesn't care.
3. Gradient descent won't get stuck in a local min with logistic.
4. If problem is convex (eg linear, logistic) and learning rate not too high, will find similar models each time
5. Increase in both val error & train error = learning rate too high  
   Increase in val error, not train error = overfitting
6. Stoch and Mini improve on average.
7. Stoch is fastest, only Batch will 100% converge
8. Poly reduce overfit  
   (1) Reduce degree, (2) regularize, (3) add more training data
9. Underfitting = high bias

---

### SVM (Ch 5)

1. Fundamental idea of svm - find widest street between two classes (hard margin)  
   Compromise between widest street and minimizing errors (soft margin)
2. Support vector = any instance on the street or on the border  
   Predictions only depend on the support vectors.
3. If we don't scale features, wider range ones will dominate
4. SVM can't directly output probability. We can use the distance obs and decision boundary to compute logistic regression though.
5. Kernelized can only use dual form to train.  
   For linear - primal complexity proportional to obs, dual complexity proportional to between $obs^2$ and $obs^3$
6. If we underfit with rbf kernel - should increase gamma or C or both

---

### Trees (Ch 6)

1. Tree depth of 1mil instance = $log(1mil)^2 = 20$ using base 2
2. Gini impurity generally lower than parent. One child could be higher if the other child is significantly lower.
3. Overfitting = decrease max depth
4. Don't need to scale features
5. Time to train = $rows * features * log(rows)$
6. `presort=True` only when small dataset

---

### Ensemble (Ch 7)

1. If we have models from different algos with similar precision, use a voting example. Ideal if each trained on different samples as well.
2. Hard vote classifiers = assigns class based on vote frequency of each of the voters.  
   Soft vote classifiers = assigns class based on average probability from each of the voters. More confidence of the voter = higher weight. Usually performs better. Requires that each voter have ability to create probability though.
3. Bagging and pasting can be distributed, boosting is sequential. 
4. (Boosting) Out of bag lets us test predictor using the instances it didn't use without having to create another validation set.
5. Extra trees use random thresholds to split.
6. If Adaboost underfits - increase number of estimators, reduce regularization of base predictor, increase learning rate
7. If gradient boosting overfits - decrease learning rate, use early stopping





# Final

**Useful links**

- [Canvas Page](https://canvas.umn.edu/courses/139500)
- [Reading List](https://canvas.umn.edu/courses/139500/pages/reading-list)
- [Self Assess](https://canvas.umn.edu/courses/139500/pages/weekly-self-assessment-questions)
- [Schedule](file:///C:/Users/Sam/Desktop/Predictive/Syllabus/Schedule.pdf)
- [Jupyter Notes](http://localhost:8888/tree/1%20Predictive)


- [Midterm Example](file:///C:/Users/Sam/Desktop/Predictive/MSBA6420%20-%20Midterm%20Exam-Example.pdf)
- [Midterm Notes](http://localhost:8888/notebooks/1%20Predictive/Midterm.ipynb)
- [Final Example](file:///C:/Users/Sam/Desktop/Predictive/Slides/MSBA6420%20-%20Final%20Exam%20Example.pdf)
- [Final Exam Slides](file:///C:/Users/Sam/Desktop/Predictive/Final%20Exam%20Slides.pdf)

---

### Week 1 - Overview

- [Notes](http://localhost:8888/notebooks/1%20Predictive/Overview%20Notes.ipynb)


```python

```

---

### Week 2 - Classification

- [Overview Notes](http://localhost:8888/notebooks/1%20Predictive/2%20-%20Classification%20(Trees%2C%20kNN%2C%20Logistic)/Classification%20Notes.ipynb)
- [Trees](http://localhost:8888/notebooks/1%20Predictive/2%20-%20Classification%20(Trees%2C%20kNN%2C%20Logistic)/(Code)%20Trees.ipynb)
- [Knn](http://localhost:8888/notebooks/1%20Predictive/2%20-%20Classification%20(Trees%2C%20kNN%2C%20Logistic)/(Code)%20Knn.ipynb)
- [Logistic](http://localhost:8888/notebooks/1%20Predictive/2%20-%20Classification%20(Trees%2C%20kNN%2C%20Logistic)/(Code)%20Logistic.ipynb)


```python

```


```python

```

### Week 6 - Ensemble

- [Canvas](https://canvas.umn.edu/courses/139500/pages/week-6-ensemble-methods)
- [Github Notes](https://github.com/SamMusch/Predictive-and-EDA/blob/master/Notes/Ensemble%20Notes.ipynb)


```python

```

---

### Weeks 8, 9, 10 - Neural Networks

- [Canvas](https://canvas.umn.edu/courses/139500/pages/week-8-10-deep-neural-network-cnn-and-rnn)
- [Notes](https://github.com/SamMusch/Predictive-and-EDA/blob/master/Notes/Neural%20Network.ipynb)



---

### Weeks 12, 13 - Recommend

- [Canvas](https://canvas.umn.edu/courses/139500/pages/week-12-13-recommender-system)
- [Notes](http://localhost:8888/notebooks/1%20Predictive/12%20-%20Recommend/Recommender%20Systems.ipynb)


