## Both



### Quick Plots

```python
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()
```

```python
corr_matrix = df.corr()

attributes = ["col1", 'col2', 'col3']
scatter_matrix(df[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
```





### 1. Split

```python
# Initial splits
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
X_train = train_set.drop('col', axis=1)
X_test = test_set.drop('col', axis=1)
y_train = train_set['col']
y_test = test_set['col']

# Shuffling train
shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# Folded splits
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
  
tscv = TimeSeriesSplit(n_splits=3, test_size=.2)
```





### 2. Transform

```python
df['col'].get_dummies(prefix = 'key')
```



```python
# Scale = normalize = 0 to 1
scaler = MinMaxScaler()
pd.DataFrame(scaler.fit_transform(numeric_columns), columns = numeric_columns.columns)

# Standard = how far from mean
scaler = StandardScaler()
```

```python
imputer = SimpleImputer(strategy="median")
imputer.fit(df)
```

```python
# Movement size
bins = [-20, -10, 0, 10, 20]
trades['bins'] = pd.cut(trades['lag'], bins=bins)
threshold_size = trades['size'].quantile(.95)
trades['above_95'] = trades['size'] > threshold_size
```



#### Pipeline

```python
num_attribs = [['a', 'b']]
cat_attribs = [['c', 'd']]

# Numerical columns
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

# Categorical columns
cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

# Numerical + categorical
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

# Ready to split
df_prepared = full_pipeline.fit_transform(df)
```



### 3. Nested Cross Val

```python
# Opt model for each fold
grid_search = GridSearchCV(classifier, 
                    param_grid = param_grid, 
                    cv = inner_cv, 
                    scoring='xxxx')
grid_search.fit(X_train, y_train)
final_model = grid_search.best_estimator_

nested_score = cross_val_score(grid_search, X_train, y_train, cv=outer_cv)
each_score = np.sqrt(-nested_score)
```



#### Eval

[Classification curves](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

- Matrix
- Precision Recall
- ROC

[Regression curves](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

- [Learning curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)



```python
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)
```

```python
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
print("Prediction Accuracy: ",accuracy_score(y_test, grid.predict(X_test)))
```



### 4. Fit

```python
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```



#### Saving Model


```python
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)

# The following method is more efficient on big data, but can only pickle to the disk and not to a string:
from sklearn.externals import joblib
joblib.dump(clf, 'kNN.pkl')

# Later you can load back the pickled model (possibly in another Python process) with:
clf = joblib.load('kNN.pkl' )
```





## Classification

### Gridsearch

[KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) 
[Trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 
[Logistic](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)


```python
# Knn
param_grid = dict(n_neighbors = list(range(1,31)), 
                  weights = ["uniform", "distance"])
knn = KNeighborsClassifier()

# Tree
param_grid = dict(criterion = ["gini", "entropy"], 
                  max_depth = range(2,10),
                  min_samples_leaf = range(2,8),
                  min_impurity_decrease = [0,1e-8,1e-7,1e-6,1e-5,1e-4])
grid_tree_clf = tree.DecisionTreeClassifier(random_state=45)


# Logistic
param_grid = dict(penalty = ['l1', 'l2'], 
                  C = range(1,10))
```



### Report


```python
from sklearn.metrics import classification_report
from sklearn import metrics

target_names = ['malignant', 'benign']
y_true = y_test
y_pred = y_pred

print(target_names)
print("Accuracy: {0:.2%}".format(accuracy_score(y_true, y_pred)))
print("Precision: {0:.2%}".format(metrics.precision_score(y_true, y_pred)))
print("Recall: {0:.2%}".format(metrics.recall_score(y_true, y_pred)))
print("F1: {0:.2%}".format(metrics.f1_score(y_true, y_pred)))
print('-------------------------------------')
print(classification_report(y_true, y_pred))
```





## Regression

```python

```



# Techniques



### Knn

```python
# KNN regression model
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))
```



### Decision trees

```python
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```



# Extra

#### 2 features


```python
## Visualization of the data set
from mpl_toolkits.mplot3d import Axes3D

X = cancer.data[:, :2] # we only take the first two features.
Y = cancer.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
```



#### 3 features


```python
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
from sklearn.decomposition import PCA

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
    cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
```





### Visualize Decision Boundary


```python
# Parameters
featureA, featureB = 0, 3     # select two variables to visualize
plot_colors = "bry"
plot_step = 0.02              # step size in the mesh
X = iris.data[:, [featureA, featureB] ]   # We only take the two corresponding features
y = iris.target

n_classes = len(set(y))
clf = clf.fit(X, y)


# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.xlabel(iris.feature_names[featureA])
plt.ylabel(iris.feature_names[featureB])
plt.axis("tight")

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                cmap=plt.cm.Paired)

plt.axis("tight")
plt.suptitle("Decision surface of a logistic regression classifier")
plt.legend()
plt.show()
```



# Details

## Cross Validate

**Hands On 2 - End to end ML project [Code](https://github.com/ageron/handson-ml/blob/73bbb65905e04f98379185b09f0e2d0883ab9955/02_end_to_end_machine_learning_project.ipynb)  
Hands On 3 - Classification [Code](https://github.com/ageron/handson-ml/blob/73bbb65905e04f98379185b09f0e2d0883ab9955/03_classification.ipynb)  
DS Business 7 - What is a good model?  
DS Business 8 - Visual model performance**

Nested CV

- use when hyperparameters also need to be optimized
- removes overfit "leak" from evaluating on train set
- estimates the generalization error of the underlying model & hypers


- inner loop - fit model to each training set, then select hypers over validation set
- outer loop - est generalization error by averaging test set scores over several dataset splits


For 1 model - which hyperparameters? N-fold over train subset + validation  
Which model tends to perform the best? Nested cross validation



**N-fold stratified cross validation**

We want to make sure the **mean=high** and that the **stdev=low**  
The target variable should be consistent in each group

<img src=https://i.imgur.com/2klk06l.png width="350" height="300" align="left">





**Nested cross validation - score can be accuracy or others**

Inner = find best hyperparameters  
Outer = evaluate  

<img src=https://i.imgur.com/Fbm3RpK.png width="350" height="300" align="left">



## Decision Tree

http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier


**Parameters**  
Tree structure

**Hyperparameters**  
Max depth, min sample leaf


The advantages include

- Dont need data normalisation, dummy variables (Note however that this module does not support missing values.)
- Able to handle both numerical and categorical data.
- Able to handle multi-output problems.
- Uses a white box model
- Possible to validate a model using statistical tests
- Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.



**disadvantages of decision trees include:**

- Overfitting   
  Pruning (not currently supported)  
  Minimum number of samples required at a leaf node  
  Maximum depth of the tree  
  Laplace to make sure groups aren't tiny  
  [Ensemble](http://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html)

---

- Small var in data could have big impact on tree  
  Just use ensemble


- Can't guarentee global optimum  
  Use ensemble, features and samples are randomly sampled with replacement


- Not good with problems like XOR, parity or multiplexer problems.


- Decision tree learners create biased trees if some classes dominate  
  Balance the dataset prior to fitting with the decision tree.


---

## Logistic Regression

**Functions**

`Logit` for only 2 groups  
`SoftMax Regression` has multiple groups - assign new into the highest prob group  
`Ordered Logit` has ordered multiple groups


Parametric model, meaning number of parameters is set before fitting

Step 1 - Building the logit plot.  
For each observation in the raw data, we calculate the sum of the coefficients for all variables. This is f(x).

Step 2 - (Looking at the logit plot)  
The x-axis location on the logit plot is the f(x) we calculated above for each observation.

Step 3 - For each observation, we then assign the probability of fitting into the top or bottom group according to the logit curve.

Step 4 - We use the probabilities that we found in the logit model as a way to best fit the line that separates our groups apart to minimize the error in our sample. This is how we determine our original slope and intercept.

Step 5 - We assign L1 as the maximum f(x) that we are willing to allow.  (This is lambda)
More complex = higher f(x) = larger diamond

Step 6 - We 'underfit' the model until it connects with the blue diamond (ie max complexity we allow). This new model will provide us with a new slope and intercept that will generalize better.



Hyperparameters

- C: strength of regularization
- multi_class: how many groups we are splitting into
  - binomial = 'ovr'  = one-vs-rest (OvR)
  - multi_class = 'multinomial' = Softmax = cross-entropy loss




---

### Technique 1 - Linear discriminant

Draw linear line between the two groups  
Parameters are the `y-intercept` and the `slope`  
Set up to minimize error

<img src=https://i.imgur.com/OzTuY13.png width="300" height="240" align="left">  













### Technique 2 - Logit

Draw log line between the two groups  

<font size="4"> $probability(x) \: = \: \frac{1}{e^(-1 \: * \: regression \: model)}$ </font>

<img src=https://i.imgur.com/U9xDqXS.png width="300" height="240" align="left">



### Technique 3 - Ordered Logit

```python
import mord
from mord.datasets.base import load_housing
from sklearn import linear_model, metrics, preprocessing

data = load_housing()
features = data.data

le = preprocessing.LabelEncoder()
le.fit(data.target)
data.target = le.transform(data.target)

features.loc[features.Infl == 'Low', 'Infl'] = 1
features.loc[features.Infl == 'Medium', 'Infl'] = 2
features.loc[features.Infl == 'High', 'Infl'] = 3

features.loc[features.Cont == 'Low', 'Cont'] = 1
features.loc[features.Cont == 'Medium', 'Cont'] = 2
features.loc[features.Cont == 'High', 'Cont'] = 3

le = preprocessing.LabelEncoder()
le.fit(features.loc[:,'Type'])
features.loc[:,'type_encoded'] = le.transform(features.loc[:,'Type'])

X, y = features.loc[:,('Infl', 'Cont', 'type_encoded')], data.target

clf1 = linear_model.LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial')
clf1.fit(X, y)

print('Mean Absolute Error of LogisticRegression: %s' %
      metrics.mean_absolute_error(clf1.predict(X), y))

clf2 = mord.LogisticAT(alpha=1.)
clf2.fit(X, y)
print('Mean Absolute Error of LogisticAT %s' %
      metrics.mean_absolute_error(clf2.predict(X), y))

clf3 = mord.LogisticIT(alpha=1.)
clf3.fit(X, y)
print('Mean Absolute Error of LogisticIT %s' %
      metrics.mean_absolute_error(clf3.predict(X), y))

clf4 = mord.LogisticSE(alpha=1.)
clf4.fit(X, y)
print('Mean Absolute Error of LogisticSE %s' %
      metrics.mean_absolute_error(clf4.predict(X), y))
```

