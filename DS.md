

## Similarity

<img src=https://i.imgur.com/vVFUNGz.png width="400" height="240" align="left">



Euclidean(A,B) = $\sqrt{((a_1-b_1)^2+ \ldots +(a_k-b_k)^2)}$

Manhattan(A,B) = $|a_1-b_1| + \ldots + |a_k-b_k|$

Max-coordinate(A,B) = $max_i |a_i-b_i|$

- Chebyshev: only uses the most dissimilar variable



**Measuring mixed data - Gower Distance** 
For each variable type: Gower selects a particular distance metric, then scale it to fall between 0 and 1.
Then, calculates a linear combo to create the final distance matrix.

* `interval`: Manhattan distance
* `ordinal`: variable is first ranked, then Manhattan distance is used with a special adjustment for ties
* `nominal`: variables of k categories are first converted into k binary columns and then the [Dice coefficient](http://stats.stackexchange.com/a/55802/21654) is used

---

## Distributions

### 1d

Boxplot 5 number summary
min, Q1, med, Q3, max

Quantile plot for dist of 1 variable, normalized data on a scale of 0 to 1
<img src="https://i.imgur.com/ALHW7ej.png" width="500" />



### 2d

Quantile quantile plot for dist of 2 variables
line shows point where the two match up
dots rep tendency to differ from match
<img src="https://i.imgur.com/0KMGiD1.png" width="500" />



# Unsupervised Learning

<img src="https://www.researchgate.net/profile/Muhammad-Usama-36/publication/319952798/figure/fig1/AS:540941791567873@1505981981481/Taxonomy-of-Unsupervised-Learning-Techniques.png" style="zoom:50%;" />



## Data Preparation

### Missing Values

1. Completely at random (collection system issue?)
2. At random (people not entering their info)
3. Not at random (happening for a reason)



### Noise

1. Smooth
   1. Separate into bins, replace with bin **mean** or with nearest bin **boundary**
   2. Regression line
   3. Distribution line
2. Discretization
   1. Helpful for concept hierarchies, think age 67 vs 68
3. Normalize



### Reduction

1. Dimensionality reduction = reduce number of variables
   1. Principal components analysis combines columns to create more meaningful ones, removes the old ones
   2. Attribute subset selection removes unneeded columns, uses some stats test to determine which
2. Numerosity reduction = replace data with smaller form
   1. Parametric (regression)
   2. Nonparametric (clustering, sampling)
3. Data compression = reconstruct dataset
   1. Discrete wavelet transformation transforms a column into wavelet coefficients, and then drops the rows that arent significant



## Anomoly detection

Outlier types:

- Global: outlier relative to all points
- Contextual (75 degree day in Canada winter)
  - Context: Date, location
  - Behavioral: Temperature
- Collective: group of points together form an outlier



Possibilities:

- Matches an existing anomaly
- Multiple anomalies related to each other
- Something happening more/less often than before
- Different distribution than before



**Subset scanning**: how anomalous is the subset?

- WSARE & chisq - is the rule independent of time?
- Bayesian Network - does the row fall into a "normal" category?
- Build a predictive model, select points that don't fit



### Statistical

**Parameter** - assumes a distribution a priori

Univariate

- Maximum likelihood (ie based on our observed data, what is the most likely underlying distribution? Both mean and stdev)
- Grubb's test - uses Z score

Multivariate

- Mahalanobis
- Chi-squared
- Expectation maximization algo (normal distribution for each variable)

**Non-parameter** - does not assume a distribution a priori

- Histogram ([Example](https://i.imgur.com/OHdloVX.png))
  - Outlier score = $\frac{1}{bin \%}$
  - For 7.5 --> $\frac{1}{.2 \%}$



---

### Proximity

**Distance based** - how far from your neighborhood?

**Density based** - density of you & your neighbors

---

### Cluster

- Does it belong to a cluster? If **no** --> outlier
- Is it far away from its cluster? If **yes** --> outlier
- Is the cluster small or sparse? if **yes** --> outlier

**DBSCAN**





## Clustering

<img src=https://i.imgur.com/FOolcRz.png width="300" height="240" align="left">



### Partitioning

K-means

- High influence of outliers
  - Could use k-medioids instead
- Only works for continuous
  - K-modes (Hamming Distance)
  - K-prototype: kmeans + kmode (Gower Distance)
- No hierarchy provided
  - Use hierarchy cluster and then partition cluster
- Bias towards circles
  - Use dbscan instead



### Hierarchical

1. Agglomerative - opposite of divisive
   1. AGNES (AGglomerative NESting)
2. Divisive - start with 1 cluster, split apart
   1. DIANA (DIvisive ANAlysis)

<img src=https://i.imgur.com/3i4Ma0H.png width="350" height="200" align="left">



### Density

**DBscan** = continue growing cluster as long as we are meeting some threshold (min data points)

1. Epsilon = for each data point, radius of region if its the mean point 
2. Density of neighborhood = number of data points in the region  
3. MinPts = threshold to be considered dense
4. Core point = if the data point's region has MinPts
5. Direct density reachable = if a point is within the core point's region
6. Density reachable = direct density reachable from a point thats direct density reachable to the core point
7. Density connected = direct density reachable from a point thats direct density reachable to the neighborhood of a core point
8. Density based cluster = group of density connected points

**Mixture model** = start with data, identify true underlying distribution



### Measuring Performance

1. **Silhouette** = max homo within, max hetero between
   1. Coef between -1 and 1
   2. $Silhouette \: coef = \frac{b - a}{Max \: b - max \: a}$
   3. a = Avg distance from all points in *its own* cluster
   4. b = Avg distance from all points in *nearest* cluster

2. Gap statistic = measures improvement for each addition cluster. Visualized with the **elbow plot**
   1. x = Number of clusters
   2. y = SSE (each error = point - cluster mean)







