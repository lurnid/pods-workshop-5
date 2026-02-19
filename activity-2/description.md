
# W5 Workshop activity 2: Implementing clustering (Python)

**In this workshop activity, you'll practise building a model and changing its clusters and features.**

In this activity, you'll be using the well explored [Wine Dataset from scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset).

There are 13 features and a 'target' column representing the wine class. Please remember that in the real world you will not have a target column, as the goal of clustering is to find out how many distinct classes there are and then analyse the properties of each, in order to derive some value.

### 1. Load the libraries and dataset

Load the following libraries:

- pandas
- numpy
- matplotlib.pyplot
- sklearn.cluster (KMeans)
- sklearn.datasets (load_wine)

Load the wine dataset using **load_wine()** from sklearn.datasets. Create a DataFrame and a variable called **features** which contains only the feature columns (excluding the target).

```python
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
features = df.drop('target', axis=1)
```

### 2. Familiarise yourself

Familiarise yourself with the dataset. Seek to understand:

- the types of data.
- distributions of features.
- obvious patterns.
- errors, nulls, outliers.

### 3. Build your first model

After exploring the dataset, you can apply your clustering algorithm. Start by using the raw data as-is, though down the line you may make some changes.

1. Create a KMeans model using `KMeans(n_clusters=2, random_state=42)`.
2. Fit the model and get the cluster labels using `.fit_predict(features)`.
3. Add the labels as a new column to your DataFrame.
4. Compute the mean value of each feature per cluster label using `.groupby()`. What are the key differences between the groups?
5. Plot the clustered data: If you have time, try to display the count of actual label vs. predicted label for each group using `pd.crosstab()`.

### 4. Changing clusters and features

1. Repeat step 3 but using three clusters. Compare this to the original classes as you did above.
2. Repeat it again but this time using only the below five features:
    1. alcohol
    2. alcalinity_of_ash
    3. magnesium
    4. color_intensity
    5. proline
3. How does this model with five features compare to the model with all? Why do you think this is the case? What would you do to resolve the issue?
4. Re-run once more with only one feature (you should see which).
5. Examine the output of this.

Note: For each model, create new columns in the DataFrame to store the results, e.g., `kmeans2` and `kmeans3`.

### 5. Scaling

1. Your model was being affected by the large scale of some of our features. Apply scaling to each feature using `StandardScaler` from sklearn.preprocessing.
2. Repeat step 3, fitting KMeans to the scaled dataset.
3. What difference do you see?

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```

### 6. Choosing the optimal number of clusters

Run the below code and examine the output. What does it show you?

```python
# Initialize a list to store WCSS values
wcss_values = []

# Loop through a range of cluster sizes
for k in range(1, 11):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)

    # Get the within-cluster sum of squares (inertia)
    wcss_values.append(kmeans.inertia_)

# Plot the WCSS values
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss_values, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method: WCSS vs Number of Clusters')
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.show()
```

### Extension activity: Interpret the silhouette score

Examine the output of the below code which computes the silhouette coefficient and visualises it. Explore the metric and interpret the outputs.

```python
from sklearn.metrics import silhouette_score

# Initialize a list to store silhouette scores
silhouette_scores = []

# Loop through a range of cluster sizes (silhouette requires at least 2 clusters)
for k in range(2, 11):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # Compute silhouette score
    score = silhouette_score(scaled_features, cluster_labels)
    silhouette_scores.append(score)

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.title('Average Silhouette Score vs Number of Clusters (k)')
plt.xticks(range(2, 11))
plt.grid(True, alpha=0.3)
plt.show()
```
