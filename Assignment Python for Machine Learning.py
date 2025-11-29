import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. LOAD & PREPARE CATEGORICAL DATA
# -------------------------------------------------
data_path = r"C:\Users\kanak\Downloads\breast+cancer\breast-cancer.data"

# The file is comma-separated and has no header row
df = pd.read_csv(data_path, header=None)

# Replace '?' with NaN and drop missing rows
df = df.replace("?", np.nan)
df = df.dropna()

# Drop the class label column (column 0) so clustering is unsupervised
X = df.iloc[:, 1:].copy()

# Convert ALL categorical columns into one-hot encoded numeric columns
X_numeric = pd.get_dummies(X)

# 🔹 Make sure all feature names are strings
X_numeric.columns = X_numeric.columns.astype(str)

# 2. SCALE FEATURES
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric.values)

# 3. COMPUTE SSE FOR DIFFERENT k
# -------------------------------------------------
sse = []
k_values = range(1, 11)  # try k = 1 to 10; adjust if needed

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    # inertia_ is the within-cluster sum of squared distances (SSE)
    sse.append(kmeans.inertia_)

# Print SSE values for each k
print("\nSum of Squared Errors (SSE) for each k:")
for k, val in zip(k_values, sse):
    print(f"k = {k}: SSE = {val}")

# 4. ELBOW PLOT TO CHOOSE OPTIMAL k
# -------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(k_values, sse, marker="o")
plt.xticks(k_values)
plt.xlabel("Number of clusters k")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal k (Breast Cancer Data)")
plt.tight_layout()
plt.show()

# from your elbow plot we chose:
k_opt = 3
print(
    "Using the elbow method I was able to determine that K 3 was the optimal number."
    "This came about through trial and error, the elbow plot didn't have a distinct and sharp curve/ change in slope."
    "After testing several K values, 3 appeared to be the best fit based on the final cluster plot.")
# 5. FIT K-MEANS WITH OPTIMAL k AND GET CLUSTERS
# -------------------------------------------------
kmeans_opt = KMeans(
    n_clusters=k_opt,
    random_state=42,
    n_init=10
)
cluster_labels = kmeans_opt.fit_predict(X_scaled)
centroids = kmeans_opt.cluster_centers_

# 6. REDUCE TO 2D WITH PCA FOR PLOTTING
# -------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
centroids_2d = pca.transform(centroids)

# 7. PLOT CLUSTERS + CENTROIDS
# -------------------------------------------------
plt.figure(figsize=(6, 5))

# scatter plot of data points colored by cluster
plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=cluster_labels,
    alpha=0.6,
    s=30,
    edgecolor="k"
)

# plot centroids
plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    s=200,
    marker="X",
    edgecolor="k"
)
print(
    "The PCA cluster plot shows how the data groups into three distinct clusters when k=3 "
    "is used. Each point represents a breast cancer case projected into two principal "
    "components, and the large 'X' markers indicate the centroids of each cluster. "
    "Two clusters on the right side appear tight and compact, indicating more similarity "
    "among those cases, while the cluster on the left is more spread out due to higher "
    "internal variability in the underlying categorical features. Overall, the cluster "
    "plot visually confirms that the dataset naturally separates into three meaningful "
    "groups when viewed with PCA."
)

plt.title(f"K-Means Clusters (k = {k_opt}) on Breast Cancer Data (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
