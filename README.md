# Olist E-Commerce: EDA & Customer Segmentation (K-Means)

An end-to-end analysis of the Brazilian **Olist** e-commerce data. The project has two parts:

1. **Exploratory Data Analysis (EDA)** of the merged order dataset: cleaning, statistics, outliers, skewness, and visualisations.
2. **Customer segmentation** with **K-Means**: orders are turned into customer-level features, and customers are grouped into 5 business-friendly segments.

---

## Project Structure

```
.
├── Olist_Complete_EDA.ipynb                  # Part 1: EDA and data checks
├── Olist_KMeans_Clustering_v2_fixed.ipynb    # Part 2: K-Means customer segmentation
├── olist_merged_dataset.csv                  # Input data (not included in the repo)
├── EDA_Plots/                                # Saved figures (created automatically)
├── customer_segments.csv                     # Output: every customer with its cluster/segment
└── cluster_profile.csv                       # Output: summary table per segment
```

---

## Part 1: Exploratory Data Analysis

**Notebook:** `Olist_Complete_EDA.ipynb`

This notebook is EDA and data preparation only. It does **not** encode, scale, split, or train any model.

| Step | What is done |
|------|--------------|
| Overview | Shape, head/tail/sample, column names, dtypes |
| Data quality | Missing values (median for numeric, mode for categorical), blank values, duplicate rows, duplicate IDs |
| Descriptive stats | Mean, median, mode, variance, standard deviation |
| Grouping | Orders, average price/freight and total sales by order status and by customer state |
| Statistical tests | Levene (variance homogeneity), Chi-Square (order status vs state), Cramér's V, D'Agostino-Pearson normality test |
| Distribution | Skewness and kurtosis before and after treatment |
| Outliers | IQR detection and IQR clipping for `price` and `freight_value` |
| Transformation | Yeo-Johnson, applied only to highly skewed continuous variables |
| Visuals | Histograms, boxplots, price vs freight scatter, pairplot, correlation heatmap, order-status count, top customer states |

All plots are saved to `EDA_Plots/`.

---

## Part 2: Customer Segmentation with K-Means

**Notebook:** `Olist_KMeans_Clustering_v2_fixed.ipynb`

### Pipeline

1. **Load & clean:** keep only *delivered* orders with valid dates and non-negative price/freight (113,425 → 110,189 rows).
2. **Customer-level features:** orders are aggregated first per order, then per `customer_unique_id` (93,350 customers).
3. **Feature selection:** 7 features (below) plus a correlation check.
4. **Skew handling:** `log1p` on spending and delivery days, then winsorising at the 1st/99th percentile.
5. **Scaling:** `StandardScaler`.
6. **Choosing K:** elbow, silhouette, Davies-Bouldin and Calinski-Harabasz for K = 2 to 8.
7. **Final model:** K-Means with **K = 5**, `n_init=10`, `random_state=42`.
8. **Profiling & visualisation:** cluster profile table, z-score heatmap, cluster sizes, 2-D and 3-D PCA views.

### Features Used

| Feature | Meaning |
|---------|---------|
| `Recency_Days` | Days since the customer's last purchase |
| `Is_Repeat` | 1 if the customer has more than one order |
| `Total_Spending` | Sum of item prices (log-transformed) |
| `Avg_Items_Per_Order` | Average number of items per order |
| `Freight_Share` | Freight as a share of spending plus freight |
| `Avg_Delivery_Days` | Average purchase-to-delivery time (log-transformed) |
| `Avg_Delivery_Delay` | Delivered date minus estimated date (negative = early) |

### Why K = 5?

| K | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|-----------|----------------|-------------------|
| 4 | **0.233** | 1.308 | 22,210 |
| **5** | 0.223 | **1.227** | **23,455** |
| 6 | 0.225 | 1.205 | 23,997 |

Silhouette peaks at K = 4, but K = 5 is almost identical and better on Davies-Bouldin and Calinski-Harabasz. K = 5 also separates a business-relevant **late-delivery** segment that K = 4 merges into other groups.

**Stability:** re-running with 4 other random seeds gives an Adjusted Rand Index of 0.981 to 0.999, so the clusters are essentially the same every time.

### Results: The 5 Segments

| Segment | Customers | Share | Avg. spending | Items/order | Freight share | Delivery days | Delay (days) |
|---------|-----------|-------|---------------|-------------|---------------|---------------|--------------|
| Fast-delivery, mid-value buyers | 40,649 | 43.5% | 182.76 | 1.00 | 0.13 | 8.9 | -14.3 |
| Low-value, freight-heavy | 25,350 | 27.2% | 35.36 | 1.01 | 0.35 | 10.8 | -13.0 |
| Late-delivery customers | 16,207 | 17.4% | 149.15 | 1.01 | 0.18 | 25.5 | +0.7 |
| Multi-item basket buyers | 8,343 | 8.9% | 209.63 | 2.43 | 0.23 | 11.1 | -13.0 |
| Repeat customers | 2,801 | 3.0% | 260.05 | 1.21 | 0.20 | 12.3 | -11.9 |

Segment names are assigned automatically by rules from the cluster profile, so they stay correct even if cluster IDs change between runs.

### Key Takeaways

- **Delivery performance is the strongest separator.** The late-delivery group waits about 25 days and is the only group delivered after the estimated date.
- **Low-value, freight-heavy customers** (27% of the base) spend little, and freight is about 35% of their total.
- **Only 3% of customers are repeat buyers**, but they spend the most on average.
- **Recency barely differs between segments** (about 220 to 245 days), so it contributes little to separating them.

### Visualisations

- Feature correlation heatmap
- Elbow / silhouette / Davies-Bouldin plots for choosing K
- Cluster profile heatmap (z-scores vs overall average)
- Customers per segment
- 2-D and 3-D PCA views of the segments (PCA is for visualisation only; clustering uses all 7 features)

---

## Getting Started

### Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`
- `jupyter` or VS Code with the Jupyter extension
- Optional, for the interactive 3-D plot: `plotly` and `nbformat`

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter plotly nbformat
```

### Run

1. Place `olist_merged_dataset.csv` in the project folder (the path is set by `DATA_PATH` in the notebooks).
2. Run `Olist_Complete_EDA.ipynb` from top to bottom.
3. Run `Olist_KMeans_Clustering_v2_fixed.ipynb` from top to bottom.

Outputs are written to `EDA_Plots/`, `customer_segments.csv` and `cluster_profile.csv`.

---

## Limitations

- Silhouette is about 0.22, which means moderate separation; the segments overlap, as the PCA plots show.
- The 3-D PCA view explains about 60% of the variance (25% + 20% + 15%), so some overlap in the plot is expected.
- The dataset covers delivered orders only; cancelled and unavailable orders are excluded.
- With only 3% repeat customers, loyalty-related insights are based on a small group.

## Possible Next Steps

- Try other algorithms (Gaussian Mixture, DBSCAN, hierarchical clustering) and compare.
- Add review scores and payment features to the customer profile.
- Turn each segment into an action plan (for example, delivery improvements for the late-delivery group).
- Build a small dashboard for exploring the segments interactively.

## Dataset

Based on the public Brazilian e-commerce dataset by **Olist**, merged into a single file (`olist_merged_dataset.csv`).
