# ============================================================
# OLIST E-COMMERCE DATASET - COMPLETE EDA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.stats import chi2_contingency


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("olist_merged_dataset.csv")

print("\n========== DATASET LOADED ==========")
print("Shape:", df.shape)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("EDA_Plots", exist_ok=True)


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================

print("\n========== HEAD ==========")
print(df.head())

print("\n========== TAIL ==========")
print(df.tail())

print("\n========== SAMPLE ==========")
print(df.sample(5, random_state=42))

print("\n========== SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())


# ============================================================
# 3. DATA INFORMATION
# ============================================================

print("\n========== DATA INFORMATION ==========")

df.info()

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    include="object"
).columns.tolist()

print("\nNumerical Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage": df.isnull().mean() * 100
})

missing = missing[
    missing["Missing Values"] > 0
].sort_values(
    "Missing Values",
    ascending=False
)

print(missing)


# ============================================================
# BLANK VALUES
# ============================================================

print("\n========== BLANK VALUES ==========")

for col in categorical_cols:

    blank = df[col].astype(str).str.strip().eq("").sum()

    if blank > 0:
        print(col, ":", blank)


# ============================================================
# 5. DUPLICATES
# ============================================================

print("\n========== DUPLICATES ==========")

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

for col in [
    "order_id",
    "customer_id",
    "product_id",
    "seller_id"
]:

    if col in df.columns:

        print(
            f"Duplicate {col}:",
            df[col].duplicated().sum()
        )


# ============================================================
# 6. UNIQUE VALUES
# ============================================================

print("\n========== UNIQUE VALUES ==========")

for col in df.columns:

    print(
        col,
        "->",
        df[col].nunique()
    )


# ============================================================
# 7. VALUE COUNTS
# ============================================================

print("\n========== VALUE COUNTS ==========")

for col in [
    "order_status",
    "customer_state",
    "customer_city"
]:

    if col in df.columns:

        print(f"\n--- {col} ---")

        print(
            df[col]
            .value_counts()
            .head(15)
        )


# ============================================================
# 8. DESCRIPTIVE STATISTICS
# ============================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(
    df[numeric_cols].describe()
)

print("\n========== MEDIAN ==========")

print(
    df[numeric_cols].median()
)

print("\n========== MODE ==========")

print(
    df[numeric_cols]
    .mode()
    .iloc[0]
)


# ============================================================
# 9. GROUPING ANALYSIS
# ============================================================

print("\n========== GROUPING ANALYSIS ==========")

group_status = df.groupby(
    "order_status"
).agg(

    Orders=("order_id", "nunique"),

    Average_Price=("price", "mean"),

    Average_Freight=("freight_value", "mean"),

    Total_Sales=("price", "sum")
)

print(group_status)


# ============================================================
# STATE ANALYSIS
# ============================================================

print("\n========== STATE ANALYSIS ==========")

state_analysis = df.groupby(
    "customer_state"
).agg(

    Orders=("order_id", "nunique"),

    Customers=(
        "customer_unique_id",
        "nunique"
    ),

    Average_Price=("price", "mean"),

    Total_Sales=("price", "sum")
)

print(
    state_analysis
    .sort_values(
        "Total_Sales",
        ascending=False
    )
    .head(15)
)


# ============================================================
# 10. COMPARISON OF MEANS
# ============================================================

print("\n========== COMPARISON OF MEANS ==========")

mean_comparison = df.groupby(
    "order_status"
)[
    [
        "price",
        "freight_value"
    ]
].mean()

print(mean_comparison)


# ============================================================
# 11. VARIANCE CHECK
# ============================================================

print("\n========== VARIANCE CHECK ==========")

variance = df[numeric_cols].var()

print(variance)

print("\nStandard Deviation:")

print(
    df[numeric_cols].std()
)


# ============================================================
# 12. INDEPENDENCE CHECK
# ============================================================

print("\n========== INDEPENDENCE CHECK ==========")

table = pd.crosstab(
    df["order_status"],
    df["customer_state"]
)

chi2, p_value, dof, expected = chi2_contingency(
    table
)

print("Chi-Square:", chi2)

print("P-value:", p_value)

if p_value < 0.05:

    print(
        "Result: Variables are statistically associated."
    )

else:

    print(
        "Result: No significant association detected."
    )


# ============================================================
# 13. UNIVARIATE ANALYSIS
# ============================================================

print("\n========== UNIVARIATE ANALYSIS ==========")


# ------------------------------------------------------------
# PRICE HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["price"].dropna(),
    kde=True
)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "EDA_Plots/price_histogram.png"
)

plt.close()


# ------------------------------------------------------------
# FREIGHT HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["freight_value"].dropna(),
    kde=True
)

plt.title("Freight Value Distribution")
plt.xlabel("Freight Value")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "EDA_Plots/freight_histogram.png"
)

plt.close()


# ------------------------------------------------------------
# PRICE BOXPLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["price"]
)

plt.title("Price Boxplot")

plt.tight_layout()

plt.savefig(
    "EDA_Plots/price_boxplot.png"
)

plt.close()


print("Univariate plots saved successfully.")


# ============================================================
# 14. BIVARIATE ANALYSIS
# ============================================================

print("\n========== BIVARIATE ANALYSIS ==========")

sample = df.sample(
    min(10000, len(df)),
    random_state=42
)


plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=sample,
    x="price",
    y="freight_value",
    alpha=0.5
)

plt.title("Price vs Freight Value")
plt.xlabel("Price")
plt.ylabel("Freight Value")

plt.tight_layout()

plt.savefig(
    "EDA_Plots/price_vs_freight.png"
)

plt.close()


print("Bivariate plot saved successfully.")


# ============================================================
# 15. MULTIVARIATE ANALYSIS
# ============================================================

print("\n========== MULTIVARIATE ANALYSIS ==========")

pair_data = df[
    [
        "price",
        "freight_value",
        "order_item_id",
        "order_status"
    ]
].dropna()

pair_data = pair_data.sample(
    min(3000, len(pair_data)),
    random_state=42
)


pair_plot = sns.pairplot(
    pair_data,
    hue="order_status"
)

pair_plot.savefig(
    "EDA_Plots/pairplot.png"
)

plt.close("all")


print("Multivariate plot saved successfully.")


# ============================================================
# 16. SKEWNESS
# ============================================================

print("\n========== SKEWNESS ==========")

skewness = df[numeric_cols].skew()

print(skewness)

print("\nInterpretation:")

print("0 approximately = symmetric")
print("Positive = right skewed")
print("Negative = left skewed")


# ============================================================
# 17. KURTOSIS
# ============================================================

print("\n========== KURTOSIS ==========")

kurtosis = df[numeric_cols].kurt()

print(kurtosis)

print("\nInterpretation:")

print("Positive = heavier tails")
print("Negative = lighter tails")
print("Approximately 0 = normal-like tails")


# ============================================================
# 18. OUTLIER DETECTION - IQR
# ============================================================

print("\n========== OUTLIER DETECTION ==========")

outlier_table = []

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    count = (
        (df[col] < lower) |
        (df[col] > upper)
    ).sum()

    percentage = (
        count /
        df[col].notna().sum()
    ) * 100

    outlier_table.append([
        col,
        Q1,
        Q3,
        IQR,
        lower,
        upper,
        count,
        percentage
    ])


outlier_df = pd.DataFrame(
    outlier_table,
    columns=[
        "Column",
        "Q1",
        "Q3",
        "IQR",
        "Lower Bound",
        "Upper Bound",
        "Outlier Count",
        "Outlier %"
    ]
)

print(outlier_df)


# ============================================================
# 19. NORMALIZATION CHECK ONLY
# ============================================================

print("\n========== NORMALIZATION CHECK ==========")

scale_check = pd.DataFrame({

    "Minimum":
        df[numeric_cols].min(),

    "Maximum":
        df[numeric_cols].max(),

    "Range":
        (
            df[numeric_cols].max()
            -
            df[numeric_cols].min()
        ),

    "Mean":
        df[numeric_cols].mean(),

    "Std":
        df[numeric_cols].std()
})

print(scale_check)

print("\nIMPORTANT:")

print("Normalization is ONLY checked.")

print("Normalization is NOT applied.")


# ============================================================
# 20. CORRELATION ANALYSIS
# ============================================================

print("\n========== CORRELATION ==========")

corr = df[numeric_cols].corr()

print(corr)


plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "EDA_Plots/correlation_heatmap.png"
)

plt.close()


# ============================================================
# 21. ORDER STATUS VISUALIZATION
# ============================================================

print("\n========== ORDER STATUS VISUALIZATION ==========")

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="order_status",
    order=df[
        "order_status"
    ].value_counts().index
)

plt.title("Order Status Distribution")

plt.xlabel("Order Status")

plt.ylabel("Number of Records")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/order_status_countplot.png"
)

plt.close()


# ============================================================
# 22. TOP CUSTOMER STATES
# ============================================================

print("\n========== TOP CUSTOMER STATES ==========")

top_states = (
    df["customer_state"]
    .value_counts()
    .head(15)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_states.values,
    y=top_states.index
)

plt.title("Top 15 Customer States")

plt.xlabel("Number of Records")

plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    "EDA_Plots/top_states.png"
)

plt.close()


# ============================================================
# 23. PATTERNS & RELATIONSHIPS
# ============================================================

print("\n========== PATTERNS & RELATIONSHIPS ==========")

print(
    "\nMost common order status:",
    df["order_status"]
    .value_counts()
    .idxmax()
)

print(
    "Most common customer state:",
    df["customer_state"]
    .value_counts()
    .idxmax()
)

print(
    "Average price:",
    round(
        df["price"].mean(),
        2
    )
)

print(
    "Maximum price:",
    round(
        df["price"].max(),
        2
    )
)

print(
    "Average freight value:",
    round(
        df["freight_value"].mean(),
        2
    )
)

print(
    "Maximum freight value:",
    round(
        df["freight_value"].max(),
        2
    )
)


# ============================================================
# 24. FINAL EDA FINDINGS
# ============================================================

print(
    "\n" + "=" * 60
)

print("FINAL EDA FINDINGS")

print("=" * 60)


print("""
1. The dataset contains customer, order, product,
   seller, price and shipping information.

2. Missing values are present in some order and
   delivery-related columns.

3. Duplicate records were checked.

4. Unique values and category frequencies were analyzed.

5. Descriptive statistics were calculated for
   numerical variables.

6. Grouping analysis was performed using order status
   and customer state.

7. Means and variances were compared between groups.

8. Chi-Square test was used to check the relationship
   between order status and customer state.

9. Univariate, bivariate and multivariate analysis
   were performed.

10. Skewness and kurtosis were checked to understand
    distributions.

11. IQR method was used for outlier detection.

12. Numerical variables have different ranges,
    therefore normalization was CHECKED but NOT applied.

13. Correlation analysis was performed to identify
    relationships between numerical variables.

14. Important patterns and relationships were identified
    through statistical analysis and visualization.
""")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nEDA COMPLETED SUCCESSFULLY!")

print(
    "\nAll graphs are saved inside:"
)

print(
    os.path.abspath("EDA_Plots")
)