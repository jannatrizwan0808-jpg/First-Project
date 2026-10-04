# ============================================================
# OLIST E-COMMERCE DATASET - COMPLETE EDA
# EDA ONLY
# NO ENCODING
# NO SCALING
# NO TRAIN/TEST SPLIT
# NO MACHINE LEARNING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from scipy.stats import (
    chi2_contingency,
    levene,
    normaltest
)

from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("olist_merged_dataset.csv")

print("\n" + "=" * 70)
print("DATASET LOADED")
print("=" * 70)

print("Shape:", df.shape)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("EDA_Plots", exist_ok=True)


# ============================================================
# 3. DATASET OVERVIEW
# ============================================================

print("\n========== HEAD ==========")
print(df.head())

print("\n========== TAIL ==========")
print(df.tail())

print("\n========== SAMPLE ==========")
print(
    df.sample(
        min(5, len(df)),
        random_state=42
    )
)

print("\n========== SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())


# ============================================================
# 4. DATA INFORMATION
# ============================================================

print("\n========== DATA INFORMATION ==========")

df.info()

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)


# ============================================================
# 5. MISSING VALUES
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

if missing.empty:
    print("No missing values found.")


# ============================================================
# 6. MISSING VALUE TREATMENT
# ============================================================

print("\n========== MISSING VALUE TREATMENT ==========")

# Numerical columns -> Median
for col in numeric_cols:

    if df[col].isnull().sum() > 0:

        median_value = df[col].median()

        df[col] = df[col].fillna(
            median_value
        )

        print(
            col,
            "-> filled with median:",
            median_value
        )


# Categorical columns -> Mode
for col in categorical_cols:

    if df[col].isnull().sum() > 0:

        mode_value = df[col].mode()

        if len(mode_value) > 0:

            df[col] = df[col].fillna(
                mode_value.iloc[0]
            )

            print(
                col,
                "-> filled with mode:",
                mode_value.iloc[0]
            )


print(
    "Remaining missing values:",
    df.isnull().sum().sum()
)


# ============================================================
# 7. BLANK VALUES
# ============================================================

print("\n========== BLANK VALUES ==========")

blank_found = False

for col in categorical_cols:

    blank = (
        df[col]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )

    if blank > 0:

        blank_found = True

        print(
            col,
            ":",
            blank
        )

if not blank_found:

    print("No blank values found.")


# ============================================================
# 8. DUPLICATES
# ============================================================

print("\n========== DUPLICATES ==========")

duplicate_rows = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicate_rows
)

if duplicate_rows > 0:

    df = df.drop_duplicates()

    print(
        "Duplicate rows removed."
    )

else:

    print(
        "No complete duplicate rows found."
    )


# ============================================================
# 9. DUPLICATE ID CHECK
# ============================================================

print("\n========== DUPLICATE ID CHECK ==========")

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
# 10. UNIQUE VALUES
# ============================================================

print("\n========== UNIQUE VALUES ==========")

for col in df.columns:

    print(
        col,
        "->",
        df[col].nunique()
    )


# ============================================================
# 11. VALUE COUNTS
# ============================================================

print("\n========== VALUE COUNTS ==========")

for col in [
    "order_status",
    "customer_state",
    "customer_city"
]:

    if col in df.columns:

        print(
            f"\n--- {col} ---"
        )

        print(
            df[col]
            .value_counts()
            .head(15)
        )


# ============================================================
# 12. DESCRIPTIVE STATISTICS
# ============================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(
    df[numeric_cols].describe()
)


# ============================================================
# 13. MEDIAN
# ============================================================

print("\n========== MEDIAN ==========")

print(
    df[numeric_cols].median()
)


# ============================================================
# 14. MODE
# ============================================================

print("\n========== MODE ==========")

print(
    df[numeric_cols]
    .mode()
    .iloc[0]
)


# ============================================================
# 15. GROUPING ANALYSIS
# ============================================================

print("\n========== GROUPING ANALYSIS ==========")

group_status = df.groupby(
    "order_status"
).agg(

    Orders=(
        "order_id",
        "nunique"
    ),

    Average_Price=(
        "price",
        "mean"
    ),

    Average_Freight=(
        "freight_value",
        "mean"
    ),

    Total_Sales=(
        "price",
        "sum"
    )
)

print(group_status)


# ============================================================
# 16. STATE ANALYSIS
# ============================================================

print("\n========== STATE ANALYSIS ==========")

state_analysis = df.groupby(
    "customer_state"
).agg(

    Orders=(
        "order_id",
        "nunique"
    ),

    Customers=(
        "customer_unique_id",
        "nunique"
    ),

    Average_Price=(
        "price",
        "mean"
    ),

    Total_Sales=(
        "price",
        "sum"
    )
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
# 17. COMPARISON OF MEANS
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
# 18. VARIANCE
# ============================================================

print("\n========== VARIANCE CHECK ==========")

variance = df[numeric_cols].var()

print(variance)


# ============================================================
# 19. STANDARD DEVIATION
# ============================================================

print("\n========== STANDARD DEVIATION ==========")

std = df[numeric_cols].std()

print(std)


# ============================================================
# 20. LEVENE VARIANCE TEST
# ============================================================

print("\n========== LEVENE VARIANCE TEST ==========")

price_groups = []

for status, group in df.groupby(
    "order_status"
):

    values = (
        group["price"]
        .dropna()
    )

    if len(values) > 1:

        price_groups.append(
            values
        )


if len(price_groups) >= 2:

    levene_stat, levene_p = levene(
        *price_groups,
        center="median"
    )

    print(
        "Levene Statistic:",
        levene_stat
    )

    print(
        "P-value:",
        levene_p
    )

    if levene_p < 0.05:

        print(
            "Result: Variances are significantly different."
        )

    else:

        print(
            "Result: No significant difference in variances."
        )


# ============================================================
# 21. CHI-SQUARE INDEPENDENCE TEST
# ============================================================

print("\n========== CHI-SQUARE INDEPENDENCE TEST ==========")

table = pd.crosstab(
    df["order_status"],
    df["customer_state"]
)

chi2, p_value, dof, expected = (
    chi2_contingency(table)
)

print(
    "Chi-Square:",
    chi2
)

print(
    "P-value:",
    p_value
)

print(
    "Degrees of Freedom:",
    dof
)

if p_value < 0.05:

    print(
        "Result: Variables are statistically associated."
    )

else:

    print(
        "Result: No significant association detected."
    )


# ============================================================
# 22. CRAMER'S V
# ============================================================

print("\n========== CRAMER'S V ==========")

n = table.to_numpy().sum()

phi2 = chi2 / n

rows, columns = table.shape

cramers_v = np.sqrt(
    phi2 /
    min(
        columns - 1,
        rows - 1
    )
)

print(
    "Cramer's V:",
    cramers_v
)


# ============================================================
# 23. NORMALITY TEST
# ============================================================

print("\n========== NORMALITY TEST ==========")

print(
    "D'Agostino-Pearson Normality Test"
)

normality_results = []

normal_sample = df.sample(
    min(5000, len(df)),
    random_state=42
)

for col in numeric_cols:

    values = (
        normal_sample[col]
        .dropna()
    )

    if len(values) >= 20:

        statistic, p = normaltest(
            values
        )

        normality_results.append([
            col,
            statistic,
            p
        ])

        if p < 0.05:

            result = (
                "Not normally distributed"
            )

        else:

            result = (
                "Approximately normal"
            )

        print(
            f"{col}: p={p:.6f} -> {result}"
        )


normality_df = pd.DataFrame(
    normality_results,
    columns=[
        "Column",
        "Statistic",
        "P-value"
    ]
)

print(
    "\nNormality Test Summary:"
)

print(normality_df)


# ============================================================
# 24. SKEWNESS BEFORE TREATMENT
# ============================================================

print("\n========== SKEWNESS BEFORE TREATMENT ==========")

skew_before = df[
    numeric_cols
].skew()

print(skew_before)

print("""
Interpretation:
Around 0 = approximately symmetric
Positive = right skewed
Negative = left skewed
""")

print(
    "\nHighly skewed variables:"
)

for col, value in skew_before.items():

    if abs(value) > 1:

        print(
            col,
            "-> Highly skewed:",
            round(value, 4)
        )

    elif abs(value) > 0.5:

        print(
            col,
            "-> Moderately skewed:",
            round(value, 4)
        )

    else:

        print(
            col,
            "-> Approximately symmetric:",
            round(value, 4)
        )


# ============================================================
# 25. KURTOSIS BEFORE TREATMENT
# ============================================================

print("\n========== KURTOSIS BEFORE TREATMENT ==========")

kurt_before = df[
    numeric_cols
].kurt()

print(kurt_before)

print("""
Approximately 0 = normal-like tails
Positive = heavier tails
Negative = lighter tails
""")


# ============================================================
# 26. OUTLIER DETECTION - IQR
# ============================================================

print("\n========== OUTLIER DETECTION - IQR ==========")

continuous_cols = [
    "price",
    "freight_value"
]

outlier_table = []

for col in continuous_cols:

    Q1 = df[col].quantile(
        0.25
    )

    Q3 = df[col].quantile(
        0.75
    )

    IQR = Q3 - Q1

    lower = (
        Q1 - 1.5 * IQR
    )

    upper = (
        Q3 + 1.5 * IQR
    )

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
# 27. OUTLIER TREATMENT
# ============================================================

print("\n========== OUTLIER TREATMENT ==========")

for col in continuous_cols:

    Q1 = df[col].quantile(
        0.25
    )

    Q3 = df[col].quantile(
        0.75
    )

    IQR = Q3 - Q1

    lower = (
        Q1 - 1.5 * IQR
    )

    upper = (
        Q3 + 1.5 * IQR
    )

    df[col] = df[col].clip(
        lower=lower,
        upper=upper
    )

    print(
        col,
        "-> IQR clipping applied."
    )


# ============================================================
# 28. SKEWNESS SOLUTION
# ============================================================

print("\n========== SKEWNESS SOLUTION ==========")

# Re-check after outlier treatment

skew_after_outlier = df[
    continuous_cols
].skew()

print(
    "Skewness after outlier treatment:"
)

print(skew_after_outlier)


# Apply Yeo-Johnson only to highly skewed variables

skewed_cols = [
    col
    for col in continuous_cols
    if abs(
        skew_after_outlier[col]
    ) > 1
]

print(
    "\nVariables selected for transformation:"
)

print(skewed_cols)


if len(skewed_cols) > 0:

    transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=False
    )

    df[skewed_cols] = (
        transformer.fit_transform(
            df[skewed_cols]
        )
    )

    print(
        "Yeo-Johnson transformation applied."
    )

else:

    print(
        "No highly skewed continuous variable requires transformation."
    )


# ============================================================
# 29. SKEWNESS AFTER SOLUTION
# ============================================================

print("\n========== SKEWNESS AFTER SOLUTION ==========")

skew_after = df[
    continuous_cols
].skew()

print(skew_after)


# ============================================================
# 30. KURTOSIS AFTER SOLUTION
# ============================================================

print("\n========== KURTOSIS AFTER SOLUTION ==========")

kurt_after = df[
    continuous_cols
].kurt()

print(kurt_after)


# ============================================================
# 31. NORMALITY TEST AFTER SOLUTION
# ============================================================

print("\n========== NORMALITY TEST AFTER SOLUTION ==========")

normality_after = []

normal_sample_after = df[
    continuous_cols
].sample(
    min(5000, len(df)),
    random_state=42
)

for col in continuous_cols:

    values = (
        normal_sample_after[col]
        .dropna()
    )

    statistic, p = normaltest(
        values
    )

    normality_after.append([
        col,
        statistic,
        p
    ])

    if p < 0.05:

        result = (
            "Not normally distributed"
        )

    else:

        result = (
            "Approximately normal"
        )

    print(
        f"{col}: p={p:.6f} -> {result}"
    )


normality_after_df = pd.DataFrame(
    normality_after,
    columns=[
        "Column",
        "Statistic",
        "P-value"
    ]
)


# ============================================================
# 32. UNIVARIATE ANALYSIS
# ============================================================

print("\n========== UNIVARIATE ANALYSIS ==========")


# PRICE HISTOGRAM

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    df["price"],
    kde=True
)

plt.title(
    "Price Distribution"
)

plt.xlabel(
    "Price"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/price_histogram.png",
    dpi=300
)

plt.close()


# FREIGHT HISTOGRAM

plt.figure(
    figsize=(8, 5)
)

sns.histplot(
    df["freight_value"],
    kde=True
)

plt.title(
    "Freight Value Distribution"
)

plt.xlabel(
    "Freight Value"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/freight_histogram.png",
    dpi=300
)

plt.close()


# PRICE BOXPLOT

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    x=df["price"]
)

plt.title(
    "Price Boxplot After Outlier Treatment"
)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/price_boxplot.png",
    dpi=300
)

plt.close()


# FREIGHT BOXPLOT

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    x=df["freight_value"]
)

plt.title(
    "Freight Value Boxplot After Outlier Treatment"
)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/freight_boxplot.png",
    dpi=300
)

plt.close()


# ============================================================
# 33. BIVARIATE ANALYSIS
# ============================================================

print("\n========== BIVARIATE ANALYSIS ==========")

sample = df.sample(
    min(10000, len(df)),
    random_state=42
)

plt.figure(
    figsize=(8, 5)
)

sns.scatterplot(
    data=sample,
    x="price",
    y="freight_value",
    alpha=0.5
)

plt.title(
    "Price vs Freight Value"
)

plt.xlabel(
    "Price"
)

plt.ylabel(
    "Freight Value"
)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/price_vs_freight.png",
    dpi=300
)

plt.close()


# ============================================================
# 34. MULTIVARIATE ANALYSIS
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
    "EDA_Plots/pairplot.png",
    dpi=300
)

plt.close("all")


# ============================================================
# 35. CORRELATION
# ============================================================

print("\n========== CORRELATION ==========")

corr = df[
    numeric_cols
].corr()

print(corr)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/correlation_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 36. ORDER STATUS VISUALIZATION
# ============================================================

print(
    "\n========== ORDER STATUS VISUALIZATION =========="
)

plt.figure(
    figsize=(8, 5)
)

sns.countplot(
    data=df,
    x="order_status",
    order=df[
        "order_status"
    ].value_counts().index
)

plt.title(
    "Order Status Distribution"
)

plt.xlabel(
    "Order Status"
)

plt.ylabel(
    "Number of Records"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/order_status_countplot.png",
    dpi=300
)

plt.close()


# ============================================================
# 37. TOP CUSTOMER STATES
# ============================================================

print(
    "\n========== TOP CUSTOMER STATES =========="
)

top_states = (
    df[
        "customer_state"
    ]
    .value_counts()
    .head(15)
)

plt.figure(
    figsize=(10, 6)
)

sns.barplot(
    x=top_states.values,
    y=top_states.index
)

plt.title(
    "Top 15 Customer States"
)

plt.xlabel(
    "Number of Records"
)

plt.ylabel(
    "State"
)

plt.tight_layout()

plt.savefig(
    "EDA_Plots/top_states.png",
    dpi=300
)

plt.close()


# ============================================================
# 38. FINAL CHECK
# ============================================================

print(
    "\n========== FINAL EDA CHECK =========="
)

print(
    "Remaining missing values:",
    df.isnull().sum().sum()
)

print(
    "Remaining duplicate rows:",
    df.duplicated().sum()
)

print(
    "Final Shape:",
    df.shape
)


# ============================================================
# 39. FINAL FINDINGS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EDA FINDINGS")
print("=" * 70)

print("""
1. Dataset structure and dimensions were analyzed.

2. Missing values were identified and treated using
   median for numerical variables and mode for
   categorical variables.

3. Blank values were checked.

4. Complete duplicate rows were checked and removed
   only if present.

5. Duplicate IDs were analyzed separately.

6. Unique values and categorical frequencies were checked.

7. Descriptive statistics including mean, median,
   mode, variance and standard deviation were calculated.

8. Grouping and comparison of means were performed.

9. Levene test was used to check variance homogeneity.

10. Chi-Square test was used to check independence/
    association between categorical variables.

11. Cramer's V was calculated to understand the strength
    of the categorical association.

12. D'Agostino-Pearson normality test was applied to
    check whether numerical variables follow a normal
    distribution.

13. Skewness was checked before and after treatment.

14. IQR method was used to detect continuous-variable
    outliers.

15. Outliers in price and freight_value were treated
    using IQR clipping.

16. Highly skewed continuous variables were treated
    using Yeo-Johnson transformation.

17. Kurtosis was checked before and after treatment.

18. Univariate analysis was performed.

19. Bivariate analysis was performed.

20. Multivariate analysis was performed.

21. Correlation analysis was performed.

22. PNG visualizations were generated and saved.

23. NO categorical encoding was performed.

24. NO feature scaling was performed.

25. NO train/test split was performed.

26. NO machine learning model was trained.

27. This script is EDA and data-cleaning/preparation only.
""")


# ============================================================
# 40. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "COMPLETE EDA FINISHED SUCCESSFULLY!"
)

print("=" * 70)

print(
    "\nAll PNG plots are saved inside:"
)

print(
    os.path.abspath(
        "EDA_Plots"
    )
)

print(
    "\nEDA completed."
)

