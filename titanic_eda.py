import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

RAW_CSV = "titanic_raw.csv"
CLEAN_CSV = "titanic_cleaned.csv"
CHART_DIR = "charts"


def banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_data():
    df = pd.read_csv(RAW_CSV)
    return df


def step1_inspect(df):
    banner("STEP 1: DATAFRAME OPERATIONS")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Memory used: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print("\n--- head() ---")
    print(df.head())

    print("\n--- info() ---")
    df.info()

    print("\n--- describe() numeric ---")
    print(df.describe().round(2))

    print("\n--- describe() categorical (transposed) ---")
    print(df.describe(include=["object", "bool"]).T)

    print("\n--- dtypes ---")
    print(df.dtypes)

    print("\n--- value_counts() ---")
    print(df["sex"].value_counts())
    print(df["embarked"].value_counts(dropna=False))

    print("\n--- unique counts per column ---")
    print(df.nunique().sort_values(ascending=False))


def step2_duplicates(df):
    banner("STEP 2: DUPLICATES")
    n_dup = df.duplicated().sum()
    print(f"Duplicate rows: {n_dup} ({n_dup / len(df):.1%})")
    if n_dup:
        print("Sample duplicated rows:")
        print(df[df.duplicated(keep=False)].sort_values(list(df.columns)).head(6))
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after drop_duplicates(): {df.shape}")
    print("Note: identity columns (name, ticket) are absent in this version,")
    print("so exact duplicates may still be distinct passengers; kept removal as cleaning demo.")
    return df


def step3_missing(df, charts=True):
    banner("STEP 3: MISSING VALUES")
    missing = df.isna().sum()
    missing_pct = df.isna().mean() * 100
    report = pd.DataFrame({"missing": missing, "pct": missing_pct.round(1)})
    print(report[report["missing"] > 0])

    if charts:
        fig, ax = plt.subplots(figsize=(8, 4))
        report[report["missing"] > 0]["missing"].sort_values().plot(
            kind="barh", ax=ax, color="crimson"
        )
        ax.set_title("Missing values per column (raw data)")
        ax.set_xlabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, "missing_values.png"), dpi=120)
        plt.close(fig)

    print("\n--- Handling strategy ---")
    deck_pct = df["deck"].isna().mean()
    if deck_pct > 0.5:
        df = df.drop(columns="deck")
        print(f"deck: dropped column ({deck_pct:.1%} missing - too sparse)")

    group_median_age = df.groupby(["pclass", "sex"])['age'].median()
    print("\nage: median by (pclass, sex) group used for imputation")
    print(group_median_age)
    df["age"] = df["age"].fillna(df.groupby(["pclass", "sex"])["age"].transform("median"))

    for col in ["embarked", "embark_town"]:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)
        print(f"{col}: filled with mode '{mode}'")

    print(f"\nMissing values remaining: {df.isna().sum().sum()}")
    print(f"Shape after cleaning: {df.shape}")
    return df


def step4_grouping(df):
    banner("STEP 4: GROUPING AND AGGREGATION")

    df["family_size"] = df["sibsp"] + df["parch"] + 1

    print("\n--- Survival rate by sex ---")
    print(df.groupby("sex")["survived"].agg(["count", "mean"]).round(3))

    print("\n--- Survival rate by passenger class ---")
    print(df.groupby("pclass")["survived"].agg(["count", "mean"]).round(3))

    print("\n--- Survival rate by embarkation port ---")
    print(df.groupby("embark_town")["survived"].agg(["count", "mean"]).round(3))

    print("\n--- Multi-column groupby: survival by class and sex ---")
    print(df.groupby(["pclass", "sex"])["survived"].mean().unstack().round(3))

    print("\n--- Multi-stat aggregation per class ---")
    agg = df.groupby("pclass").agg(
        passengers=("survived", "count"),
        survival_rate=("survived", "mean"),
        avg_age=("age", "mean"),
        avg_fare=("fare", "mean"),
        max_fare=("fare", "max"),
        avg_family=("family_size", "mean"),
    )
    print(agg.round(2))

    print("\n--- pivot_table: mean fare by class and port ---")
    print(df.pivot_table(index="pclass", columns="embark_town", values="fare", aggfunc="mean").round(1))

    print("\n--- crosstab: sex vs class ---")
    print(pd.crosstab(df["sex"], df["pclass"], margins=True))

    print("\n--- Age groups vs survival (pd.cut) ---")
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 12, 18, 35, 60, 100],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    )
    print(df.groupby("age_group", observed=True)["survived"].agg(["count", "mean"]).round(3))

    print("\n--- Families vs solo travelers ---")
    print(df.groupby("alone")["survived"].agg(["count", "mean"]).round(3))

    print("\n--- Top 5 most expensive tickets ---")
    print(df.nlargest(5, "fare")[["fare", "pclass", "sex", "survived"]].to_string(index=False))

    print("\n--- Correlation with survival ---")
    corr = df[["survived", "pclass", "age", "sibsp", "parch", "fare", "alone"]].corr()["survived"]
    print(corr.sort_values(key=abs, ascending=False).round(3))

    print("\n--- transform + apply(): passengers paying >2x class median fare ---")
    df["fare_ratio"] = df.groupby("pclass")["fare"].transform(lambda s: s / s.median())
    overpaying = df.groupby("pclass")["fare_ratio"].apply(lambda s: (s > 2).mean())
    print(overpaying.rename("pct_above_2x_median").round(3))

    return df


def step5_charts(df):
    banner("STEP 5: CHARTS")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.barplot(data=df, x="sex", y="survived", hue="pclass", ax=axes[0, 0])
    axes[0, 0].set_title("Survival rate by sex and class")
    axes[0, 0].set_ylabel("survival rate")

    sns.histplot(data=df, x="age", hue="survived", kde=True, bins=30, ax=axes[0, 1])
    axes[0, 1].set_title("Age distribution by survival")

    sns.boxplot(data=df, x="pclass", y="fare", hue="survived", ax=axes[1, 0])
    axes[1, 0].set_title("Fare by class and survival")
    axes[1, 0].set_ylim(0, 150)

    sns.countplot(data=df, x="embark_town", hue="survived", ax=axes[1, 1])
    axes[1, 1].set_title("Passengers and survival by port")

    fig.suptitle("Titanic EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "eda_dashboard.png"), dpi=120)
    plt.close(fig)
    print(f"Saved {CHART_DIR}/eda_dashboard.png")


def step6_insights(df):
    banner("STEP 6: KEY INSIGHTS")
    overall = df["survived"].mean()
    female = df.loc[df["sex"] == "female", "survived"].mean()
    male = df.loc[df["sex"] == "male", "survived"].mean()
    c1 = df.loc[df["pclass"] == 1, "survived"].mean()
    c3 = df.loc[df["pclass"] == 3, "survived"].mean()
    child = df.loc[df["age"] <= 12, "survived"].mean()
    solo = df.loc[df["alone"], "survived"].mean()
    fam = df.loc[~df["alone"], "survived"].mean()

    print(f"* Overall survival rate: {overall:.1%} ({df['survived'].sum()} of {len(df)})")
    print(f"* Sex was the strongest factor: women {female:.1%} vs men {male:.1%}")
    print(f"* Wealth mattered: 1st class {c1:.1%} vs 3rd class {c3:.1%}")
    print(f"* Children (<=12) survived at {child:.1%}")
    print(f"* Traveling with family helped: {fam:.1%} vs solo {solo:.1%}")
    print(f"* 3rd class paid on average {df.loc[df['pclass']==3, 'fare'].mean():.0f} vs "
          f"{df.loc[df['pclass']==1, 'fare'].mean():.0f} in 1st class")


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    df = load_data()
    step1_inspect(df)
    df = step2_duplicates(df)
    df = step3_missing(df)
    df = step4_grouping(df)
    step5_charts(df)
    step6_insights(df)

    df.to_csv(CLEAN_CSV, index=False)
    banner("DONE")
    print(f"Cleaned dataset saved to {CLEAN_CSV} ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"Charts saved to {CHART_DIR}/")


if __name__ == "__main__":
    main()