# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas==3.0.3",
#     "scikit-learn==1.9.0",
#     "skrub==0.9.0",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Conversation Around Tabular Data Is Back. The Tables Are Still Messy.

    > This notebook is a companion to the article of the same name. You can read the article straight through, or follow along here with the code — whichever works best for you. Companion notebooks are also available in marimo and Kaggle formats.

    ---

    **Running this notebook**

    The easiest way to run this notebook with no setup is via marimo's sandbox mode as it handles dependencies automatically:

    ```bash
    uvx marimo run skrub_tabular_workflow_marimo.py
    ```

    To edit it interactively:

    ```bash
    uvx marimo edit skrub_tabular_workflow_marimo.py
    ```

    Alternatively, if you already have marimo and skrub installed in your environment:

    ```bash
    marimo edit skrub_tabular_workflow_marimo.py
    ```
    ---

    The rise of tabular foundation models has brought the spotlight back to tabular data. And yet, regardless of whether you're training gradient boosting models or experimenting with TFMs, one thing has remained unchanged: the messy data.

    In this notebook, we'll walk through a complete tabular workflow using **skrub** from inspecting a messy sales leads dataset to building a model-ready scikit-learn pipeline. The task is binary classification: predict whether a sales lead converted into a customer.

    **What we'll cover:**
    1. Loading and inspecting the data
    2. Exploring with `TableReport`
    3. Cleaning with `Cleaner`
    4. Joining external data with `fuzzy_join()`
    5. Consolidating messy categories with `deduplicate()`
    6. Feature engineering with `TableVectorizer`
    7. Building a predictive pipeline with `tabular_pipeline`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installation & Setup

    First things first, let's start by installing skrub:
    """)
    return


@app.cell
def _():
    # Uncomment to install
    # !pip install skrub -U
    return


@app.cell
def _():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate

    from skrub import (
        Cleaner,
        TableReport,
        TableVectorizer,
        deduplicate,
        fuzzy_join,
        tabular_pipeline,
    )

    return (
        Cleaner,
        LogisticRegression,
        TableReport,
        TableVectorizer,
        cross_validate,
        deduplicate,
        fuzzy_join,
        pd,
        tabular_pipeline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Loading the data

    We'll work with a synthetic sales leads dataset. The objective is to predict whether a lead eventually converted into a customer. The target column, `converted`, records this observed outcome — `1` for converted, `0` otherwise.

    Let's load the dataset and take a quick look at it. We'll separate the target from the input features and also drop `lead_id`, since it simply acts as an identifier.
    """)
    return


@app.cell
def _(pd):
    _leads = pd.read_csv("data/messy_sales_leads.csv")
    conversion_target = _leads["converted"]
    lead_features = _leads.drop(columns=["converted", "lead_id"])
    lead_features.head().T
    return conversion_target, lead_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even from the first few rows, we can already spot several potential issues:
    - Missing values in multiple columns
    - Date columns stored as strings (`created_at`, `last_contacted_at`)
    - Misspelled industry names (`Trave`, `Retial`)
    - Country name variants (`Brasil`, `Espana`, `Great Britain`)
    - A column containing almost the same value throughout (`constant_source`)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Exploring the data

    The above static preview can only tell us so much. Skrub's `TableReport` combines several exploratory views into a single interactive report. Alongside summary statistics and column distributions, it also shows relationships between variables using measures such as Pearson correlation and Cramér's V.
    """)
    return


@app.cell
def _(TableReport, lead_features):
    TableReport(lead_features)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can click on any cell in the generated table and get detailed information about the corresponding column, including missing values, the number of unique values, value distributions, and the most common categories.

    If you're working outside a notebook environment, you can also open the report directly in your browser, or export it as an HTML file to share with colleagues:
    """)
    return


@app.cell
def _():
    # TableReport(lead_features).open()
    # TableReport(lead_features).write_html("lead_features_report.html")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Pre-processing the data

    Now that we have a better understanding of the dataset, it's time to clean things up. Some issues can be fixed automatically, while others require a bit more context and standardization.

    ### 3.1 Cleaning common data issues with Cleaner

    `Cleaner` is a scikit-learn-compatible transformer that automates several routine preprocessing tasks: parsing date columns correctly, cleaning null strings, dropping uninformative columns, and casting near-constant columns to strings.

    It is important to mention that while `Cleaner` standardizes null-like values and ensures they are treated consistently across the dataframe, it does **not** perform imputation. Missing-value handling strategies can be applied later depending on the modeling workflow.
    """)
    return


@app.cell
def _(Cleaner, lead_features):
    cleaned_features = Cleaner(drop_if_constant=True).fit_transform(lead_features)
    cleaned_features.info()
    return (cleaned_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see, the datetime columns have now been correctly parsed, and the `constant_source` column has been removed automatically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Joining external data with fuzzy_join

    More often than not, useful data is spread across multiple tables. In our case, each lead has an `office_country` value but that alone doesn't tell us much about the broader market. We'll bring in a small country reference dataset containing fields such as `market_size_score`, `digital_adoption_score`, `average_deal_value_index`, and `sales_cycle_index`.

    The challenge is that country names are not written consistently — the same country may appear as `USA`, `U.S.`, or `United States` while the reference table uses `United States of America`. Similarly, `Brasil` should match `Brazil`, and `Espana` should match `Spain`.

    A regular `pandas.merge()` only works when the join keys match exactly. This is where `fuzzy_join()` comes in.
    """)
    return


@app.cell
def _(pd):
    countries = pd.read_csv("data/country_reference.csv")
    countries.head()
    return (countries,)


@app.cell
def _(cleaned_features, countries, fuzzy_join):
    leads_enriched = fuzzy_join(
        cleaned_features,
        countries,
        left_on="office_country",
        right_on="country_name",
    )

    # Check the matches
    leads_enriched[["office_country", "country_name"]].drop_duplicates().sort_values(
        "country_name"
    ).head(20)
    return (leads_enriched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default, `fuzzy_join()` matches each value to its closest counterpart in the reference table. The matching threshold can be adjusted using the `max_dist` parameter.

    ### 3.3 Consolidating similar categories with deduplicate

    The country names problem was relatively easy to solve because we had a separate reference table. But what do you do when no such table exists? In our dataset, the `industry_name` column contains several entries that appear to describe the same category but are written slightly differently.
    """)
    return


@app.cell
def _(cleaned_features):
    sorted(cleaned_features["industry_name"].dropna().unique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Should `Education` and `Educaton` be treated as different industries? What about `Finanace` and `Finance`, or `Healtcare` and `Healthcare`? In situations like these, manually cleaning categories quickly becomes tedious.

    Skrub has a built-in `deduplicate()` function for these scenarios that groups similar strings together and maps them to a cleaner category. Under the hood, it uses clustering based on string similarities.
    """)
    return


@app.cell
def _(cleaned_features, deduplicate):
    cleaned_features["clean_industry_name"] = cleaned_features["industry_name"].replace(
        deduplicate(cleaned_features["industry_name"].dropna()).to_dict()
    )

    sorted(cleaned_features["clean_industry_name"].dropna().unique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One thing to keep in mind: similar-looking categories are not always the same. This step works best when spelling variations or duplicate labels genuinely refer to the same underlying category. Otherwise, important information may be lost.

    From this point onward, we'll use `model_features` as the dataframe passed to the modeling pipeline.
    """)
    return


@app.cell
def _(leads_enriched):
    model_features = leads_enriched.copy()
    model_features.head()
    return (model_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Performing feature engineering

    After cleaning and enriching the dataset, the final step before modeling is to transform the mixed collection of numeric, categorical, datetime, and text columns into representations that machine learning models can work with.

    Skrub's `TableVectorizer` simplifies much of this process by automatically selecting suitable transformations based on the characteristics of each column. By default:

    - **Low-cardinality categorical columns** are one-hot encoded
    - **High-cardinality categorical columns** are transformed using `StringEncoder`
    - **Numeric columns** are passed through unchanged
    - **Datetime columns** are encoded using `DatetimeEncoder`
    """)
    return


@app.cell
def _(TableVectorizer, model_features):
    vectorizer = TableVectorizer()
    encoded_lead_features = vectorizer.fit_transform(model_features)

    encoded_lead_features.shape
    return (vectorizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `TableVectorizer` is not a black box — the fitted object is fully inspectable. You can check how skrub classified each input column:
    """)
    return


@app.cell
def _(vectorizer):
    vectorizer.column_to_kind_
    return


@app.cell
def _(vectorizer):
    # Inspect the full fitted vectorizer
    vectorizer
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Building a predictive pipeline

    Because `TableVectorizer` follows the scikit-learn API, it can be combined with virtually any scikit-learn model using the usual pipeline utilities:
    """)
    return


@app.cell
def _(TableVectorizer):
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier

    pipeline = make_pipeline(
        TableVectorizer(),
        HistGradientBoostingClassifier(random_state=0),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But skrub also provides `tabular_pipeline`, a convenience function that automatically combines tabular preprocessing with a scikit-learn estimator — and adapts its defaults to the chosen model.
    """)
    return


@app.cell
def _(tabular_pipeline):
    pipeline_1 = tabular_pipeline("classification")
    pipeline_1
    return (pipeline_1,)


@app.cell
def _(LogisticRegression, tabular_pipeline):
    # With a specific estimator — pipeline adapts its preprocessing steps accordingly
    logistic_pipeline = tabular_pipeline(LogisticRegression(max_iter=1000))
    logistic_pipeline
    return


@app.cell
def _(conversion_target, cross_validate, model_features, pipeline_1):
    _scores = cross_validate(
        pipeline_1, model_features, conversion_target, cv=3, scoring="roc_auc"
    )
    test_scores = _scores["test_score"]
    print(f"ROC-AUC scores: {test_scores}")
    print(f"Mean: {test_scores.mean():.3f}")
    return


@app.cell
def _(conversion_target, model_features, pipeline_1):
    # Fit and generate predictions
    pipeline_1.fit(model_features, conversion_target)
    predictions = pipeline_1.predict(model_features)
    conversion_probabilities = pipeline_1.predict_proba(model_features)
    conversion_probabilities[:5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full workflow in one cell

    Here's the complete pipeline without the explanatory cells — useful if you just want to run the whole thing end to end.
    """)
    return


@app.cell
def _(Cleaner, cross_validate, deduplicate, fuzzy_join, pd, tabular_pipeline):
    _leads = pd.read_csv("data/messy_sales_leads.csv")
    conversion_target_1 = _leads.pop("converted")
    lead_features_1 = _leads.drop(columns="lead_id")
    lead_features_1 = Cleaner(drop_if_constant=True).fit_transform(lead_features_1)
    mask = lead_features_1["industry_name"].notna()
    lead_features_1.loc[mask, "industry_name"] = deduplicate(
        lead_features_1.loc[mask, "industry_name"].reset_index(drop=True)
    ).array
    countries_1 = pd.read_csv("data/country_reference.csv")
    lead_features_1 = fuzzy_join(
        lead_features_1, countries_1, left_on="office_country", right_on="country_name"
    )
    model = tabular_pipeline("classification")
    _scores = cross_validate(
        model, lead_features_1, conversion_target_1, cv=3, scoring="roc_auc"
    )
    print(f"ROC-AUC: {_scores['test_score'].mean():.3f}")
    return


if __name__ == "__main__":
    app.run()
