import marimo

__generated_with = "0.25.1"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # In-class Competition Tutorial
    This NOTEBOOK will provide an introduction to the process of creating forecast results and the basic methodology.

    First, let's review the task we will be performing (see README.ipynb for details).

    **Objective**: To predict the probability of default based on customer data.

    **Evaluation metric**: ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contents
    - [1.Setup](#scrollTo=a5KcDAE8CnmB)
    - [2.Loading the Data](#scrollTo=tp0cW0CD11Hi&line=1&uniqifier=1)
    - [3.Visualizing and Understanding the Data](#scrollTo=3NuP1zcmCnmF&line=1&uniqifier=1)
    - [4.Preprocessing and Feature Creation](#scrollTo=rsPYkguwCnmO&line=1&uniqifier=1)
    - [5.Building the Machine Learning Model](#scrollTo=FoKdK60PCnmP&line=1&uniqifier=1)
    - [6.Creating prediction results](#scrollTo=tn_kdvWYCnmQ&line=2&uniqifier=1)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Import Libraries
    Let's load basic libraries.
    Other required libraries will be loaded when we explain them.
    - numpy: Library for efficient numerical computation
    - pandas: Library useful for data analysis
    - matplotlib: Graph drawing library
    - seaborn: Graph drawing library as well
    """)
    return


@app.cell
def _():
    # Importing libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import warnings
    warnings.filterwarnings('ignore')
    return np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Connect with Google Drive
    To load the data, we first need to connect this Colab notebook with Google Drive.
    """)
    return


@app.cell
def _():
    # # If you work with Google Colaboratory, please run this as well.
    # from google.colab import drive
    # drive.mount('/content/drive')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we need to navigate to where the `competition` folder is located

    **IMPORTANT**:<br>
    Change the path in the `%cd` command below to match the folder where this notebook is saved on Google Drive by **replacing "WhereThisNotebookIsLocated" with your actual folder path**.
    """)
    return


@app.cell
def _():
    # # Specify the directory where this notebook is located after %cd.
    # %cd "/content/drive/MyDrive/WhereThisNotebookIsLocated"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell below to check if the path is correctly set.
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    # Automatically get the current working directory
    current_dir = Path(os.getcwd())

    # Define file paths using pathlib
    train_file = current_dir / "input" / "train.csv"
    test_file = current_dir / "input" / "test.csv"
    sample_sub_file = current_dir / "input" / "sample_submission.csv"

    # Check if path exists
    if train_file.exists() and test_file.exists() and sample_sub_file.exists():
        print("All files exist and path is correctly set.")
    else:
        print("Some files are missing or path is not correctly set.")
    return current_dir, os, sample_sub_file, test_file, train_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.Loading the Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Data Overview
    Run the cell to load the dataset as `pd.DataFrame`.

    **IMPORTANT:**<br>
    **When you make modifications to preprocessing or model training, always make sure to run all cells from this cell.**
    """)
    return


@app.cell
def _(pd, sample_sub_file, test_file, train_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    sample_sub = pd.read_csv(sample_sub_file)
    return sample_sub, test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before conducting a full-scale analysis, we will first review a brief overview of the data.
    """)
    return


@app.cell
def _(train):
    # Check train data
    print(f"train shape: {train.shape}")
    train.head(3)
    return


@app.cell
def _(test):
    # Check test data
    print(f"test shape: {test.shape}")
    test.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Excluding the `TARGET` column in train data and the `SK_ID_CURR` which represents ID number, you can see that there are 32 types of features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Selecting Features

    It is often difficult to perform data analysis and preprocessing on all features from the beginning. Instead, an easier way to get started is to start with a small number of features and then add features one by one.

    This notebook will focus on 5 features. For the remaining 25 types of features, please refer to the lecture materials, the methods introduced in this notebook, etc., and perform the analysis on your own.

    Feel free to also ask questions in lectures, office hours, or in the Slack community!
    """)
    return


@app.cell
def _(test, train):
    # Focus on 5 features
    use_features = [
        "NAME_CONTRACT_TYPE",
        "AMT_INCOME_TOTAL",
        "EXT_SOURCE_2",
        "OWN_CAR_AGE",
        "ORGANIZATION_TYPE",
        # add new features to use here
    ]
    target = train["TARGET"].values

    train_1 = train[use_features].copy()
    train_1["TARGET"] = target
    test_1 = test[use_features].copy()
    return test_1, train_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check the data once again.
    """)
    return


@app.cell
def _(train_1):
    # Check train data
    print(f"train shape: {train_1.shape}")
    train_1.head(3)
    return


@app.cell
def _(test_1):
    # Check test data
    print(f"test shape: {test_1.shape}")
    test_1.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **[Next Steps]**
    > + Try adding more features, starting with the ones that you think are more relevant for predicting default probability. Check `HomeCredit_columns_description.xlsx` to understand what each column represents.
    > + When you add new features, always restart from [Section 2.1](#scrollTo=2TTHzi1c3a5E&line=5&uniqifier=1) by reloading the dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.Visualizing and Understanding the Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first thing we need to do before building the machine learning model is to **understand the data**. We do this by visualizing and analyzing, to deepen our understanding of data distribution, missing values, outliers, correlations, and etc. The results of the analysis obtained at this stage will be useful for preprocessing, feature creation, and selection of machine learning models, which are all important to building models with better prediction ability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Checking missing values
    In this section, we check for missing values.
    This is important as **most machine learning models cannot be trained on data with missing values**. If there are missing values, they need to be filled with some value.
    """)
    return


@app.cell
def _(train_1):
    # Check missing values of train data
    train_1.isnull().sum()
    return


@app.cell
def _(test_1):
    # Check missing values of test data
    test_1.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We found that there are missing values in `EXT_SOURCE_2` and `OWN_CAR_AGE`. We will deal with these missing values later. Of course, there is a possibility that there are missing values for other features that we are not covering here, so please check them by yourself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Findings**:<br>
    * Need to deal with missing values in `EXT_SOURCE_2` and `OWN_CAR_AGE`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Visualization and analysis of each feature
    In this section, we visualize each feature and analyze to see what kind of characteristics it has.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### TARGET column
    """)
    return


@app.cell
def _(plt, sns, train_1):
    # The distribution of the target (default or not)
    sns.countplot(data=train_1, x="TARGET")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that the **distribution of the objective variable is highly skewed**. Data in which the distribution of the objective variable is highly skewed in this way is called **unbalanced data**.

    When dealing with unbalanced data, we need to be particularly careful in selecting evaluation metrics. For example, if you choose accuracy, you will find that simply predicting all zeros will result in a high accuracy. **Choosing such an inappropriate metric can cause the machine learning model to fail to predict well on new data**.

    Another approach to dealing with unbalanced data is to try to balance the distribution of the target variable. The method of reducing the data of the larger target variable is called undersampling, while the method of increasing the data of the smaller objective variable is called oversampling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Findings**:<br>
    * (May) need to think about methods to mitigate the skewedness of the target variable
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### NAME_CONTRACT_TYPE column
    """)
    return


@app.cell
def _(plt, sns, train_1):
    # The distribution of NAME_CONTRACT_TYPE
    sns.countplot(data=train_1, x="NAME_CONTRACT_TYPE")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are two variables in `NAME_CONTRACT_TYPE`, Cash loans and Revolving loans, but they are not evenly distributed. Also, since the machine learning model can only handle data of numeric type, it is necessary to convert the data from string type to numeric type.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Findings**:<br>
    * (May) need to think about methods to mitigate the skewedness of the target variable
    * Need to convert the data from string type to numeric type
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ORGANIZATION_TYPE column
    """)
    return


@app.cell
def _(plt, sns, train_1):
    # The distribution of ORGANIZATION_TYPE
    plt.figure(figsize=(30, 10))
    sns.countplot(data=train_1, x="ORGANIZATION_TYPE")
    plt.tick_params(axis="x", rotation=90)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are many different `ORGANIZATION_TYPE`s, and you can also see that there is an ununiformity in the number of data. This is also a string type feature, so it needs to be converted to a numeric type. Also, the second variable from the left is "XNA", which we can infer from its name to be a missing value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Findings**:<br>
    * Treat "XNA" as missing values
    * Need to convert the data from string type to numeric type
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EXT_SOURCE_2 column
    """)
    return


@app.cell
def _(plt, sns, train_1):
    # The distribution of EXT_SOURCE_2
    sns.displot(data=train_1, x="EXT_SOURCE_2")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that EXT_SOURCE_2 is normalized between 0 and 1. It seems we can handle this feature as it is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Findings**:<br>
    * No additional preprocessing is needed
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### AMT_INCOME_TOTAL column
    """)
    return


@app.cell
def _(plt, sns, train_1):
    # The distribution of AMT_INCOME_TOTAL
    sns.displot(data=train_1, x="AMT_INCOME_TOTAL")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The visualization of `AMT_INCOME_TOTAL` is hard to interpret. THis may be caused by the presence of a small number of outliers that take large values. To visualize data like this, a logarithmic transformation can be effective.
    """)
    return


@app.cell
def _(plt, sns, train_1):
    # The distribution of AMT_INCOME_TOTAL（Logarithmic transformation）
    sns.displot(data=train_1, x="AMT_INCOME_TOTAL", log_scale=10)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We displayed the graph successfully by using logarithmic transformation.
    The income is supposed to be a continuous value, but it looks like a discrete value. Let's have a look at the type of `AMT_INCOME_TOTAL` values.
    """)
    return


@app.cell
def _(train_1):
    # Check the type of AMT_INCOME_TOTAL values
    len(train_1["AMT_INCOME_TOTAL"].unique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are 171202 data in train, but `AMT_INCOME_TOTAL` consists of only 1641 different values. Let's check the top 10 values specifically.
    """)
    return


@app.cell
def _(train_1):
    # Top 10 values of AMT_INCOME_TOTAL
    train_1["AMT_INCOME_TOTAL"].value_counts().head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It appears that `AMT_INCOME_TOTAL` is not an exact annual income, but rather data compiled from a rounded number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Findings**:<br>
    * Should the outlier in the data be addressed?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### OWN_CAR_AGE column
    """)
    return


@app.cell
def _(plt, sns, train_1):
    # The distribution of OWN_CAR_AGE
    sns.displot(data=train_1, x="OWN_CAR_AGE")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `OWN_CAR_AGE` can be inferred to be in years from the scale of values. In addition, the distribution is natural from 0 to 40, but there is an unnatural distribution around 60 to 70. It is hard to imagine that the number of years a car has been purchased increases suddenly like this, so they are considered to be outliers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Findings**:<br>
    * Treat numbers above 60 as outliers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Up to this point, we have visualized and analyzed each feature. I believe that you have realized that visualization requires some ingenuity and that visualization can deepen your understanding of data. I am sure that the visualization and analysis of the 25 features not covered here will lead to improved forecasting accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **[Next Steps]**
    > + Check for missing values for the features you have added in Section 2.2.
    > + Visualize the features you have added. Is the feature categorical or continuous? What type of graph is most effective to understand it?
    > + What do you notice about the features? What kind of preprocessing is needed?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.Preprocessing and Feature Creation
    Here, we will conduct the preprocessing and create new features based on what we have learned in the preceding visualization and analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### NAME_CONTRACT_TYPE column
    Convert `NAME_CONTRACT_TYPE` to a numeric type. In this case, “Cash loans” is converted to 0 and “Revolving loans” to 1. This method of simply replacing an integer is called **Label Encoding**.
    """)
    return


@app.cell
def _(test_1, train_1):
    # Numerization of NAME_CONTRACT_TYPE（Label Encoding）
    train_2 = train_1.copy()
    test_2 = test_1.copy()
    train_2["NAME_CONTRACT_TYPE"] = train_2["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1})
    test_2["NAME_CONTRACT_TYPE"] = test_2["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1})

    train_2.head(5)
    return test_2, train_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ORGANIZATION_TYPE column
    Convert `ORGANIZATION_TYPE` to a numeric type. This time, we will convert the variable to numeric in terms of the number of data in the variable. For example, if the number of data in “Police” is 1279 and the number of data in “Bank” is 1385, convert “Police” to 1279 and “Bank” to 1385. This method of replacing the number of data with the number of data is called **Count Encoding**.
    """)
    return


@app.cell
def _(test_2, train_2):
    # Numerization of ORGANIZATION_TYPE (Count Encoding）
    train_3 = train_2.copy()
    test_3 = test_2.copy()
    organization_ce = train_3["ORGANIZATION_TYPE"].value_counts()
    train_3["ORGANIZATION_TYPE"] = train_3["ORGANIZATION_TYPE"].map(organization_ce)
    test_3["ORGANIZATION_TYPE"] = test_3["ORGANIZATION_TYPE"].map(organization_ce)

    train_3.head(5)
    return test_3, train_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EXT_SOURCE_2 column
    Fill missing values in `EXT_SOURCE_2`. There are various methods for completing missing values, but in this case, since the number of missing values is small, we simply use the average value to complete the missing values.

    **IMPORANT**:
    When you fill the missing values in the test data, you need to **fill with the average of the train data**.
    """)
    return


@app.cell
def _(test_3, train_3):
    # Complete missing values of EXT_SOURCE_2 with the average
    train_4 = train_3.copy()
    test_4 = test_3.copy()
    train_4["EXT_SOURCE_2"] = train_4["EXT_SOURCE_2"].fillna(train_4["EXT_SOURCE_2"].mean())
    test_4["EXT_SOURCE_2"] = test_4["EXT_SOURCE_2"].fillna(train_4["EXT_SOURCE_2"].mean())  # Use average of train data to fill test data

    train_4.isnull().sum()
    return test_4, train_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OWN_CAR_AGE column
    First, we will replace the unnatural outliers that are over 60 as `np.nan` (missing values).
    """)
    return


@app.cell
def _(np, test_4, train_4):
    # Treat values above 60 (outliers) in OWN_CAR_AGE as missing values
    train_5 = train_4.copy()
    test_5 = test_4.copy()
    train_5.loc[train_5["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan
    test_5.loc[test_5["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan
    return test_5, train_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we consider the handling of missing values. The original `OWN_CAR_AGE` had 112992 missing values out of 171202 data. With such a large number of missing values, it is difficult and impractical to properly fill the missing values with some value. Therefore, we will group `OWN_CAR_AGE` by decade (e.g. Group 1: 0-9 years, Group 2: 10-19 years, etc.), then apply **One Hot Encoding**.
    """)
    return


@app.cell
def _(test_5, train_5):
    # Divide OWN_CAR_AGE into groups
    train_6 = train_5.copy()
    test_6 = test_5.copy()
    train_6["OWN_CAR_AGE"] = train_6["OWN_CAR_AGE"] // 10
    test_6["OWN_CAR_AGE"] = test_6["OWN_CAR_AGE"] // 10

    train_6["OWN_CAR_AGE"].unique()
    return test_6, train_6


@app.cell
def _(pd, test_6, train_6):
    # Apply One Hot Encoding to OWN_CAR_AGE
    train_car_age_ohe = pd.get_dummies(train_6["OWN_CAR_AGE"]).add_prefix("OWN_CAR_AGE_")
    test_car_age_ohe = pd.get_dummies(test_6["OWN_CAR_AGE"]).add_prefix("OWN_CAR_AGE_")

    # Add the one hot encoded columns to train/test
    train_7 = pd.concat([train_6, train_car_age_ohe], axis=1)
    test_7 = pd.concat([test_6, test_car_age_ohe], axis=1)

    # Remove original OWN_CAR_AGE
    train_7 = train_7.drop('OWN_CAR_AGE', axis=1)
    test_7 = test_7.drop('OWN_CAR_AGE', axis=1)

    train_7.head(5)
    return test_7, train_7


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **[Next Steps]**
    > + Apply preprocessing to the features you added. Is it correctly preprocessed?
    > + Explore other preprocessing methods to apply to the features.
    > + If you have errors, try reloading the dataset by going back to [Section 2.1](#scrollTo=2TTHzi1c3a5E&line=5&uniqifier=1).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.Building the Machine Learning Model
    Now, we are ready to start building the machine learning model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 Import Additional Libraries
    First, we import the necessary libraries for training and evaluation.

    - `train_test_split`: Split data into training and evaluation data.
    - `StandardScaler`: Standardize the data.
    - `roc_auc_score`: Calculate ROC-AUC, the evaluation metric for this competition.
    """)
    return


@app.cell
def _():
    # Importing libraries
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    return StandardScaler, roc_auc_score, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Preparing the Data
    Split the data into explanatory and target variables. The target variable for this dataset is `TARGET` column and the rest are explanatory variables.
    """)
    return


@app.cell
def _(test_7, train_7):
    # Split the data into explanatory and target variables
    X = train_7.drop("TARGET", axis=1).values
    y = train_7["TARGET"].values
    X_test = test_7.values
    return X, X_test, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Standardize the data. Standardization is the operation of transforming the values so that the mean is 0 and the variance is 1. Some models, such as logistic regression and neural networks, do not learn well without scaling the values in this way.
    """)
    return


@app.cell
def _(StandardScaler, X, X_test):
    # Standardization
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    X_test_std = sc.transform(X_test)
    return X_std, X_test_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 Training the Model
    We first split the training data into training data and validation data. This method of keeping a portion of the training data for evaluation and not using it for training is called the **holdout method**. This is one method to approximate the model's predictive ability on unknown data (**generalization** performance).

    Here, we will use 70% of the data as training data and 30% as validation data
    """)
    return


@app.cell
def _(X_std, train_test_split, y):
    # Split the original data into the training data and the validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X_std, y, test_size=0.3, stratify=y, random_state=0)
    return X_train, X_valid, y_train, y_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's create models with logistic regression and random forest.
    """)
    return


@app.cell
def _(X_train, X_valid, roc_auc_score, y_train, y_valid):
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(random_state=0)
    lr.fit(X_train, y_train)

    lr_train_pred = lr.predict_proba(X_train)[:, 1]
    lr_valid_pred = lr.predict_proba(X_valid)[:, 1]
    print(f"Train Score: {roc_auc_score(y_train, lr_train_pred)}")
    print(f"Valid Score: {roc_auc_score(y_valid, lr_valid_pred)}")
    return lr_train_pred, lr_valid_pred


@app.cell
def _(X_train, X_valid, roc_auc_score, y_train, y_valid):
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(random_state=0, max_depth=10)
    rf.fit(X_train, y_train)

    rf_train_pred = rf.predict_proba(X_train)[:, 1]
    rf_valid_pred = rf.predict_proba(X_valid)[:, 1]
    print(f"Train Score: {roc_auc_score(y_train, rf_train_pred)}")
    print(f"Valid Score: {roc_auc_score(y_valid, rf_valid_pred)}")
    return rf, rf_train_pred, rf_valid_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We find that random forest results with higher validation score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 (Optional) Ensemble Learning
    Now that we have created two models, we can try combining these two models for better predictive ability (**ensemble learning**). There are various methods for ensemble learning, but here we will simply take the average of the two models.
    """)
    return


@app.cell
def _(
    lr_train_pred,
    lr_valid_pred,
    rf_train_pred,
    rf_valid_pred,
    roc_auc_score,
    y_train,
    y_valid,
):
    train_pred = (lr_train_pred + rf_train_pred) / 2
    valid_pred = (lr_valid_pred + rf_valid_pred) / 2

    print(f"Train Score: {roc_auc_score(y_train, train_pred)}")
    print(f"Valid Score: {roc_auc_score(y_valid, valid_pred)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We find that in this case, ensemble leaning does not contribute to improved score. So, **we will use the random forest model as the final model to make predictions on the test data**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **[Next Steps]**
    > + Is holdout method the best method to evaluate your model?
    > + Is the model's hyperparameters optimized? What hyperparameters needs tuning?
    > + Explore other models to use to make predictions.
    > + Explore other ensembling methods to further improve the model's performance.
    > + If you have errors, try reloading the dataset by going back to [Section 2.1](#scrollTo=2TTHzi1c3a5E&line=5&uniqifier=1).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.Creating Prediction Results
    Finally, let's make a prediction for the test data, and prepare a CSV file to submit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 Predicting on the test data
    We found in Sections 5.3 and 5.4 that the best model was random forest model. Therefore, we will use this model to make the final prediction.

    If you made any changes and found a better model, you will need to change the code below accordingly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ```python
    "
        "# If logistic regression model was better
    "
        "pred = lr.predict_proba(X_test_std)[:, 1]
    "
        "```
    """)
    return


@app.cell
def _(X_test_std, rf):
    # Make predictions for the test data
    # Change model name if needed
    pred = rf.predict_proba(X_test_std)[:, 1]
    return (pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Saving the prediction as CSV file [DO NOT CHANGE]
    **WARNING**: DO **NOT** CHANGE THE CODES BELOW!!!
    """)
    return


@app.cell
def _(pred, sample_sub):
    # Put the prediction into the format of submission
    submission = sample_sub.copy()
    submission['TARGET'] = pred
    submission
    return (submission,)


@app.cell
def _(current_dir, os, submission):
    # Create the "output" directory if it doesn't exist
    output_dir = current_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Specify the new output file path
    output_file = output_dir / "submission.csv"

    # Save the CSV file to the "output" directory
    submission.to_csv(output_file, index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's all for the tutorial of Home Credit Default Risk competition! Submit your CSV file to Omnicampus to see the result.

    Only 5 out of 30 features are covered in this notebook, so there are a lot of room for improvement. Check out **[Next Steps]** in each section to see what you can do to improve your score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Saving this notebook as a ZIP file [REQUIREMENT IF AIMING FOR HONORS/OUTSTANDING STUDENTS]
    This section compresses the notebook file (`tutorial.py`) into a ZIP file and saves it to your working directory.
    Make sure to save the notebook (Ctrl+S or Cmd+S) before running this cell to include the latest changes.

    If you are aiming to become **Honors or Outstanding Student**, you need to submit the ZIP file as well on Omnicampus.
    """)
    return


@app.cell
def _(current_dir):
    import zipfile

    # Define the file names
    notebook_filename = "tutorial.py"  # Enter the name of this notebook
    zip_filename = "tutorial.zip"

    # Define the full paths using current_dir
    notebook_path = current_dir / notebook_filename
    zip_output_path = current_dir / zip_filename

    # Check if the notebook file exists and create the ZIP file
    if notebook_path.exists():
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the notebook to the ZIP file (arcname avoids including the full folder structure)
            zipf.write(notebook_path, arcname=notebook_filename)
        print(f"Successfully created: {zip_output_path}")
    else:
        print(f"Error: Could not find '{notebook_filename}' in {current_dir}.")
        print("Please check if the notebook name and the current directory are correct.")
    return


if __name__ == "__main__":
    app.run()
