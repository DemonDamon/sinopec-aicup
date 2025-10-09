# Research Log: Time Series Forecasting Competition Strategies


## Initial Search Results

### LightGBM vs XGBoost for Time Series
*   **Title**: LightGBM vs XGBoost for time series analysis
    *   **Snippet**: In this kernel I will compare two the most popular Machine Learning algorithms of the past few years XGBoost (XGB) and LightGBM (LGB) in time series analysis ...
    *   **URL**: https://www.kaggle.com/code/nowakjakub/lightgbm-vs-xgboost-for-time-series-analysis
*   **Title**: XGBoost vs LightGBM: How Are They Different
    *   **Snippet**: LightGBM is significantly faster than XGBoost but delivers almost equivalent performance. We might wonder, what are exactly the differences between LightGBM and ...
    *   **URL**: https://neptune.ai/blog/xgboost-vs-lightgbm
*   **Title**: XGBoost & LGBM for Time Series Forecasting
    *   **Snippet**: One of the main differences between these two algorithms, however, is that the LGBM tree grows leaf-wise, while the XGBoost algorithm tree grows depth-wise: A ...
    *   **URL**: https://365datascience.com/tutorials/python-tutorials/xgboost-lgbm/
*   **Title**: Why XGBoost Champions Are Switching to LightGBM (And ...
    *   **Snippet**: The gradient boosting revolution that's quietly dominating Kaggle leaderboards and production systems worldwide.
    *   **URL**: https://medium.com/@matiasmaquieira96/why-xgboost-champions-are-switching-to-lightgbm-and-you-should-too-334fa37ee68c
*   **Title**: Comparative Analysis of Modern Machine Learning ...
    *   **Snippet**: Our evaluation demonstrates that ensemble methods, particularly LightGBM and XGBoost, outperform complex neural network architectures in terms ...
    *   **URL**: https://arxiv.org/html/2506.05941v1
*   **Title**: Which algorithm takes the crown: Light GBM vs XGBOOST?
    *   **Snippet**: Light GBM is almost 7 times faster than XGBOOST and is a much better approach when dealing with large datasets. This turns out to be a huge ...
    *   **URL**: https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

### Time Series Cross-Validation Strategies
*   **Title**: Avoiding Data Leakage in Cross-Validation | by Silva.f.francis
    *   **Snippet**: One of the major causes of data leakage is performing preprocessing steps like scaling or feature selection before splitting the data into ...
    *   **URL**: https://medium.com/@silva.f.francis/avoiding-data-leakage-in-cross-validation-ba344d4d55c0
*   **Title**: What is Data Leakage in Machine Learning?
    *   **Snippet**: Handle time-series data with care, using methods such as rolling window validation or walk-forward validation to avoid leakage from future data during training.
    *   **URL**: https://www.ibm.com/think/topics/data-leakage-machine-learning
*   **Title**: Time Series Cross-Validation: Best Practices
    *   **Snippet**: In this article, we'll cover: Rolling origin and time-aware validation methods; Avoiding temporal data leakage; Handling seasonality and special ...
    *   **URL**: https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1
*   **Title**: Preventing data leakage in time-series data splitting
    *   **Snippet**: Well, typically you perform repeated cross-validation to improve your performance estimation (and reduce the uncertainty of it). Now, when ...
    *   **URL**: https://stats.stackexchange.com/questions/658883/preventing-data-leakage-in-time-series-data-splitting
*   **Title**: Data Leakage in Time Series Data Cross-Validations in ...
    *   **Snippet**: Data leakage occurs when information from the future leaks into the past during the training and validation process of time series models.
    *   **URL**: https://hectorv.com/2023/07/06/data-leakage-in-time-series-data-cross-validations-in-machine-learning/
*   **Title**: 3.1. Cross-validation: evaluating estimator performance
    *   **Snippet**: If one knows that the samples have been generated using a time-dependent process, it is safer to use a time-series aware cross-validation scheme.
    *   **URL**: https://scikit-learn.org/stable/modules/cross_validation.html
*   **Title**: Preventing Data Leakage in Feature Engineering - dotData
    *   **Snippet**: Time-based cross-validation divides the dataset into multiple non-overlapping time periods or windows. The model is trained on the initial ...
    *   **URL**: https://dotdata.com/blog/preventing-data-leakage-in-feature-engineering-strategies-and-solutions/

### Why Gradient Boosting Models are Strong for Tabular Time Series
*   **Title**: Gradient Boosting: a Silver Bullet in Forecasting
    *   **Snippet**: Gradient boosting is a machine learning technique that builds predictive models by combining an ensemble of weak learners in a sequential manner.
    *   **URL**: https://medium.com/data-science/gradient-boosting-a-silver-bullet-in-forecasting-5820ba7182fd
*   **Title**: Forecasting with gradient boosted trees: augmentation ...
    *   **Snippet**: The ability of these methods to capture feature interactions and nonlinearities makes them exceptionally powerful and, at the same time, prone to overfitting, ...
    *   **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0169207021002090
*   **Title**: Is Gradient Boosting good as Prophet for Time Series ...
    *   **Snippet**: We discovered that, for our use case, gradient boosting outperformed Prophet overwhelmingly in terms of test errors.
    *   **URL**: https://towardsdatascience.com/is-gradient-boosting-good-as-prophet-for-time-series-forecasting-3dcbfd03775e/
*   **Title**: XGBoost Is The Best Algorithm for Tabular Data
    *   **Snippet**: XGBoost is suitable for tabular data because it produces skillful models quickly, outperforms deep models, and is preferred over neural networks on average.
    *   **URL**: https://xgboosting.com/xgboost-is-the-best-algorithm-for-tabular-data/
*   **Title**: Why do tree-based models still outperform deep learning ...
    *   **Snippet**: Tree-based models remain state-of-the-art on medium-sized tabular data, even without accounting for their superior speed. Deep learning's ...
    *   **URL**: https://arxiv.org/abs/2207.08815
*   **Title**: Use XGBoost for Time-Series Forecasting
    *   **Snippet**: XGBoost is a powerful algorithm for time-series forecasting, offering several advantages such as handling non-linear relationships, feature ...
    *   **URL**: https://www.analyticsvidhya.com/blog/2024/01/xgboost-for-time-series-forecasting/

### LightGBM/XGBoost Time Series Forecasting Examples & Tutorials
*   **Title**: [Tutorial] Time Series forecasting with XGBoost
    *   **Snippet**: In this notebook we will walk through time series forecasting using XGBoost. The data we will be using is hourly energy consumption.
    *   **URL**: https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost
*   **Title**: Time Series Forecasting with XGBoost and LightGBM
    *   **Snippet**: In this article, we will take a quick but practical look at how this is done by incorporating Ensemble models such as extreme gradient boosting or XGBoost and ...
    *   **URL**: https://medium.com/mlearning-ai/time-series-forecasting-with-xgboost-and-lightgbm-predicting-energy-consumption-460b675a9cee
*   **Title**: XGBoost & LGBM for Time Series Forecasting
    *   **Snippet**: In this tutorial, we will go over the definition of gradient boosting, look at the two algorithms, and see how they perform in Python.
    *   **URL**: https://365datascience.com/tutorials/python-tutorials/xgboost-lgbm/
*   **Title**: Forecasting with XGBoost and LightGBM - Skforecast Docs
    *   **Snippet**: Gradient boosting models like XGBoost and LightGBM are effective for forecasting, handling non-linear relationships and large data, but have limitations in ...
    *   **URL**: https://skforecast.org/0.13.0/user_guides/forecasting-xgboost-lightgbm
*   **Title**: Mastering Time Series Forecasting with LightGBM
    *   **Snippet**: In this article, we'll walk step-by-step through building a time series prediction model using LightGBM. From data preparation to model training ...
    *   **URL**: https://medium.com/data-science-collective/mastering-time-series-forecasting-with-lightgbm-a-practical-guide-2dff8d1a72bb

### Implementation Roadmap for Time Series Forecasting Competitions
*   **Title**: Time Series Analysis and Forecasting
    *   **Snippet**: It supports time series forecasting tasks and provides tools for data preprocessing, feature engineering, model selection, and evaluation in a ...
    *   **URL**: https://www.geeksforgeeks.org/machine-learning/time-series-analysis-and-forecasting/
*   **Title**: Feature Engineering Techniques For Time Series Data
    *   **Snippet**: In this article, we will look at various feature engineering techniques for extracting useful information using the date-time column.
    *   **URL**: https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/
*   **Title**: Automated Time Series Feature Engineering with MLforecast
    *   **Snippet**: Target transformations improve forecasting accuracy by preprocessing the target variable. For example, differencing transforms trending ...
    *   **URL**: https://www.nixtla.io/blog/automated-time-series-feature-engineering-with-mlforecast
*   **Title**: Time Series Preprocessing and Feature Engineering
    *   **Snippet**: In this series, we will introduce you to time series modeling – the act of building predictive models on time series data.
    *   **URL**: https://h2o.ai/blog/2021/an-introduction-to-time-series-modeling-time-series-preprocessing-and-feature-engineering/
*   **Title**: Practical Guide for Feature Engineering of Time Series Data
    *   **Snippet**: Learn how to enhance your time series forecasting models with effective feature engineering techniques. Discover the power of lagged ...
    *   **URL**: https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/


### Detailed Findings from URLs (Phase 2, Round 1)

#### Validation Strategies for Time Series
*   **From: Time Series Cross-Validation: Best Practices [[ref]](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1)**
    *   **Time-based Splitting**: This method divides data chronologically into training, validation, and test sets. It explicitly states to **respect the order** of time to prevent data leakage. Future data must not be used to train models predicting the past. A common split could be 70% for training, 15% for validation, and 15% for testing, always respecting the time order.
    *   **Rolling Origin/Walk-Forward Validation**: This technique involves an initial training period, followed by predicting the next time step(s). Then, the training window is rolled forward, incorporating the observed data from the previous prediction step into the new training set. This process is repeated. It simulates a real-world scenario more accurately by only using historical data to make future predictions. It helps assess model stability over time and its robustness to changing data dynamics.
    *   **Stratified Time Series Split**: When dealing with seasonal data, it suggests using a strategy to ensure that each fold contains a proportional representation of all seasons, even while maintaining the temporal order. (Though the focus is more on basic time series splits for this prompt, it's a good advanced note).

*   **From: Data Leakage in Time Series Data Cross-Validations in Machine Learning [[ref]](https://hectorv.com/2023/07/06/data-leakage-in-time-series-data-cross-validations-in-machine-learning/)**
    *   **Definition of Data Leakage**: Occurs when information from the future leaks into the past during model training and validation. This leads to overly optimistic performance estimates.
    *   **Failure of Standard Cross-Validation**: Random shuffling in traditional k-fold or stratified sampling is not suitable for time series data due to its temporal dependence. Shuffling breaks the chronological order, leading to data leakage.
    *   **Validation Methods Emphasized**: Reaffirms Time-Based Validation (simple train-test split respecting time), Rolling Window Validation (a form of walk-forward, where the training window moves over time and predictions are made for a fixed forecast horizon), and Walk-Forward Validation (similar to rolling window but emphasizes continuous updates of the training data).

*   **From: 3.1. Cross-validation: evaluating estimator performance (Scikit-learn) [[ref]](https://scikit-learn.org/stable/modules/cross_validation.html)**
    *   **TimeSeriesSplit**: Scikit-learn provides `TimeSeriesSplit`, which is a specialized cross-validation iterator. It generates train/test indices where test sets are always subsequent to training sets. The training set is a prefix of the full data, and the test set is a split further down the timeline. The length of the training window can increase in each split (default) or remain constant (sliding window with fixed size). This is a crucial tool for implementing time-series-aware cross-validation.

#### Why Gradient Boosting Models (LightGBM & XGBoost) are Strong Baselines for Time Series
*   **From: Gradient Boosting: a Silver Bullet in Forecasting [[ref]](https://medium.com/data-science/gradient-boosting-a-silver-bullet-in-forecasting-5820ba7182fd)**
    *   **Ensemble Method**: Gradient Boosting combines multiple 

weak learners (typically decision trees) sequentially. Each new model corrects the errors of the previous ones, leading to improved overall prediction accuracy. This iterative error correction is a key strength for complex forecasting tasks.
    *   **Versatility**: Can handle various types of data and relationships, making it adaptable to different time series characteristics (trends, seasonality, etc.).

*   **From: Why do tree-based models still outperform deep learning on tabular data? [[ref]](https://arxiv.org/abs/2207.08815)**
    *   **State-of-the-Art for Tabular Data**: Tree-based models (including Gradient Boosting Machines like XGBoost and LightGBM) remain superior to deep learning on medium-sized tabular datasets, often with better speed.
    *   **Categorical Feature Handling**: Tree-based models are inherently better at handling categorical features, which are often derived in time series feature engineering (e.g., day of week, month).
    *   **Robustness**: Less sensitive to hyperparameter tuning than deep learning models.
    *   **Missing Data Handling**: Can natively handle missing data, a common issue in real-world datasets.

*   **From: Use XGBoost for Time-Series Forecasting [[ref]](https://www.analyticsvidhya.com/blog/2024/01/xgboost-for-time-series-forecasting/)**
    *   **Advantages for Time Series**: XGBoost can effectively handle non-linear relationships and complex patterns often found in time series data.
    *   **Feature Importance**: Provides insights into which features (e.g., lagged values, time-based features) are most influential in predictions.
    *   **Robustness to Missing Values**: Can handle missing data during training.
    *   **Flexibility**: Adaptable to various time series forecasting scenarios.
    *   **Scalability**: Designed for efficiency and scalability on large datasets.
    *   **Overfitting Control**: Incorporates regularization (L1, L2) to prevent overfitting.
    *   **Relevant Feature Engineering for Time Series**: Emphasizes lagged features (past observations as predictors), rolling window statistics (e.g., moving averages, standard deviations over a time window), and date/time features (e.g., year, month, day, day of week, hour, quarter).
    *   **Evaluation Metrics**: Mentions MAE, RMSE, and MAPE as common metrics for time series forecasting. These are standard in competitions.


### Implementation Roadmap and Validation Strategy (Phase 2, Round 2)

#### 1. Data Preprocessing and Feature Engineering
Based on research from various sources, especially [[ref]](https://h2o.ai/blog/2021/an-introduction-to-time-series-modeling-time-series-preprocessing-and-feature-engineering/) and [[ref]](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/), a comprehensive approach for time-series competitions includes:

*   **Data Loading & Initial Structuring**:
    *   Ensure data is loaded with a proper time index, and records are sorted chronologically.
    *   Handle missing timestamps if the series is expected to be continuous.
*   **Missing Value Handling**:
    *   Common techniques include forward-fill (`ffill`), backward-fill (`bfill`), mean imputation, or median imputation. The choice depends on the nature of the data and the missingness pattern.
*   **Outlier Detection and Treatment**:
    *   Identify and appropriately handle extreme values that could skew model training.
*   **Time-Based Feature Engineering**:
    *   **Lagged Features**: Create new features representing past observations of the target variable or other relevant predictors (e.g., sales from 1 day, 7 days, 28 days ago). These are crucial for capturing temporal dependencies [[ref]](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/).
    *   **Rolling Window Statistics**: Calculate aggregate statistics over defined past periods (e.g., moving average, standard deviation, minimum, maximum, sum) for the target and other features. This helps capture trends and volatility [[ref]](https://h2o.ai/blog/2021/an-introduction-to-time-series-modeling-time-series-preprocessing-and-feature-engineering/).
    *   **Date/Time Components**: Extract granular features from the timestamp, such as year, month, day, day of week, day of year, week of year, hour, and specific flags like `is_weekend`, `is_month_start`, `is_holiday`. These capture cyclical patterns and seasonality [[ref]](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/).
    *   **External Features**: Incorporate external data that may influence the time series, such as weather conditions, economic indicators, or holiday calendars. This often requires careful alignment with the time index.
*   **Target Variable Transformations**:
    *   Apply transformations like logarithmic or power transforms if the target variable has a skewed distribution or large variance. Differencing the target can also be beneficial if the series is non-stationary, though tree-based models are often more robust to this.

#### 2. Model Training and Hyperparameter Tuning
*   **Model Selection**: Focus on Gradient Boosting models, specifically LightGBM and XGBoost, due to their strong performance on tabular and time-series data.
*   **Training Process**: Models are trained on the engineered features. The iterative nature of gradient boosting helps capture complex non-linear relationships and interactions among features.
*   **Hyperparameter Tuning**: Optimize model parameters (e.g., `n_estimators`, `learning_rate`, `max_depth`, `colsample_bytree`, `subsample`) using techniques like Grid Search, Randomized Search, or Bayesian Optimization. This tuning should ideally be performed within a time-series cross-validation framework to prevent leakage.

#### 3. Validation Strategy for Time-Series Data
A robust validation strategy is critical to prevent data leakage and obtain a reliable estimate of model performance, which directly impacts private leaderboard scores in competitions [[ref]](https://hectorv.com/2023/07/06/data-leakage-in-time-series-data-cross-validations-in-machine-learning/). Standard k-fold cross-validation is inappropriate for time-series data due to temporal dependencies.

*   **Time-Based Splitting**:
    *   This is the simplest form of time-aware validation. The dataset is split chronologically into distinct training, validation, and test sets. For example, use data up to date X for training, data from X+1 to Y for validation, and data from Y+1 to Z for final testing. This strictly prevents future information from influencing past predictions [[ref]](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1).
*   **Time-Series Cross-Validation (Walk-Forward / Rolling Origin)**:
    *   **General Principle**: This method mimics real-world forecasting by sequentially training models on increasing (or fixed-size) historical windows and evaluating them on the subsequent unseen periods. This ensures that the model is only ever trained on data preceding the period it is trying to forecast [[ref]](https://hectorv.com/2023/07/06/data-leakage-in-time-series-data-cross-validations-in-machine-learning/).
    *   **Implementation with `TimeSeriesSplit`**: Scikit-learn's `TimeSeriesSplit` utility is specifically designed for this purpose. It generates folds where the validation set is always chronologically after the training set. The training set size can either expand or remain fixed (sliding window) for each subsequent split [[ref]](https://scikit-learn.org/stable/modules/cross_validation.html).
        *   **Expanding Window (Forward Chaining)**: The training data for each successive fold includes all data from previous folds, leading to an ever-growing training set. This is effective when historical context is consistently relevant.
        *   **Sliding Window**: The training window size remains constant, and the window slides forward by a fixed step for each fold. This is useful when older data might be less relevant due to regime changes or when computational resources are constrained.
*   **Importance for Private Leaderboard**: A robust time-series validation strategy is essential because it provides an honest estimate of the model's out-of-sample performance. Models optimized using improper validation (e.g., random shuffling) will likely overfit to the temporal dependencies present in the public leaderboard data, leading to a significant drop in score on the private leaderboard (which uses later, unseen data). Correct validation ensures the model generalizes well to future periods.

#### 4. Evaluation and Submission File Generation
*   **Evaluation Metrics**: Commonly used metrics for time-series forecasting include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). The competition will specify the primary metric. These metrics should be calculated on the out-of-sample predictions from the validation folds and the final test set.
*   **Prediction Generation**: After training the final model (or ensemble of models) on the full historical training data using the optimized hyperparameters, generate predictions for the competition's submission period. These predictions must align with the specified submission file format (e.g., date, item_id, predicted_value).
*   **Ensembling**: Combining predictions from multiple diverse models (e.g., different Gradient Boosting models, or models with different seeds/feature sets) can often lead to improved robustness and higher scores. This should also be validated with time-series appropriate methods.