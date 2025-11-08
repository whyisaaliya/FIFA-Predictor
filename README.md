
# FIFA World Cup Predictor

This project uses historical match data and team rankings to build and evaluate machine learning models (Logistic Regression and Random Forest) capable of predicting the outcome of FIFA World Cup matches. The project includes data cleaning, extensive feature engineering, model training, and a final tournament simulation.

## 🗂️ Table of Contents

  * [Data Sources](https://www.google.com/search?q=%23-data-sources)
  * [Data Cleaning & Preparation](https://www.google.com/search?q=%23-data-cleaning--preparation)
  * [Feature Engineering & Selection](https://www.google.com/search?q=%23-feature-engineering--selection)
  * [Model Training](https://www.google.com/search?q=%23-model-training)
  * [Model Evaluation & Results](https://www.google.com/search?q=%23-model-evaluation--results)
  * [Tournament Simulation Results](https://www.google.com/search?q=%23-tournament-simulation-results)
  * [Core Modules Used](https://www.google.com/search?q=%23-core-modules-used)

-----

## 💾 Data Sources

The models were trained using three primary data sources:

  * **`matches_1930_2022.csv`**: Contains historical match data, including home/away teams, scores, tournament round, attendance, and (for later years) expected goals (xG).
  * **`fifa_ranking_2022-10-06.csv`**: Provides the official FIFA ranking and points for each national team as of October 2022.
  * **`transfermarkt_team_stats_all-time.csv`**: A supplementary file (loaded via `try...except`) intended to provide additional team statistics.

-----

## 🧹 Data Cleaning & Preparation

Before feature engineering, the raw data was processed as follows:

  * **Filtering**: The match dataset was filtered to include only matches from **1998 onwards**.
  * **Type Conversion**: Columns such as `home_score`, `away_score`, `home_xg`, `away_xg`, and `Attendance` were converted to numeric types to ensure they could be used in calculations.
  * **Missing Data**: Duplicate rows were dropped, and empty strings were replaced with `NA` values to be handled during the model's `dropna` step.

-----

## 🛠️ Feature Engineering & Selection

This was a critical phase to create a predictive feature set.

### Target Variable

The model predicts a binary outcome. The target variable, **`result`**, was engineered as:

  * **`1`**: If the **home team wins** (`home_score > away_score`).
  * **`0`**: If the **away team wins or the match is a draw** (`home_score <= away_score`).

### Engineered Features

Several new features were created by combining the datasets:

  * **`rank_diff`**: The difference between the home team's and away team's FIFA rank (`home_rank - away_rank`).
  * **`points_diff`**: The difference between the home team's and away team's FIFA points (`home_points - away_points`).
  * **`is_host`**: A binary flag (1 if the home team is also the host nation, 0 otherwise).
  * **`knockout`**: A binary flag (1 if the match `Round` contains "Final", "Semi", "Quarter", or "Round of 16", 0 otherwise).

### Final Feature Set

The models were trained on the following 7 features:

1.  `rank_diff`
2.  `points_diff`
3.  `home_xg`
4.  `away_xg`
5.  `Attendance`
6.  `is_host`
7.  `knockout`

-----

## 🤖 Model Training

The data was split into an **80% training set** and a **20% test set**. Two different models were trained.

### 1\. Logistic Regression

  * **Preprocessing**: The training data (`X_train`) was first scaled using **`StandardScaler`**. The test data (`X_test`) was transformed using the same scaler.
  * **Training**: A `LogisticRegression` model (with `max_iter=1000`) was fit on the **scaled** training data.

### 2\. Random Forest

  * **Preprocessing**: This model was trained on the **original, unscaled** feature data, as tree-based models are not sensitive to feature scaling.
  * **Training**: A `RandomForestClassifier` (with `n_estimators=200`) was fit on the unscaled training data.

-----

## 📊 Model Evaluation & Results

Both models were evaluated on the held-out test set.

### Overall Performance (ROC AUC)

Both models demonstrated strong and nearly identical discriminatory power, achieving an **Area Under the Curve (AUC) of 0.88**. This indicates a high probability that the models will rank a randomly chosen positive instance (Home Win) higher than a randomly chosen negative instance (Away Win/Draw).

### Classification Reports

The **Logistic Regression** model showed slightly better overall performance, with a higher accuracy (80%) and weighted F1-score (0.81) compared to the Random Forest (76% accuracy, 0.78 F1-score).

**Logistic Regression Metrics:**

  * **Accuracy**: 80%
  * **Weighted Avg Precision**: 0.85
  * **Weighted Avg Recall**: 0.80
  * **Weighted Avg F1-Score**: 0.81

<!-- end list -->

```
=== Logistic Regression ===
              precision    recall  f1-score   support

           0       0.94      0.79      0.86        19
           1       0.56      0.83      0.67         6

    accuracy                           0.80        25
   macro avg       0.75      0.81      0.76        25
weighted avg       0.85      0.80      0.81        25
```

**Random Forest Metrics:**

  * **Accuracy**: 76%
  * **Weighted Avg Precision**: 0.83
  * **Weighted Avg Recall**: 0.76
  * **Weighted Avg F1-Score**: 0.78

<!-- end list -->

```
=== Random Forest ===
              precision    recall  f1-score   support

           0       0.93      0.74      0.82        19
           1       0.50      0.83      0.62         6

    accuracy                           0.76        25
   macro avg       0.72      0.79      0.72        25
weighted avg       0.83      0.76      0.78        25
```

### Random Forest Feature Importance

The feature importance plot from the Random Forest model clearly shows which factors had the biggest impact on its decisions.

As shown, the model's predictions are overwhelmingly driven by:

1.  **`points_diff` (0.237)**
2.  **`rank_diff` (0.228)**

The expected goals (`home_xg`, `away_xg`) and `Attendance` provided moderate value. The `is_host` and `knockout` flags had almost no importance in the final model.

-----

## 🏆 Tournament Simulation Results

A tournament simulation was run based on the trained models to predict a final winner. The last three rounds of the simulation are shown below.

Based on the simulation:

  * **Quarter-Finals (implied)**: Germany, England, and Argentina advanced (among others).
  * **Semi-Finals (Round 5)**: England defeated Germany to advance to the final.
  * **Final (Round 6)**: **Argentina** defeated England.

The final prediction of this model is that **Argentina** is the tournament winner.

-----

## 📦 Core Modules Used

  * **`pandas`**: For loading, cleaning, and manipulating all tabular data.
  * **`sklearn`**: The primary library for machine learning, used for:
      * `model_selection.train_test_split`: Splitting data.
      * `preprocessing.StandardScaler`: Scaling features for Logistic Regression.
      * `linear_model.LogisticRegression`: The Logistic Regression model.
      * `ensemble.RandomForestClassifier`: The Random Forest model.
      * `metrics`: Generating `classification_report`, `confusion_matrix`, and AUC scores.
  * **`joblib`**: Used to serialize and save the final trained models (`.pkl` files) and the `scaler` for future use.
  * **`os`**: To create the `models/` directory for storing saved files.


<img width="800" height="600" alt="ROC_AUC" src="https://github.com/user-attachments/assets/0b69ff5a-1bb4-45c3-8ad2-a78989e7be03" />
<img width="800" height="500" alt="RAndom_Forest_featureIMp" src="https://github.com/user-attachments/assets/97243ceb-ed86-441d-9e69-5805b013fa37" />


  

