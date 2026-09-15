# 🏠 House Price Prediction using Machine Learning

An end-to-end machine learning project for predicting residential house prices using the Ames Housing dataset.

The project covers the complete machine learning workflow, from data exploration and preprocessing to model comparison, hyperparameter tuning, feature importance, error analysis and API deployment.

---

## 📌 Project Overview

The goal of this project is to build a machine learning system that can estimate the selling price of a residential property based on its characteristics.

The dataset contains information such as:

- Overall house quality
- Living area
- Lot area
- Garage information
- Basement information
- Year built
- Number of rooms
- Neighborhood
- Other property characteristics

---

## 📊 Dataset

The project uses the Ames Housing dataset.

- **Records:** 1,460
- **Original features:** 81
- **Target variable:** `SalePrice`
- **Problem type:** Regression

The target variable, `SalePrice`, represents the selling price of the house.

---

## 🔎 Exploratory Data Analysis

The project includes exploratory analysis of:

- Dataset structure
- Numerical and categorical variables
- Missing values
- Target distribution
- Feature distributions
- Skewness
- Relationships between important features and house prices

The `SalePrice` distribution was positively skewed, so target transformations were evaluated during model development.

---

## ⚙️ Data Preprocessing

The preprocessing workflow includes:

### Numerical Features

- Missing values are handled using median imputation.
- Numerical features are passed through the preprocessing pipeline.

### Categorical Features

- Missing values are handled using most-frequent imputation.
- Categorical variables are converted using one-hot encoding.

### Sparse Features

The following columns were removed because they contained a very high percentage of missing values:

- `PoolQC`
- `MiscFeature`
- `Alley`
- `Fence`

The `Id` column was also removed because it is an identifier rather than a meaningful predictive feature.

---

## 🔄 Target Transformation

Two target transformations were evaluated:

- Log transformation
- Yeo-Johnson transformation

The transformations were compared using cross-validation performance.

The log transformation produced the stronger overall results and was selected for further model development.

---

## 🤖 Models Evaluated

### Baseline Models

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Advanced Models

- XGBoost
- LightGBM
- CatBoost
- Extra Trees
- HistGradientBoosting
- AdaBoost

The models were compared using 5-fold cross-validation.

---

## 🏆 Model Selection and Hyperparameter Tuning

The strongest advanced candidates were selected for further tuning.

Hyperparameter optimization was performed using:

- RandomizedSearchCV
- GridSearchCV

RandomizedSearchCV was used to efficiently explore a wider hyperparameter space, followed by more focused grid searches.

CatBoost produced the strongest cross-validation performance among the tuned candidates and was selected as the final model.

---

## 📈 Final Model Performance

The final CatBoost model was evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| **R²** | **0.9102** |
| **MAE** | **~$15.2K** |
| **RMSE** | **~$26.2K** |

### What these metrics mean

**R²:** Measures how much of the variation in house prices is explained by the model.

**MAE:** Represents the average absolute difference between actual and predicted prices.

**RMSE:** Gives more weight to larger prediction errors.

The test set was kept separate from model selection and hyperparameter tuning and was used for final evaluation.

---

## 🔍 Feature Importance

Feature-importance analysis showed that the following features were among the strongest price drivers:

- Overall Quality
- Living Area
- Garage Capacity
- Basement Area
- Lot Area
- Year Built
- Year Remodeled

`OverallQual` was the most influential feature in the final model.

---

## 🧪 Error Analysis

The project also examines individual prediction errors and groups prediction performance by price range.

The model performs well overall but shows larger errors for some high-priced and unusual properties.

This highlights an important limitation of the current model and provides direction for future feature engineering and model improvement.

---

## 🚀 API Deployment

The trained machine learning pipeline is saved using Joblib and exposed through a FastAPI application.

### Available endpoints

#### `GET /`

Checks whether the API is running.

#### `POST /predict`

Accepts house features and returns the predicted sale price.

Example response:

```json
{
    "predicted_sale_price": 250000.00
}