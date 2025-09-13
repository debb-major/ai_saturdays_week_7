# AI Saturdays Week 7 Assessment

This repository contains my Week 7 Machine Learning assessment.  
The task was to build **classification and regression models** using non-linear algorithms, perform **hyperparameter tuning** 
with cross-validation, and compare results to linear models.


## Contents
- `Week7_Assessment.ipynb`: Jupyter Notebook containing both parts of the assessment, inside the `notebook` folder.
- `Titanic-Dataset.csv`: Classification dataset (Titanic survival prediction), inside the `data` folder.
- `housing_data.csv`: Regression dataset (Housing prices prediction), inside the `data` folder.
- `requirements.txt`: Contains project dependencies.


## Part A: Classification (Titanic Dataset)
- **Algorithms Used:**
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
- **Tuning Methods:**
  - GridSearchCV (with 5-fold cross-validation)
  - RandomizedSearchCV (with 5-fold cross-validation)
- **Evaluation Metrics:**
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

**Model Comparison:**

| Model                          | Accuracy |
|--------------------------------|----------|
| Baseline KNN                   | 81%      |
| Tuned KNN (GridSearchCV)       | 79%      |
| Tuned KNN (RandomizedSearchCV) | 79%      |
| Decision Tree                  | 76%      |

## Part B: Regression (Housing Prices Dataset)
- **Algorithms Used:**
  - Decision Tree Regressor
  - Random Forest Regressor
- **Tuning Methods:**
  - GridSearchCV (with 5-fold cross-validation)
  - RandomizedSearchCV (with 5-fold cross-validation)
- **Evaluation Metrics:**
  - R² Score
  - Root Mean Squared Error (RMSE)

## Observations & Comparison
- Linear models (from earlier projects) gave reasonable results but struggled with complex relationships in the data.  
- Non-linear models (KNN, Decision Trees, Random Forest) performed better, with higher accuracy in classification and higher R²/lower RMSE in regression.  
- This improvement comes from their ability to capture **non-linear relationships and feature interactions** that linear models miss.  
- However, non-linear models require **hyperparameter tuning** and careful validation to avoid overfitting.


## Conclusion
- **Classification:** Non-linear models improved survival prediction accuracy compared to linear models.  
- **Regression:** Non-linear regressors, especially Random Forest, provided better performance in predicting housing prices.  
- **Cross-Validation:** Used within GridSearchCV and RandomizedSearchCV ensured robust evaluation of tuned models.  


## How to Run

1. **Clone this repository**
     
   ```bash
   git clone www.github.com/debb-major/ai_saturdays_week_7.git
   cd ai_saturdays_week_7
    ```
   
2. **Install dependencies**
   
```
pip install -r requirements.txt
```
  *If you don’t have a `requirements.txt`, install the basics manually:*
  ```
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

3. **Open the notebook**
   
   ```
   jupyter notebook Week7_Assessment.ipynb
   ```
   
4. **Run all cells**

   Make sure `Titanic-Dataset.csv` and `housing_data.csv` paths are mapped correctly in the notebook.
