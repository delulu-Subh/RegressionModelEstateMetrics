# 🏡 Housing Price Prediction using Linear Regression with Regularization

Predicting housing prices is a critical task in real estate analytics. This project implements Linear Regression with L1 (Lasso) and L2 (Ridge) regularization to predict house prices based on various features. The model is tuned for optimal performance and evaluated using standard regression metrics.

📊 Project Overview

This project leverages a dataset of housing features to predict the price of houses. The key steps include:

Data Preprocessing

Selected numerical and categorical features

One-hot encoding for categorical variables (Address)

Standardization of numerical features

Modeling

Implemented Ridge Regression (L2 regularization)

Implemented Lasso Regression (L1 regularization)

Hyperparameter tuning using different alpha values for best regularization

Evaluation

Regression metrics:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Visual comparison of actual vs predicted prices

🛠️ Dataset

The dataset contains 5000 housing records with the following features:

Feature	Description
Avg. Area Income	Average income of residents in the area
Avg. Area House Age	Average age of houses in the area
Avg. Area Number of Rooms	Average number of rooms per house
Avg. Area Number of Bedrooms	Average number of bedrooms per house
Area Population	Population of the area
Address	House address (encoded using One-Hot)
Price	House price (target variable)
⚡ Installation
# Clone the repository
git clone https://github.com/yourusername/HousingPricePrediction.git

# Navigate to the project directory
cd HousingPricePrediction

# Install required packages
pip install -r requirements.txt

🧠 Modeling
Ridge Regression (L2)
from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=10)
ridgeReg.fit(x_train, y_train)
y_pred = ridgeReg.predict(x_test)

Lasso Regression (L1)
from sklearn.linear_model import Lasso
lassoReg = Lasso(alpha=10)
lassoReg.fit(x_train, y_train)
y_pred_lasso = lassoReg.predict(x_test)

<img width="547" height="446" alt="EstateMetrics" src="https://github.com/user-attachments/assets/3d5a9ede-f267-4707-b9b5-ae53b853a1a8" />

Performance Metrics
Metric	Ridge Regression
MAE	82,553
MSE	10,540,961,946
RMSE	102,669
R²	0.915
📈 Results Visualization
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()


The plot shows that the predicted prices closely follow the actual prices, indicating high model accuracy.

🧩 Key Takeaways

Regularization prevents overfitting and improves model generalization.

Ridge Regression performed slightly better on this dataset.

Hyperparameter tuning for alpha is essential to balance bias-variance tradeoff.

Strong correlation between features and target allows for reliable predictions.

🔮 Future Enhancements

Explore ElasticNet Regression (combining L1 + L2)
Implement feature selection to reduce dimensionality.
Use cross-validation for more robust hyperparameter tuning.
Deploy the model as a web application for real-time house price predictions.
