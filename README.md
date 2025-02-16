# 🏪 BigMart Sales Prediction  

## 📌 Problem Statement  
The goal of this project is to predict sales of various products in different outlets of **BigMart** using historical sales data. The dataset contains information about the products, stores, and sales, and the objective is to build a predictive model that can estimate sales based on these features.  

## 📂 Dataset  

The dataset is from an **Analytics Vidhya Hackathon** and consists of two files:  

### 🏋️ Train Dataset (`train.csv`)  
- Contains **8,523 records** with both **input** and **output variables** (sales values).  
- The model is trained on this dataset.  

| Variable                  | Description |
|---------------------------|-------------|
| **Item_Identifier**       | Unique product ID |
| **Item_Weight**           | Weight of product |
| **Item_Fat_Content**      | Whether the product is low fat or not |
| **Item_Visibility**       | % of total display area of all products in a store allocated to the particular product |
| **Item_Type**             | The category to which the product belongs |
| **Item_MRP**              | Maximum Retail Price (list price) of the product |
| **Outlet_Identifier**     | Unique store ID |
| **Outlet_Establishment_Year** | The year in which the store was established |
| **Outlet_Size**           | The size of the store in terms of ground area covered |
| **Outlet_Location_Type**  | The type of city in which the store is located |
| **Outlet_Type**           | Whether the outlet is just a grocery store or some sort of supermarket |
| **Item_Outlet_Sales**     | 🔥 Sales of the product in the particular store (Target Variable) 🔥 |

### 🛒 Test Dataset (`test.csv`)  
- Contains **5,681 records** without sales values.  
- The model will predict sales for this dataset.  

| Variable                  | Description |
|---------------------------|-------------|
| **Item_Identifier**       | Unique product ID |
| **Item_Weight**           | Weight of product |
| **Item_Fat_Content**      | Whether the product is low fat or not |
| **Item_Visibility**       | % of total display area of all products in a store allocated to the particular product |
| **Item_Type**             | The category to which the product belongs |
| **Item_MRP**              | Maximum Retail Price (list price) of the product |
| **Outlet_Identifier**     | Unique store ID |
| **Outlet_Establishment_Year** | The year in which the store was established |
| **Outlet_Size**           | The size of the store in terms of ground area covered |
| **Outlet_Location_Type**  | The type of city in which the store is located |
| **Outlet_Type**           | Whether the outlet is just a grocery store or some sort of supermarket |

The goal is to train the model using `train.csv` and **predict sales** for `test.csv`. 🚀  

---

## 🔍 Approach  

### 1️⃣ Data Preprocessing  
- Loaded the dataset and checked for missing values.  
- **Item_Weight** (numerical) was imputed using the **median** value.  
- **Outlet_Size** (categorical) was imputed using **RandomForestClassifier** and **Iterative Imputer** based on dependent features:  
  - Outlet_Identifier  
  - Outlet_Establishment_Year  
  - Outlet_Location_Type  
  - Outlet_Type  
- Applied **label encoding** and **one-hot encoding** to categorical variables for better model performance.  

### 2️⃣ Exploratory Data Analysis (EDA)  
- Standardized categorical values (e.g., converted different forms of "low fat" to a common label).  
- Visualized sales trends across different outlet types and product categories.  

### 3️⃣ Feature Engineering  
- Derived new features to enhance model learning.  

### 4️⃣ Model Selection & Training  
- Used a **Multi-Layered Perceptron (MLP)** neural network to predict sales.   
- Evaluated model performance using **Root Mean Squared Error (RMSE)** and other metrics.  

## 🚀 Installation & Usage  
To run this project on your local machine:  

```bash
# Clone the repository
git clone https://github.com/yourusername/BigMart-Sales-Prediction.git

# Navigate to the project directory
cd BigMart-Sales-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook BigMart_Sales_Prediction.ipynb
```
## 📊 Results & Insights
✅ The MLP model successfully predicts sales with a reasonable accuracy.<br>
✅ Outlet type and item visibility are key features influencing sales.<br>
✅ Further feature engineering can enhance performance
