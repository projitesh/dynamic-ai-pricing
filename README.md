# 💹 Dynamic AI Pricing Engine for Online Stores

> **MCA Data Science — Final Year Project**
> UPES Dehradun | 2024

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange?style=flat&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple?style=flat&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 About the Project

**Dynamic AI Pricing Engine** is a Machine Learning-powered web application that helps online store managers decide the **optimal price** for their products.

Instead of manually guessing prices, a store manager simply enters 7 product details — competitor price, category, region, season, units sold, discount, and month — and the AI model instantly recommends the best price to **maximize revenue**.

### 🎯 Problem It Solves
- Competitor prices change daily — manual tracking is impossible
- Seasonal demand and promotions affect sales but are ignored in manual pricing
- Price Elasticity is never measured by small stores
- Revenue is lost to guesswork — prices set by intuition, not data

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Overview** | KPI cards — Total Revenue, Avg Price, Units Sold, Top Category + 4 charts |
| 📊 **Data Analysis** | 2 tabs — Pricing Analysis (price comparison, scatter, heatmap) + Sales Analysis |
| 📉 **Price Elasticity** | Elasticity table + 3 charts by Category, Region, and Season |
| 🤖 **AI Price Predictor** | 7-input form → Random Forest model → Optimal price recommendation |
| 💰 **Revenue Simulator** | Revenue curve + Optimal price finder + Discount × Elasticity matrix |

---

## 📊 Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| **R² Score** | `0.9882` | Model explains 98.82% of price variation |
| **MAE** | `₹2.43` | Average prediction error of only ₹2.43 |
| **RMSE** | `₹2.85` | Even large errors are very small |
| **Training Rows** | `58,480` | 80% of dataset |
| **Test Rows** | `14,620` | 20% of dataset |

---

## 🗂️ Dataset Overview

| Property | Value |
|----------|-------|
| Total Records | 73,100 |
| Date Range | Jan 2022 – Dec 2023 |
| Features | 15 original + 4 engineered |
| Stores | 5 (S001 – S005) |
| Products | 20 (P0001 – P0020) |
| Categories | Electronics, Clothing, Furniture, Groceries, Toys |
| Regions | North, South, East, West |
| Missing Values | Zero |

---

## 🧠 ML Model — Random Forest Regressor

```
Algorithm    : Random Forest Regressor
Trees        : 100 Decision Trees
Max Depth    : 10
Train Split  : 80% (58,480 rows)
Test Split   : 20% (14,620 rows)
Random State : 42
```

### 7 Input Features (X)
```
1. Competitor Pricing  → most important signal
2. Units Sold          → demand level
3. Discount            → promotion context
4. Month               → monthly seasonality
5. Category (encoded)  → product type
6. Region (encoded)    → geographic segment
7. Season (encoded)    → seasonal patterns
```

### Target Variable (Y)
```
Price (₹) → Optimal selling price
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Main programming language |
| **Streamlit** | Interactive web dashboard |
| **Scikit-learn** | Random Forest ML model |
| **Pandas** | Data loading and preprocessing |
| **NumPy** | Mathematical operations |
| **Plotly** | Interactive charts and visualizations |
| **Joblib** | Saving and loading ML model |

---

## 📁 Project Structure

```
dynamic_ai_pricing_engine/
│
├── app.py                    ← Main Streamlit app (run this)
│
├── data/
│   ├── dataset.csv           ← Original synthetic dataset (73,100 rows)
│   └── realistic_dataset.csv ← Realistic dataset with real patterns
│
└── requirements.txt          ← Python dependencies
```

---

## ⚡ How to Run

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/dynamic-ai-pricing-engine.git
cd dynamic-ai-pricing-engine
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Dashboard
```bash
streamlit run app.py
```

### Step 4 — Open in Browser
```
http://localhost:8501
```

---

## 📦 Requirements

```txt
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.18.0
scikit-learn>=1.4.0
joblib>=1.3.0
```

Install all at once:
```bash
pip install streamlit pandas numpy plotly scikit-learn joblib
```

---

## 📸 Screenshots

### 🏠 Overview Page
> 4 KPI cards showing Total Revenue ₹550M, Average Price ₹55.14,
> Total Units Sold 9.97M, and Top Category Furniture.
> Monthly Revenue Trend line chart and Revenue Share pie chart.

### 📊 Data Analysis Page
> Pricing Analysis tab with Our Price vs Competitor Price grouped bars
> and Correlation Matrix showing 0.99 correlation between prices.

### 📉 Price Elasticity Page
> Elasticity summary table with category-wise results.
> Horizontal bar chart with threshold line at E = -1.

### 🤖 AI Price Predictor Page
> 7-input form with sliders and dropdowns.
> Green result box showing recommended optimal price.
> Feature Importance chart showing Competitor Pricing as #1 factor.

### 💰 Revenue Simulator Page
> Hill-shaped revenue curve with green star at optimal price ₹44.70.
> Revenue Matrix heatmap for Discount × Elasticity combinations.

---

## 💡 Key Features

- ✅ **Single file project** — everything in `app.py`
- ✅ **Dataset switcher** — switch between Synthetic and Realistic data
- ✅ **Live interactive charts** — powered by Plotly
- ✅ **Real-time predictions** — model predicts instantly
- ✅ **Sidebar filters** — filter by Category and Region
- ✅ **Caching** — fast loading with `@st.cache_data`
- ✅ **Dark mode compatible** — custom CSS for all themes
- ✅ **Revenue optimization** — finds maximum revenue price automatically

---

## 🔍 Important Concepts Used

### Data Leakage — Detected & Fixed
Initially R² was 1.00 (suspiciously perfect). Investigation revealed
features like `Effective_Price` and `Price_Gap` contained the target
variable `Price` in their formulas. These were removed.
**Before fix:** R² = 1.00 ❌ | **After fix:** R² = 0.9882 ✅

### Price Elasticity Formula
```
Elasticity = (% Change in Quantity Demanded) / (% Change in Price)
```
- `E < -1` → Elastic (price sensitive buyers)
- `-1 < E < 0` → Inelastic (not price sensitive)

### Revenue Simulator Formula
```
New Units = Base Units × (1 + Elasticity × ((New Price - Base Price) / Base Price))
Revenue   = New Units × New Price × (1 - Discount / 100)
```

---

## 👥 Project Team

| Name | Roll Number | Role |
|------|-------------|------|
| **Jitesh Bhojwani** | 590013713 | ML Model + Dashboard |
| **Aditi Aggarwal** | 590015334 | EDA + Price Elasticity |

**Faculty Guide:** Dr. Khushboo Jain
**Institution:** UPES Dehradun — School of Computer Science
**Program:** MCA — Master of Computer Applications (Data Science)
**Academic Year:** 2023 – 2024

---

## 🚀 Future Scope

- [ ] Real-time competitor price tracking via web scraping
- [ ] Cloud deployment on AWS / Google Cloud
- [ ] LSTM deep learning for demand forecasting
- [ ] Replace synthetic dataset with real retail data
- [ ] Customer segmentation for personalized pricing
- [ ] REST API for e-commerce platform integration

---

## 📄 License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute with attribution.

---

<div align="center">

**Made with ❤️ by Jitesh Bhojwani & Aditi Aggarwal**

*MCA Data Science | UPES Dehradun | 2024*

⭐ **Star this repo if you found it helpful!** ⭐

</div>
