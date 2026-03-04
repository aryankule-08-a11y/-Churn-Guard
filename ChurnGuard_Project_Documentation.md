# 🛡️ ChurnGuard – Customer Churn Prediction System
## Complete Step-by-Step Project Documentation

> **Written in simple, easy-to-understand language for beginners and students.**

---

---

# 📌 SECTION 1: WHAT IS THIS PROJECT?

## 1.1 Project Name
**ChurnGuard – Customer Churn Prediction System**

## 1.2 What Does "Churn" Mean?
**Customer Churn** means when a customer **stops using a company's service** and leaves.

**Example:**
Imagine you are using a mobile network (like Airtel). One day you get frustrated with the service and switch to Jio. This is called **churning** – you left Airtel.

Companies lose a lot of money when customers leave. So they want to **predict in advance** which customers *might* leave — so they can offer discounts, better plans, or support to keep them.

## 1.3 What Does ChurnGuard Do?
ChurnGuard is a **web application** that:
- Takes customer data as input
- Uses **Machine Learning (AI)** to analyze patterns
- Predicts: **"Will this customer leave or not?"**
- Shows the result with a **probability percentage** (e.g., "78% chance of churning")

## 1.4 Who Is This Project For?
- 📚 Students learning Data Science / Machine Learning
- 🏢 Businesses that want to reduce customer loss
- 👨‍💻 Developers learning how to build ML-powered web apps

---

---

# 📌 SECTION 2: TOOLS & TECHNOLOGIES USED

## 2.1 Programming Language — Python
**Why Python?**
Python is the most popular language for Data Science and Machine Learning. It has ready-made libraries (tools) for nearly everything — data handling, building models, and creating web apps.

## 2.2 Streamlit — Web App Framework
**Why Streamlit?**
Streamlit lets you build a **beautiful web application using only Python** — no HTML, CSS, or JavaScript needed. It is perfect for Data Science projects because it is simple and fast to develop.

## 2.3 Pandas — Data Handling
**Why Pandas?**
Pandas is used to work with **tables of data** (like Excel sheets). We use it to:
- Load the dataset (CSV file)
- Clean the data
- Explore rows and columns

## 2.4 NumPy — Numbers & Math
**Why NumPy?**
NumPy handles **number calculations** very efficiently. It works behind the scenes when we do math operations on large datasets.

## 2.5 Scikit-learn — Machine Learning
**Why Scikit-learn?**
This is the main Machine Learning library. We use it to:
- **Encode** text data into numbers
- **Scale** numbers to the same range
- **Train** ML models (Random Forest, Logistic Regression)
- **Evaluate** model accuracy

## 2.6 Plotly — Interactive Charts
**Why Plotly?**
Plotly creates **interactive, beautiful charts** that users can zoom into, hover over, and explore. We use it for:
- Pie charts (churn vs retained)
- Bar charts (monthly charges)
- Confusion matrix heatmap
- ROC Curve
- Gauge chart (prediction result)

## 2.7 Matplotlib & Seaborn — Supporting Charts
These are traditional Python charting libraries. They are included as backup/support for some visualizations.

---

---

# 📌 SECTION 3: PROJECT STRUCTURE (FILES)

```
ChurnGuard/
│
├── app.py                     ← Main web application (all 5 pages)
├── generate_sample_data.py    ← Generates fake customer data for testing
├── requirements.txt           ← List of Python libraries to install
└── README.md                  ← Project description for GitHub
```

### What Each File Does:

| File | Purpose |
|------|---------|
| `app.py` | The heart of the project — all pages, UI, ML logic |
| `generate_sample_data.py` | Creates 1,000 fake customer records for demo |
| `requirements.txt` | Tells Python *which libraries to install* |
| `README.md` | Description shown on GitHub repo page |

---

---

# 📌 SECTION 4: STEP-BY-STEP — HOW THE PROJECT WORKS

---

## ✅ STEP 1: Generate / Load the Dataset

### What Happens Here?
Before we can train a Machine Learning model, we need **data** — specifically, a table of customer information.

We have **two options**:
1. **Upload your own CSV file** (your company's real customer data)
2. **Use the built-in Sample Dataset** (generated automatically for testing)

### About the Sample Dataset (`generate_sample_data.py`)
The file `generate_sample_data.py` creates **1,000 fake customer rows** with these columns:

| Column | What It Means | Example |
|--------|--------------|---------|
| `customerID` | Unique ID for each customer | CUST-0001 |
| `Gender` | Male or Female | Male |
| `Age` | Customer's age | 34 |
| `Tenure` | How many months they've been a customer | 24 |
| `MonthlyCharges` | How much they pay per month ($) | 65.50 |
| `TotalCharges` | Total amount paid so far ($) | 1572.00 |
| `ContractType` | Type of contract they have | Month-to-month |
| `InternetService` | Type of internet they use | Fiber optic |
| `PaymentMethod` | How they pay | Electronic check |
| `Churn` | Did they leave? (0 = No, 1 = Yes) | 1 |

### Why is Churn Rate Realistic?
The code doesn't generate random churn. It uses **rules** like:
- Month-to-month contract → higher chance of churning (no commitment)
- Fiber optic users → slightly higher churn
- Electronic check payment → higher churn (less engagement)
- Short tenure (new customers) → more likely to leave

This makes the data **realistic and useful** for training.

---

## ✅ STEP 2: Data Exploration (Understanding the Data)

### What Happens Here?
Before training any model, we **explore the data** to understand it. This is called **Exploratory Data Analysis (EDA)**.

### Sub-Step 2.1: Dataset Preview
We show the first 20 rows of the table so you can **visually inspect** what the data looks like. We also show:
- Total number of rows
- Total number of columns
- Count of missing values

### Sub-Step 2.2: Descriptive Statistics
We show a **statistics table** with:
- **Mean** (average value)
- **Min / Max** (smallest and largest values)
- **Standard Deviation** (how spread out the values are)

This helps us understand the **range and distribution** of each column.

### Sub-Step 2.3: Visualizations (Charts)

#### 📊 Chart 1 — Churn vs Retained (Pie Chart)
- Shows **what % of customers churned vs stayed**
- Example: 73% Retained, 27% Churned
- **Why?** To understand the class imbalance — if 95% stayed and only 5% left, the model needs special handling

#### 📊 Chart 2 — Monthly Charges by Churn (Histogram)
- Compares **monthly charges of churned vs retained customers**
- **Why?** We can see if higher-paying customers leave more often

#### 📊 Chart 3 — Correlation Matrix (Heatmap)
- Shows how strongly each feature is **related to other features**
- Values close to **+1** = strong positive relationship
- Values close to **-1** = strong negative relationship
- **Why?** Helps us understand which features matter most

---

## ✅ STEP 3: Data Preprocessing (Cleaning & Preparing Data)

### What Happens Here?
Raw data cannot be fed directly into a Machine Learning model. We must **clean and transform** it first. This process is called **Preprocessing**.

### Sub-Step 3.1: Drop ID Columns
The `customerID` column (like `CUST-0001`) is just a label — it has **no predictive value**. So we remove it.

**Why?** If we kept it, the model might memorize customer IDs instead of learning real patterns.

---

### Sub-Step 3.2: Handle Missing Values
Sometimes data has **empty/blank cells** (missing values). The model can't work with blanks.

**Our fix:**
- For **number columns** (Age, Tenure, etc.) → fill blank with the **median** (middle value)
- For **text columns** (Gender, Contract, etc.) → fill blank with the **most common value (mode)**

**Why Median for numbers?**
Median is more stable than the average. If 99 people earn ₹10,000 and 1 person earns ₹1,000,000 — the average is misleading, but the median is still around ₹10,000.

---

### Sub-Step 3.3: Encode Categorical Features
Machine Learning models only understand **numbers** — not words like "Male", "Fiber optic", or "Month-to-month".

So we convert text to numbers using **Label Encoding**:

| Original Text | Encoded Number |
|--------------|---------------|
| Male | 0 |
| Female | 1 |
| DSL | 0 |
| Fiber optic | 1 |
| No | 2 |
| Month-to-month | 0 |
| One year | 1 |
| Two year | 2 |

**Why Label Encoding?**
It's simple and works well with tree-based models like Random Forest.

---

### Sub-Step 3.4: Scale Numerical Features
Different columns have very different ranges:
- **Age**: 18 to 80
- **MonthlyCharges**: 18 to 120
- **TotalCharges**: 0 to 8,000+

If we don't scale them, the model might think **TotalCharges** is more important just because its numbers are larger.

We use **Standard Scaling** (also called Z-score normalization):

```
Scaled Value = (Original Value - Mean) / Standard Deviation
```

After scaling, all features have **mean = 0** and a similar range.

**Why?** This makes the model learn **fairly** from all features — no feature dominates just due to its scale.

---

### Sub-Step 3.5: Split Data into Train & Test Sets

We split our 1,000 rows into:
- **80% for Training** (800 rows) — the model learns from these
- **20% for Testing** (200 rows) — we test how well the model performs on data it has NEVER seen

**Why split?** If we tested on the same data we trained on, the model would score 100% — but that's **cheating**. The test set gives an honest, real-world performance score.

---

## ✅ STEP 4: Machine Learning Model Training

### What Happens Here?
We take the prepared training data and **teach a Machine Learning model** to recognize patterns that predict churn.

### Two Models Available:

---

#### 🤖 Model Option 1: Random Forest (Default)
**What is it?**
Random Forest is like asking **200 different experts (decision trees)** the same question and taking a **majority vote**.

**How it works:**
1. It creates 200 different "decision trees"
2. Each tree looks at different random pieces of the data
3. Each tree makes a prediction (Churn or No Churn)
4. The final answer = whatever **majority of trees voted for**

**Why Random Forest?**
- Very accurate
- Works well with both numbers and categories
- Handles **non-linear relationships** (complex patterns)
- Is robust to outliers (extreme values)
- Provides **feature importance** (tells us which columns matter most)

---

#### 🤖 Model Option 2: Logistic Regression
**What is it?**
Despite the name, Logistic Regression is a **classification** model (not regression). It finds a **straight line** that separates churned vs not-churned customers.

**How it works:**
1. It calculates a **score** for each customer based on their features
2. It passes that score through a **Sigmoid function** (S-shaped curve)
3. Output: a probability between 0 and 1
4. If probability > 0.5 → Churn, otherwise → No Churn

**Why Logistic Regression?**
- Fast to train
- Easy to interpret (we can see which features push toward churn)
- Good **baseline model** to compare against
- Works well when data is roughly linearly separable

---

## ✅ STEP 5: Model Evaluation (How Good Is Our Model?)

### What Happens Here?
After training, we test the model on the **Test Set (20% data it never saw)** and measure performance using multiple metrics.

---

### 📏 Metric 1: Accuracy
**What it means:** Out of all predictions, how many were **correct**?

```
Accuracy = (Correct Predictions) / (Total Predictions) × 100%
```

**Example:** If the model correctly predicted 175 out of 200 customers → Accuracy = **87.5%**

---

### 📏 Metric 2: Confusion Matrix
This is a **2×2 table** that shows:

| | Predicted: No Churn | Predicted: Churn |
|--|--|--|
| **Actual: No Churn** | ✅ True Negative (TN) | ❌ False Positive (FP) |
| **Actual: Churn** | ❌ False Negative (FN) | ✅ True Positive (TP) |

- **True Positive:** Model said "will churn" → customer actually churned ✅
- **True Negative:** Model said "won't churn" → customer actually stayed ✅
- **False Positive:** Model said "will churn" → customer actually stayed ❌ (False alarm)
- **False Negative:** Model said "won't churn" → customer actually churned ❌ (Missed!)

**Why it matters?** Missing a churning customer (False Negative) costs the company money. We want to minimize this.

---

### 📏 Metric 3: Precision & Recall
- **Precision:** Of all customers we *predicted* will churn — how many *actually* churned?
- **Recall:** Of all customers who *actually* churned — how many did we *catch*?

In a churn problem, **Recall is more important** — we want to catch as many churning customers as possible.

---

### 📏 Metric 4: AUC-ROC Score
**AUC** stands for Area Under the Curve. The **ROC Curve** plots:
- X-axis: False Positive Rate
- Y-axis: True Positive Rate

A score of **1.0 = perfect model**, **0.5 = random guessing**

Our Random Forest typically scores **~0.85–0.92** on this dataset.

---

### 📏 Metric 5: Feature Importance
Only available for Random Forest. This chart shows **which features the model found most useful** for predicting churn.

**Example ranking (typical):**
1. 🥇 MonthlyCharges — most important
2. 🥈 Tenure — very important
3. 🥉 TotalCharges — important
4. ContractType — moderate
5. PaymentMethod — moderate
6. Age — moderate
7. InternetService — some importance
8. Gender — least important

---

## ✅ STEP 6: Single Customer Prediction

### What Happens Here?
After training the model, you can enter details of **any individual customer** and get an instant prediction.

### Input Fields:

| Field | What to Enter |
|-------|--------------|
| 👤 Gender | Male or Female |
| 🎂 Age | Age of the customer (18–80) |
| 📅 Tenure | How many months they've been a customer |
| 💰 Monthly Charges | Their monthly bill amount |
| 📝 Contract Type | Month-to-month / One year / Two year |
| 🌐 Internet Service | DSL / Fiber optic / No |
| 💳 Payment Method | How they pay |

### What Happens Behind the Scenes When You Click "Predict":

1. Input data is collected from the form
2. Text values (like "Male", "Fiber optic") are **encoded** using the same encoder from training
3. Numbers are **scaled** using the same scaler from training
4. Data is passed to the **trained ML model**
5. Model returns: prediction (0 or 1) + probability (0.0 to 1.0)

### Output:

**If Churn = 1:**
```
⚠️ Customer Will Churn
Churn Probability: 78.4%
🔴 This customer is at HIGH RISK of leaving.
Consider offering retention incentives.
```

**If Churn = 0:**
```
✅ Customer Will Not Churn
Retention Probability: 91.2%
🟢 This customer is LIKELY TO STAY. Keep up the good service!
```

A **Gauge chart** also visually shows the churn risk:
- 🟢 0–30% → Low Risk
- 🟡 30–70% → Medium Risk
- 🔴 70–100% → High Risk

---

---

# 📌 SECTION 5: USER INTERFACE (5 PAGES)

## Page 1 — 🏠 Home
- Welcome screen with project description
- Option to upload CSV or load sample data
- Quick stats: Total Customers, Features, Churn Rate, Retained %

## Page 2 — 📊 Data Exploration
Three tabs:
- **Preview Tab** → see first 20 rows + row/column/missing value counts
- **Statistics Tab** → descriptive stats + data types table
- **Visualizations Tab** → Pie chart + Histogram + Correlation heatmap

## Page 3 — 🤖 Model Training
- Choose algorithm (Random Forest or Logistic Regression)
- Click **"Train Model"** button
- See accuracy, AUC, precision, recall
- Four tabs: Confusion Matrix / Classification Report / Feature Importance / ROC Curve

## Page 4 — 🔮 Prediction
- Fill in customer details in a clean 3-column form
- Click **"Predict Churn"**
- See result (churn/no churn), probability %, and gauge chart

## Page 5 — ℹ️ About
- Project description
- How it works (5 steps)
- Technologies used
- Model comparison table

---

---

# 📌 SECTION 6: APP DESIGN & UI

## Why Dark Theme?
A professional, dark-themed UI was chosen because:
- It looks modern and premium
- Easier on the eyes during long use
- Widely used in data/analytics dashboards

## Key Design Features:
| Feature | Why Used |
|---------|---------|
| **Gradient Sidebar** (dark blue) | Visually separates navigation from content |
| **Gradient Headers** (purple/indigo) | Catches attention, looks professional |
| **Styled Metric Cards** | Clearly displays key numbers at a glance |
| **Gradient Buttons** | Clear call-to-action, modern look |
| **Hover animation on buttons** | Feels interactive and alive |
| **Rounded corners everywhere** | Softer, modern design |
| **Inter font (Google Fonts)** | Clean, readable, professional typography |

---

---

# 📌 SECTION 7: HOW TO RUN THE PROJECT

## Step 1 — Clone from GitHub
```bash
git clone https://github.com/aryankule-08-a11y/-Churn-Guard.git
cd -Churn-Guard
```

## Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

This installs: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly

## Step 3 — Launch the App
```bash
streamlit run app.py
```

## Step 4 — Open in Browser
Go to: **http://localhost:8501**

## Step 5 — Use the App
1. Click **"🎲 Load Sample Dataset"** on the Home page
2. Navigate to **"🤖 Model Training"** → click **"🚀 Train Model"**
3. Go to **"🔮 Prediction"** → fill form → click **"🔮 Predict Churn"**
4. See the result!

## OR — Use the Live App
No installation needed → **https://churnguardx.streamlit.app/**

---

---

# 📌 SECTION 8: KEY CONCEPTS SUMMARY

| Concept | Simple Explanation |
|---------|-------------------|
| **Customer Churn** | Customer leaving/stopping the service |
| **Machine Learning** | Teaching a computer to find patterns in data |
| **Training Data** | Examples the model learns from (80%) |
| **Test Data** | New examples used to check model accuracy (20%) |
| **Label Encoding** | Converting text (Male/Female) → numbers (0/1) |
| **Standard Scaling** | Making all number columns have similar range |
| **Random Forest** | 200 decision trees voting together |
| **Logistic Regression** | Finding a boundary line between churn vs no churn |
| **Accuracy** | % of correct predictions |
| **Precision** | Of predicted churners, how many actually churned |
| **Recall** | Of actual churners, how many did we find |
| **AUC-ROC** | Overall model quality score (0.5 = bad, 1.0 = perfect) |
| **Confusion Matrix** | Table showing correct and wrong predictions |
| **Feature Importance** | Which columns matter most for the prediction |
| **Probability** | Confidence level of the prediction (0–100%) |

---

---

# 📌 SECTION 9: PROJECT FLOW DIAGRAM

```
START
  │
  ▼
[Load Dataset] ──── Upload CSV
  │                    OR
  │              Load Sample Data (1000 rows)
  │
  ▼
[Data Exploration]
  │── Preview table
  │── Show statistics
  └── Show charts (Pie, Histogram, Heatmap)
  │
  ▼
[Preprocessing]
  │── Remove ID columns
  │── Fill missing values (median / mode)
  │── Encode text → numbers (LabelEncoder)
  │── Scale numbers (StandardScaler)
  └── Split → 80% Train / 20% Test
  │
  ▼
[Model Training]
  │── Random Forest (200 trees)
  │        OR
  └── Logistic Regression
  │
  ▼
[Model Evaluation]
  │── Accuracy, AUC-ROC
  │── Confusion Matrix
  │── Feature Importance
  └── ROC Curve
  │
  ▼
[Single Prediction]
  │── Fill customer form
  │── Encode + Scale input
  │── Model predicts: Churn / No Churn
  └── Show probability + Gauge chart
  │
  ▼
END
```

---

---

# 📌 SECTION 10: CONCLUSION

## What We Achieved
✅ Built a complete, working Machine Learning web application
✅ Used real preprocessing techniques used in industry
✅ Implemented two different ML models
✅ Created an interactive, professional UI with charts
✅ Deployed live on Streamlit Cloud
✅ Published on GitHub with proper documentation

## What You Learned from This Project
- How to handle real-world data (missing values, encoding, scaling)
- How Random Forest and Logistic Regression work
- How to evaluate an ML model properly
- How to build a web app using Streamlit
- How to create interactive charts with Plotly
- How to deploy and share a Python web app

## Real-World Use Case
This exact type of system is used by:
- **Telecom companies** (Airtel, Jio, Vodafone)
- **Streaming services** (Netflix, Hotstar)
- **Banks and insurance companies**
- **SaaS (Software) businesses**

---

**Document Prepared for:** ChurnGuard Project
**Author:** Aryan Kule
**Date:** March 2026
**Live App:** https://churnguardx.streamlit.app/
**GitHub:** https://github.com/aryankule-08-a11y/-Churn-Guard
