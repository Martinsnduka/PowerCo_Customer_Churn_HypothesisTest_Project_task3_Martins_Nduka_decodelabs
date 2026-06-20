# PowerCo Customer Churn — Is Price Sensitivity the Driver?

A data science investigation into why PowerCo, a utility company, is losing customers — built to test a specific business hypothesis rather than to ship a production model.

**Note:** This project is exploratory/analytical, not a deployed application. The goal was to answer a concrete business question with evidence, not to productionize a predictive service. If you're looking for an end-to-end deployed churn model from me, check out my **[TelCo Customer Churn project](#)**  which covers that workflow.


##  Business Question

PowerCo is losing customers. This attrition (churn) is eating into margins and threatening long-term growth. **What's driving this churn?**

### Hypothesis
 Customer churn is driven by **price sensitivity** — the degree to which a customer's likelihood to leave changes in response to changes in the price they pay.

### Test of Hypothesis
Test whether churn is primarily driven by customers' sensitivity to price changes, using historical client and pricing data.


## Dataset
Two datasets were provided and merged on client ID:

| Dataset | Rows | Columns | Description |
|---|---|---|---|
| client_data | 14,606 | 26 | Client demographics, consumption, contract dates, margins, churn flag |
| price_data | 193,002 | 8 | Monthly energy & power prices per client across off-peak, peak, and mid-peak periods |

**Target variable:** churn — whether the client churned within the next 3 months (binary: 1 = churned, 0 = retained)

**Class balance:** 9.7% of clients churned (1,419 of 14,606) — a meaningfully imbalanced target, handled accordingly during modeling.

## Project Workflow

### 1. Data Cleaning & Preprocessing
- Checked both datasets for missing values and duplicates
- Converted date columns (date_activ, date_end, date_modif_prod, date_renewal, price_date) to proper datetime types
- Standardized inconsistent categorical labels (e.g., MISSING → `Unknown` in channel_sales)

### 2. Exploratory Data Analysis (EDA)
Initial questions were used to interrogate the hypothesis directly from the raw data:

| Question | Finding |
|---|---|
| Do churned customers have lower margins? | **No** — churned customers actually had *higher* average net margin (228 vs. 185) |
| Do churned customers have shorter tenure? | **Yes** — churned customers averaged 4.6 years vs. 5.0 years for retained customers |
| Did churned customers consume more electricity? | **No** — retained customers had higher median 12-month consumption |

- Consumption-related features (cons_12m, cons_gas_12m, cons_last_month, imp_cons) were **highly right-skewed**, with outliers at the high end
- Boxplots confirmed the presence of outliers across most numerical features, addressed during feature engineering
- A correlation check against `churn` was used to flag features worth deeper investigation

### 3. Feature Engineering — Price Sensitivity
Since the hypothesis centers on price sensitivity, most engineering effort went into quantifying **how much each customer's prices changed**, at different levels of granularity:

| Feature Group | What it Captures |
|---|---|
| **Jan–Dec price change** | Annual change in off-peak energy & power price from Jan 2015 to Dec 2015 (macro-level shift) |
| **Average price differences across periods** | Mean difference between off-peak / peak / mid-peak prices across the full year |
| **Max monthly price differences across periods** | The largest month-to-month swing between pricing periods for each client (micro-level volatility) |

Time-based features were also engineered:
- **tenure** — years as a PowerCo client
- **months_to_end** — how close the client is to contract expiry
- **months_renewal** — how recently the client renewed their contract

 **Key finding:** Clients with **4 years of tenure or less were notably more likely to churn** than longer-tenured clients — churn risk is concentrated early in the customer lifecycle.

### 4. Data Transformation
Skewed consumption and forecast-related features were **log-transformed**  to reduce skew and stabilize variance ahead of modeling.

### 5. Encoding & Train/Test Split
- Binary encoding for has_gas (t/f → 1/0)
- Manual mapping for channel_sales` categories to integers
- 70/30 train-test split (X_train: 10,224 rows · X_test: 4,382 rows), with random_state=42


## Modeling

A **Random Forest Classifier** (1,000 trees, class_weight=balanced to counter class imbalance) was trained to predict churn — not as an end deliverable, but as a **diagnostic tool** to rank which features matter most for churn, in order to test the hypothesis.

RandomForestClassifier(
    n_estimators=1000,
    class_weight='balanced',
    random_state=42
)

### Performance

| Metric | Score |
|---|---|
| Accuracy | **89.94%** |
| Precision (Churn = Yes) | 0.50 |
| Recall (Churn = Yes) | 0.15 |
| F1-score (Churn = Yes) | 0.23 |

 **Important caveat:** Despite the high overall accuracy, the model struggles to actually *catch* churners — recall on the churn class is just 15%. The purpose of the model was to extract feature importances for hypothesis testing, not to serve as a deployable classifier.


## Results — Answering the Hypothesis

Feature importance rankings from the trained model revealed:

1. **Net margin and 12-month consumption** were the top drivers of churn
2. **Margin on power subscription** was also highly influential
3. **Time-based features** (tenure, months to contract end, months since renewal) were strong contributors
4. **Price sensitivity features were scattered throughout the importance ranking but were not dominant**

###  Conclusion

**Is churn driven by customers' price sensitivity?**

**No — not primarily.** Price sensitivity emerged as a **weak, secondary contributor** to churn rather than the main driver. Margin, consumption behavior, and contract timing (especially tenure) matter more in explaining why PowerCo customers leave.

**Business implication:** Retention strategy should prioritize **early-tenure customers** (first ~4 years) and **margin/consumption-based risk signals**, rather than leaning primarily on price-matching or discount strategies aimed at price-sensitive segments.



##  Future Improvements

If this were extended toward a production use case:
- Address class imbalance more directly (SMOTE, threshold tuning, or cost-sensitive learning) to improve recall on churners


##  Author
Email : ndukamartins2019@gmail.com
Feel free to connect or reach out with questions/feedback about this project.
