# DATA MINING EXAM - QUICK REFERENCE GUIDE

## 📚 TP OVERVIEW & WHEN TO USE EACH TECHNIQUE

---

## TP1: ASSOCIATION RULES (Market Basket Analysis)
**File:** `tp1.ipynb`

### When to use:
- Finding patterns in transactional/purchase data
- Discovering which items are bought together
- Recommending products based on shopping cart
- Market basket analysis

### Key Concepts:
- **Support**: How frequently itemset appears (min_support=0.4 = 40% of transactions)
- **Confidence**: Probability of buying B given A was bought (min_threshold=0.6 = 60%)
- **Lift**: Strength of association
  - Lift > 1: Items positively correlated (bought together MORE than chance)
  - Lift = 1: Independent (no relationship)
  - Lift < 1: Negatively correlated (bought together LESS than chance)

### Main Steps:
1. Load transactional data
2. Convert to boolean format
3. Apply Apriori algorithm with min_support
4. Generate association rules with min_confidence
5. Filter rules by lift > 1
6. Find best rules (highest lift)

### Key Functions:
- `apriori(dataset, min_support=0.4, use_colnames=True)`
- `association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)`
- `.ge({'item'})` - check if itemset contains specific item

---

## TP2: PCA - MANUAL IMPLEMENTATION
**File:** `tp2.ipynb`

### When to use:
- Reduce number of features while keeping variance
- Remove correlation between features
- Visualize high-dimensional data
- Data compression

### Key Concepts:
- **Centering**: Subtract mean (required for PCA)
- **Covariance Matrix**: Shows how features vary together
- **Eigenvalues**: Variance explained by each component
- **Eigenvectors**: Direction of principal components
- **PC1**: Component with highest variance

### Main Steps:
1. Load numerical data
2. Center data (subtract mean)
3. Calculate covariance matrix
4. Calculate eigenvalues/eigenvectors
5. Sort by eigenvalues (descending)
6. Transform data to PC space
7. Keep top components (dimensionality reduction)
8. Reconstruct data (inverse transformation)

### Key Functions:
- `dataset.mean()` - calculate mean
- `x_cnetre.cov()` - covariance matrix
- `np.linalg.eig(cov_matrix)` - eigenvalues/vectors
- `np.dot(x_cnetre, v)` - transform to PC space

---

## TP3: PCA - SKLEARN IMPLEMENTATION
**File:** `tp3.ipynb`

### When to use:
- Same as TP2 but faster/easier with sklearn
- Production code (more reliable)

### Key Concepts:
- Same as TP2 but automated

### Main Steps:
1. Load data
2. Create PCA object: `PCA()` or `PCA(n_components=k)`
3. Fit and transform: `pca.fit_transform(dataset)`
4. Check explained variance: `pca.explained_variance_`
5. Check components: `pca.components_`
6. Reconstruct: `pca.inverse_transform(x_pca)`
7. Visualize (optional)

### Key Functions:
- `PCA()` - keep all components
- `PCA(n_components=1)` - keep only 1 component
- `pca.fit_transform(dataset)` - center + compute + transform
- `pca.inverse_transform(x_pca)` - back to original space
- `pca.explained_variance_` - variance per component
- `pca.components_` - eigenvectors

---

## TP3.1 & TP3.2: LINEAR REGRESSION
**Files:** `tp3.1.ipynb`, `tp3.2.ipynb`

### When to use:
- Predicting continuous numerical values
- Understanding relationship between features and target
- Example: predict profit, price, salary, temperature

### Key Concepts:
- **Features (X)**: Input variables (R&D, Marketing, etc.)
- **Target (y)**: Output to predict (Profit)
- **Encoding**: Convert categorical to numerical (get_dummies)
- **Scaling**: Standardize features (mean=0, std=1)
- **R² Score**: Model accuracy (0-1, higher is better)
- **Train/Test Split**: Avoid overfitting

### Main Steps:
1. Load data
2. Check quality (duplicates, missing values)
3. Visualize outliers (boxplot - optional)
4. Separate X (features) and y (target)
5. Encode categorical variables: `pd.get_dummies(X, drop_first=True)`
6. Split train/test: `train_test_split(X, y, test_size=0.2)`
7. Scale numerical features: `StandardScaler()`
8. Train model: `LinearRegression().fit(X_train, y_train)`
9. Evaluate: `r2_score(y_test, y_pred)`
10. Make predictions on new data

### Key Functions:
- `pd.get_dummies(X, drop_first=True, dtype=int)` - encode categorical
- `train_test_split(X, y, test_size=0.2, random_state=0)` - split data
- `StandardScaler()` - standardize features
- `sc.fit(X_train)` - learn mean/std from training
- `sc.transform(X_test)` - apply same scaling to test
- `LinearRegression()` - create model
- `regressor.fit(X_train, y_train)` - train
- `regressor.predict(X_test)` - predict
- `r2_score(y_test, y_pred)` - evaluate accuracy
- `regressor.coef_` - feature coefficients
- `regressor.intercept_` - intercept term

### Important Notes:
- **Always fit scaler ONLY on training data** (avoid data leakage)
- **drop_first=True** in get_dummies (avoid multicollinearity)
- **Check mean ≈ 0 and std ≈ 1** after scaling

---

## TP4: CLASSIFICATION - Decision Trees & Random Forest
**File:** `tp4.ipynb`

### When to use:
- Predicting categorical outcomes (Yes/No, 0/1, classes)
- Example: predict diabetes, disease, customer churn, spam/not spam
- When you need interpretable model (Decision Tree visualization)

### Key Concepts:
- **Classification**: Predict discrete classes (not continuous values)
- **Decision Tree**: Tree-based model (easy to interpret)
- **Entropy**: Measure of impurity (criterion='entropy')
- **Max Depth**: Limits tree depth (prevents overfitting)
- **Cross-Validation**: Test model stability (cv=5 means 5 folds)
- **Grid Search**: Find best hyperparameters automatically
- **Random Forest**: Multiple trees (ensemble method, reduces overfitting)
- **Feature Importance**: Which features matter most

### Main Steps:

#### Basic Decision Tree:
1. Load data
2. Check correlations and missing values
3. Separate X and y
4. Split train/test: `train_test_split(x, y, test_size=0.3)`
5. Train: `DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)`
6. Predict: `dt.predict(x_test)`
7. Evaluate: `classification_report(y_test, y_pred)`

#### Prevent Overfitting:
8. Add max_depth: `DecisionTreeClassifier(criterion='entropy', max_depth=3)`
9. Visualize tree: `tree.plot_tree(clf, feature_names=x.columns)`

#### Cross-Validation:
10. Validate: `cross_val_score(clf, x_train, y_train, cv=5)`
11. Average score: `np.mean(cv_scores)`

#### Hyperparameter Tuning:
12. Define grid: `pgrid = {'max_depth': [1,2,3...], 'min_samples_split': [2,3,5...]}`
13. Grid search: `GridSearchCV(DecisionTreeClassifier(), param_grid=pgrid, cv=5)`
14. Fit: `grid_search.fit(x_train, y_train)`
15. Best params: `grid_search.best_params_`
16. Best score: `grid_search.best_estimator_.score(x_test, y_test)`

#### Random Forest:
17. Train RF: `RandomForestClassifier(n_estimators=500, criterion='entropy')`
18. Get feature importance: `rf.feature_importances_`
19. Visualize: `sns.barplot(x=feature_score, y=feature_score.index)`
20. Select top features and retrain

### Key Functions:
- `DecisionTreeClassifier(criterion='entropy')` - create tree
- `DecisionTreeClassifier(criterion='entropy', max_depth=3)` - limited depth
- `tree.plot_tree(clf, feature_names=x.columns, class_names=['0','1'], filled=True)` - visualize
- `cross_val_score(clf, x_train, y_train, cv=5)` - cross-validation
- `GridSearchCV(model, param_grid=pgrid, cv=5)` - find best params
- `RandomForestClassifier(n_estimators=500)` - ensemble of 500 trees
- `classification_report(y_test, y_pred)` - precision, recall, f1-score
- `rf.feature_importances_` - importance of each feature

### Metrics:
- **Precision**: Of predicted positives, how many correct?
- **Recall**: Of actual positives, how many found?
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

---

## 🎯 QUICK DECISION GUIDE

### What type of problem?

**1. Finding patterns in transactions → TP1 (Association Rules)**
- "Which products are bought together?"
- "What should we recommend?"

**2. Too many features, need to reduce → TP2 or TP3 (PCA)**
- "Can we simplify our data?"
- "Which features capture most variance?"

**3. Predicting continuous numbers → TP3.1/TP3.2 (Linear Regression)**
- "What will the profit be?"
- "Predict salary/price/temperature"

**4. Predicting categories → TP4 (Classification)**
- "Does patient have diabetes? (Yes/No)"
- "Is email spam? (0/1)"
- "Which class does this belong to?"

---

## 💡 COMMON EXAM PATTERNS

### Data Preprocessing (All TPs):
```python
# Load data
dataset = pd.read_csv("path/to/file.csv")

# Check quality
dataset.isnull().sum()  # Missing values
dataset.duplicated().sum()  # Duplicates
dataset.corr()  # Correlations

# Separate features and target
X = dataset.iloc[:, :-1]  # All except last
y = dataset.iloc[:, -1]   # Last column
```

### Train/Test Split (Regression & Classification):
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
```

### Encoding Categorical Variables:
```python
X = pd.get_dummies(X, drop_first=True, dtype=int)
```

### Scaling Features:
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train[cols])  # Fit ONLY on training
X_train[cols] = sc.transform(X_train[cols])
X_test[cols] = sc.transform(X_test[cols])
```

### Evaluation:
```python
# Regression
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# Classification
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## 🔥 EXAM TIPS

1. **Read the problem carefully**: Is it regression (continuous) or classification (categorical)?

2. **Always check data quality first**:
   - Missing values: `dataset.isnull().sum()`
   - Duplicates: `dataset.duplicated().sum()`

3. **Remember the order**:
   - Load → Clean → Split → Scale → Train → Evaluate → Predict

4. **Scaling**: 
   - Fit ONLY on training data
   - Transform both training and test

5. **Categorical variables**:
   - Use `pd.get_dummies(X, drop_first=True, dtype=int)`

6. **Evaluation**:
   - Regression: R² score (higher = better)
   - Classification: Precision, Recall, F1-score

7. **Overfitting**:
   - Training score >> Test score = Overfitting
   - Solution: max_depth, cross-validation, Random Forest

8. **Use comments**: Explain your reasoning in the exam!

---

## 📝 CHEAT SHEET

### Association Rules:
```python
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(dataset, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
best_rules = rules[rules['lift'] > 1]
```

### PCA (sklearn):
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_var = pca.explained_variance_
```

### Linear Regression:
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

### Decision Tree:
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### Random Forest:
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, criterion='entropy')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
feature_importance = rf.feature_importances_
```

---

## Good luck on your exam! 🚀

