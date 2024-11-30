import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv(r'C:\Users\Om\Downloads\StudentsPerformance.csv')

# Drop 'outcome' column if it exists
if 'outcome' in df.columns:
    df = df.drop('outcome', axis=1)

# Create a binary target for logistic regression and KNN ('pass' if math score >= 60)
df['pass_math'] = (df['math score'] >= 60).astype(int)

# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and targets
X = df.drop(['math score', 'pass_math'], axis=1)
y_math = df['math score']  # For Linear Regression
y_pass = df['pass_math']   # For Logistic Regression and KNN

# Split the dataset into training and testing sets
X_train, X_test, y_train_math, y_test_math = train_test_split(X, y_math, test_size=0.3, random_state=42)
_, _, y_train_pass, y_test_pass = train_test_split(X, y_pass, test_size=0.3, random_state=42)

# Standardize the features for KNN and clustering
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train_math)
y_pred_lin = lin_reg.predict(X_test)
mse = mean_squared_error(y_test_math, y_pred_lin)
print(f"Linear Regression Mean Squared Error (MSE): {mse:.2f}")

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train_pass)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test_pass, y_pred_log)
print(f"Logistic Regression Accuracy: {accuracy_log:.2f}")
print("Classification Report for Logistic Regression:")
print(classification_report(y_test_pass, y_pred_log))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train_pass)
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test_pass, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.2f}")
print("Confusion Matrix for KNN:")
print(confusion_matrix(y_test_pass, y_pred_knn))

# ---- Seed Data for Prediction ----
seed_scores = pd.DataFrame({
    'gender': [0, 1, 0, 1, 0],  # Encoded
    'race/ethnicity': [1, 2, 3, 1, 2],  # Encoded
    'parental level of education': [2, 3, 1, 0, 4],  # Encoded
    'lunch': [0, 1, 0, 1, 0],  # Encoded
    'test preparation course': [1, 0, 1, 0, 1],  # Encoded
    'reading score': [75, 85, 65, 90, 70],  # Added reading scores
    'writing score': [78, 82, 64, 88, 72]   # Added writing scores
})

# Standardize the seed data for KNN
seed_scores_scaled = scaler.transform(seed_scores)

# Logistic Regression Predictions
log_reg_preds = log_reg.predict(seed_scores)
print("\nLogistic Regression Predictions for Seed Data (Pass=1, Fail=0):")
print(log_reg_preds)

# KNN Predictions
knn_preds = knn.predict(seed_scores_scaled)
print("\nKNN Predictions for Seed Data (Pass=1, Fail=0):")
print(knn_preds)

# ---- K-Means Clustering ----
# Fit K-Means to find clusters in the data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_scaled)

# Predict cluster for seed data
seed_cluster_preds = kmeans.predict(seed_scores_scaled)

print("\nK-Means Cluster Predictions for Seed Data:")
print(seed_cluster_preds)

# Print K-Means cluster centers
print("\nK-Means Cluster Centers (scaled features):")
print(kmeans.cluster_centers_)
