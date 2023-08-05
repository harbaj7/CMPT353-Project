
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('analysis/activity_analysis.csv')
# Define the models
models = {
    "Logistic Regression": LogisticRegression(random_state=33),
    "Support Vector Machine": svm.SVC(kernel='poly'),
    "Decision Tree": DecisionTreeClassifier(splitter='best'),
    "Random Forest": RandomForestClassifier(n_estimators=50),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Gaussian Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=49),
    "Stochastic Gradient Descent": SGDClassifier(max_iter=1000, tol=1e-3),
    "Multi-layer Perceptron": MLPClassifier(random_state=42, max_iter=3000),
    "Ridge": RidgeClassifier(),
    "XGBoost": XGBClassifier( eval_metric='logloss')
}

# Instantiate the encoder
le = LabelEncoder()

# Transform the 'activity' column
df['activity'] = le.fit_transform(df['activity'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['dominant_frequency_x', 'dominant_frequency_y', 'dominant_frequency_z']], df['activity'], test_size=0.45, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate each model
for name, model in models.items():
    clf = model
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Model: {name}")
    print(f"Training accuracy: {train_score}")
    print(f"Testing accuracy: {test_score}")
    print("----------")