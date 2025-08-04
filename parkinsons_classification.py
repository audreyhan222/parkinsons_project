from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

def run_classification_pipeline(df):
    model_df = df.dropna(subset=['mean_hold', 'std_hold', 'mean_latency', 'std_latency', 'Age', 'Parkinsons'])

    model_df['Parkinsons'] = model_df['Parkinsons'].astype(int)

    features_to_use = ['mean_hold', 'std_hold', 'mean_latency', 'std_latency', 'Age']
    X = model_df[features_to_use]
    y = model_df['Parkinsons']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
        "Support Vector Machine": SVC(class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results_list = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        results_list.append({
            'Model': name,
            'Accuracy': accuracy,
            'Healthy_Recall (0)': report['0']['recall'],
            'PD_Recall (1)': report['1']['recall'],
            'Healthy_Precision (0)': report['0']['precision'],
            'PD_Precision (1)': report['1']['precision'],
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='Healthy_Recall (0)', ascending=False)
    return results_df