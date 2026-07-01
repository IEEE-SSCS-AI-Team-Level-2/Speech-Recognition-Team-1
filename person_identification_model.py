import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def train_person_identification_models(X, y):
    # X:numpy array (n_samples, n_features)
    # y:numpy array for Speaker labels

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    models = {
        "SVM": SVC(kernel='rbf', C=1.0,gamma='scale'),

        "Random Forest": RandomForestClassifier(n_estimators=200,random_state=42),

        "KNN": KNeighborsClassifier(n_neighbors=5),

        "XGBoost": XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,objective='multi:softmax',eval_metric='mlogloss',random_state=42)
    }

    results = {}

    best_model = None
    best_accuracy = 0


    # Training Loop
    for name, model in models.items():

        print("\n")
        print(f"{name}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)

        print(f"Accuracy: {accuracy:.4f}\n")

        print(classification_report(y_test,predictions))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test,predictions))

        results[name] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("\n")
    print("RESULTS")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.4f}")
    print("\nBest Model Accuracy:", best_accuracy)

    return best_model,scaler,results