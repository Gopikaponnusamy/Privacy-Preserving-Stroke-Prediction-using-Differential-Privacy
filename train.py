from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from src.preprocess import load_and_preprocess
from src.dp_model import DPLogisticRegression

def evaluate_models(data_path):

    X, y, _, _ = load_and_preprocess(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    normal_model = LogisticRegression(max_iter=1000)
    normal_model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, normal_model.predict(X_test))

    epsilons = [0.1, 0.5, 1, 2, 5, 10]
    dp_results = []

    for eps in epsilons:
        model = DPLogisticRegression(epsilon=eps)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        dp_results.append((eps, acc))

    return baseline_acc, dp_results