import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools
from multiprocessing import Pool, cpu_count
import time

def load_data():
    print("Loading data...")
    mnist = fetch_openml('mnist_784', as_frame=False)
    X, y = mnist.data, mnist.target
    y = y.astype(np.uint8)
    return X, y


def evaluate_model(params):
    n_estimators, max_depth, min_samples_split, criterion, X_train, y_train, X_val, y_val = params

    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        criterion = criterion,
        random_state = 42,
        n_jobs = 1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Evaluated: n_estimators = {n_estimators}, max_depth = {max_depth}, min_samples_split = {min_samples_split}, criterion = {criterion}, accuracy = {accuracy}")

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'n_estimators': [10, 30],
        'max_depth': [5, 10],
        'min_samples_split': [2, 4],
        'criterion': ['gini', 'entropy', 'log_loss']
    }

    param_combinations = list(itertools.product(
        params['n_estimators'],
        params['max_depth'],
        params['min_samples_split'],
        params['criterion']
    ))

    print(f"Total combinations: {len(param_combinations)}")

    params_list = [(n_estimators, max_depth, min_samples_split, criterion, X_train, y_train, X_val, y_val) for n_estimators, max_depth, min_samples_split, criterion in param_combinations]

    n_procesors = 8

    start = time.time()

    with Pool(n_procesors) as pool:
        results = pool.map(evaluate_model, params_list)

    end = time.time()
    print(f"Time taken: {end - start}")

    best_params = max(results, key=lambda x: x['accuracy'])
    print(f"Best parameters: {best_params}")

# speed up, eficiencia y tiempo de ejecucion