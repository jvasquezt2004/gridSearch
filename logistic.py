import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
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
    C, max_iter, solver, penalty, X_train, y_train, X_val, y_val = params

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        penalty=penalty,
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Evaluated: C={C}, max_iter={max_iter}, solver={solver}, penalty={penalty}, accuracy={accuracy}")

    return {
        "C": C,
        "max_iter": max_iter,
        "solver": solver,
        "penalty": penalty,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'C': [0.1, 1.0],  # Inverse of regularization strength
        'max_iter': [100, 200],  # Maximum number of iterations
        'solver': ['lbfgs', 'sag', 'saga'],  # Algorithm to use in the optimization
        'penalty': ['l2']  # Regularization penalty
    }

    param_combinations = list(itertools.product(
        params['C'],
        params['max_iter'],
        params['solver'],
        params['penalty']
    ))

    print(f"Total combinations: {len(param_combinations)}")

    params_list = [(C, max_iter, solver, penalty, X_train, y_train, X_val, y_val) 
                   for C, max_iter, solver, penalty in param_combinations]

    n_procesors = 8

    start = time.time()

    with Pool(n_procesors) as pool:
        results = pool.map(evaluate_model, params_list)

    end = time.time()
    print(f"Time taken: {end - start}")

    best_params = max(results, key=lambda x: x['accuracy'])
    print(f"Best parameters: {best_params}")
