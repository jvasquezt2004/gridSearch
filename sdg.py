import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools
from mpi4py import MPI
import time

def load_data():
    print("Loading data...")
    # Fetch MNIST by OpenML data_id directly
    X, y = fetch_openml(data_id=554, as_frame=False, return_X_y=True)
    y = y.astype(np.uint8)
    return X, y

def evaluate_model(params):
    alpha, loss, penalty, eta0, X_train, y_train, X_val, y_val = params

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = SGDClassifier(
        alpha=alpha,
        loss=loss,
        penalty=penalty,
        learning_rate="constant",
        eta0=eta0,
        max_iter=1000,
        random_state=42,
        n_jobs=1,
        early_stopping=True,
        validation_fraction=0.1,
        tol=1e-3
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Evaluated: alpha={alpha}, loss={loss}, penalty={penalty}, eta0={eta0}, accuracy={accuracy}")

    return {
        "alpha": alpha,
        "loss": loss,
        "penalty": penalty,
        "eta0": eta0,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        X, y = load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'loss': ['log_loss', 'hinge', 'modified_huber', 'squared_hinge'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'eta0': [0.001, 0.01, 0.1]
        }
        param_combinations = list(itertools.product(
            params['alpha'], params['loss'], params['penalty'], params['eta0']
        ))
        print(f"Total combinations: {len(param_combinations)}")
        params_list = [
            (alpha, loss, penalty, eta0, X_train, y_train, X_val, y_val)
            for alpha, loss, penalty, eta0 in param_combinations
        ]
        chunks = [params_list[i::size] for i in range(size)]
        start = time.time()
    else:
        chunks = None
        X_train = X_val = y_train = y_val = None

    # Broadcast training and validation data
    X_train, X_val, y_train, y_val = comm.bcast(
        (X_train, X_val, y_train, y_val), root=0
    )
    # Distribute parameter subsets
    tasks = comm.scatter(chunks, root=0)
    # Each rank evaluates its assigned tasks
    local_results = [evaluate_model(task) for task in tasks]
    # Gather all results at root
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        end = time.time()
        # Flatten results from all ranks
        results = [res for sublist in all_results for res in sublist]
        execution_time = end - start
        print(f"Time taken: {execution_time:.2f} seconds")
        baseline = 1114.4441
        speedup = baseline / execution_time
        efficiency = speedup / size
        print(f"Speedup: {speedup:.2f}")
        print(f"Efficiency: {efficiency:.2f}")
        best_params = max(results, key=lambda x: x['accuracy'])
        print(f"Best parameters: {best_params}")
