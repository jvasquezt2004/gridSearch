import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools
from mpi4py import MPI
import time
import os
import psutil  # You may need to install this: pip install psutil

def load_data():
    print("Loading data...")
    mnist = fetch_openml('mnist_784', as_frame=False)
    X, y = mnist.data, mnist.target
    y = y.astype(np.uint8)
    return X, y


def evaluate_model(params):
    rank = MPI.COMM_WORLD.Get_rank()
    n_estimators, max_depth, min_samples_split, criterion, X_train, y_train, X_val, y_val = params

    print(f"[Process {rank}] Starting evaluation: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, criterion={criterion}")
    
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        criterion = criterion,
        random_state = 42,
        n_jobs = 1
    )

    print(f"[Process {rank}] Fitting model...")
    model.fit(X_train, y_train)
    
    print(f"[Process {rank}] Predicting...")
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"[Process {rank}] Completed: n_estimators = {n_estimators}, max_depth = {max_depth}, min_samples_split = {min_samples_split}, criterion = {criterion}, accuracy = {accuracy}")

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get and print process and CPU information
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_num = process.cpu_num()  # Get the CPU the process is currently running on
    
    # Each process prints its own information
    print(f"Process rank {rank}/{size-1} with PID {pid} running on CPU {cpu_num}")
    comm.Barrier()  # Synchronize to prevent output mixing
    
    # Load data on all processes
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

    if rank == 0:
        print(f"Total combinations: {len(param_combinations)}")
        print(f"Running with {size} MPI processes")

    # Create full parameter list
    all_params_list = [(n_estimators, max_depth, min_samples_split, criterion, X_train, y_train, X_val, y_val) 
                    for n_estimators, max_depth, min_samples_split, criterion in param_combinations]
    
    # Distribute work among processes
    start = time.time()
    
    # Calculate how many parameter combinations each process should handle
    local_params = []
    for i in range(len(all_params_list)):
        if i % size == rank:
            local_params.append(all_params_list[i])
    
    print(f"[Process {rank}] Assigned {len(local_params)} parameter combinations out of {len(param_combinations)}")
    
    # Each process evaluates its assigned parameter combinations
    local_results = []
    for i, params in enumerate(local_params):
        print(f"[Process {rank}] Processing combination {i+1}/{len(local_params)}")
        result = evaluate_model(params)
        local_results.append(result)
        print(f"[Process {rank}] Progress: {i+1}/{len(local_params)} combinations completed")
    
    # Gather all results at rank 0
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # Flatten the list of lists
        results = [item for sublist in all_results for item in sublist]
        
        end = time.time()
        print(f"Time taken: {end - start}")

        best_params = max(results, key=lambda x: x['accuracy'])
        print(f"Best parameters: {best_params}")

# speed up, eficiencia y tiempo de ejecucion