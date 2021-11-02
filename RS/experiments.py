from tqdm import tqdm
import pandas as pd
from NReinas import NQueensBaseAnnealer

# -------------------- Auto experiments ------------------
def execute_auto_experiment(n):
    ann = NQueensBaseAnnealer(n_queens=n)
    schedule = ann.auto(minutes=1)

    ann = NQueensBaseAnnealer(n_queens=n)
    ann.set_schedule(schedule)
    best_cost = ann.anneal()[1]

    return ann.epochs, best_cost, schedule


def execute_auto_experiments(n_final, iterations):
    column_names = ['n_reinas', 'epochs', 'best_cost', 'schedule']
    df = pd.DataFrame(columns=column_names)

    for n in range(60, n_final+1):
        for _ in range(iterations):           
            epochs, best_cost, schedule = execute_auto_experiment(n)

            df = df.append({
                'n_reinas': n,
                'epochs': epochs,
                'best_cost': best_cost,
                'schedule': schedule
            }, ignore_index=True)

    df.to_csv('./results/experiments_auto_grande.csv', index=False)


# -------------------- Normal experiments ------------------
def execute_experiment(n):
    ann = NQueensBaseAnnealer(n_queens=n)
    best_cost = ann.anneal()[1]

    return ann.epochs, best_cost


def execute_experiments(n_final, iterations):
    column_names = ['n_reinas', 'epochs', 'best_cost']
    df = pd.DataFrame(columns=column_names)

    for n in range(7,n_final+1):
        for _ in range(iterations):           
            epochs, best_cost = execute_experiment(n)

            print(best_cost)
            df = df.append({
                'n_reinas': n,
                'epochs': epochs,
                'best_cost': best_cost,
            }, ignore_index=True)

    df.to_csv('./results/experiments.csv', index=False)

            
if __name__=='__main__':
    n_final = 100
    iterations = 5
    
    execute_auto_experiments(n_final, iterations)
    # execute_experiments(n_final, iterations)