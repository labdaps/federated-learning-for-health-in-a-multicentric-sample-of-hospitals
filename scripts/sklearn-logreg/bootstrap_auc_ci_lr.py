import pandas as pd
import prep_iacov 
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, ci=0.95, seed=42):
    """
    Calculates the bootstrapped confidence interval for AUC-ROC score.

    Args:
        y_true (array-like): True labels.
        y_score (array-like): Predicted scores.
        n_bootstraps (int): Number of bootstrap samples.
        ci (float): Confidence interval level (e.g., 0.95 for 95%).
        seed (int): Seed for reproducibility.

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval.
    """
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # Generate random indices with replacement
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        
        # Ensure that the resampled data contains at least two classes
        if len(np.unique(y_true[indices])) < 2:
            continue # Skip this bootstrap sample if not enough classes
        
        score = roc_auc_score(y_true[indices], y_score[indices])
        bootstrapped_scores.append(score)
    
    # Calculate the percentile-based confidence interval
    sorted_scores = np.sort(bootstrapped_scores)
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    
    lower = np.percentile(sorted_scores, lower_percentile)
    upper = np.percentile(sorted_scores, upper_percentile)
    return lower, upper

# List to store results for each hospital
results = []

# Assuming 'lista_hospitais.csv' and 'F_Tabela_Geral_Final.csv' are in the same directory
# as this script, and 'prep_iacov.py' is importable.
try:
    # Loop through each hospital to process data and calculate AUCs with CIs
    for hospital in pd.read_csv('lista_hospitais.csv').hospital:

        dados = pd.read_csv('F_Tabela_Geral_Final.csv').query("X == @hospital")
        
        # Assuming prep_iacov.Prep(dados).executar_prep() correctly returns splits
        X_train, X_test, y_train, y_test  = prep_iacov.Prep(dados).executar_prep()

        # --- Local Model Training and Prediction ---
        model_local = LogisticRegression(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
            solver='liblinear' # Specify solver for LogisticRegression with warm_start and max_iter=1
        )

        model_local.fit(X_train, y_train)

        y_score_local = model_local.predict_proba(X_test)[:, 1]
        auc_local = roc_auc_score(y_test, y_score_local)
        
        # Calculate confidence interval for the local model
        ci_local_lower, ci_local_upper = bootstrap_auc_ci(np.array(y_test), np.array(y_score_local))


        # --- Global Model Prediction ---
        # Assuming 'model_final.pkl' contains the trained global model
        with open('model_final.pkl', 'rb') as f:
            model_global = pickle.load(f)

        y_score_global = model_global.predict_proba(X_test)[:, 1]
        auc_global = roc_auc_score(y_test, y_score_global)
        
        # Calculate confidence interval for the global model
        ci_global_lower, ci_global_upper = bootstrap_auc_ci(np.array(y_test), np.array(y_score_global))

        # Append results to the list
        results.append({
            'hospital': hospital,
            'ci_local_lower': ci_local_lower,
            'ci_local_upper': ci_local_upper,
            'ci_global_lower': ci_global_lower,
            'ci_global_upper': ci_global_upper
        })

    # Convert results to a pandas DataFrame and save to CSV
    df_results = pd.DataFrame(results)

    print("Confidence intervals calculated and saved to 'auc_ci_por_hospital.csv'")
    print(df_results)

except FileNotFoundError as e:
    print(f"Error: One of the required files was not found: {e}. Please ensure 'lista_hospitais.csv', 'F_Tabela_Geral_Final.csv', 'prep_iacov.py', and 'model_final.pkl' are in the correct directory.")
except ImportError as e:
    print(f"Error: A required module could not be imported: {e}. Please ensure 'prep_iacov.py' is correctly placed and accessible.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
