import pandas as pd
import prep_iacov # This module needs to be available in your environment

import pickle
import numpy as np # Import numpy for bootstrap
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

session_id = 550

def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, ci=0.95, seed=42):
    """
    Calculates the bootstrapped confidence interval for AUC-ROC score.

    Args:
        y_true (array-like): True labels.
        y_score (array-like): Predicted scores.
        n_bootstraps (int): Number of bootstrap samples.
        ci (float): Confidence interval level (e.95 for 95%).
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


# Predição Local e Global
results_df_list = [] # List to store results for each hospital's DataFrame row

# Assuming 'lista_hospitais.csv' and 'F_Tabela_Geral_Final.csv' are in the same directory
# as this script, and 'prep_iacov.py' is importable.
try:
    dados_geral = pd.read_csv('F_Tabela_Geral_Final.csv')
    normalizacao = dados_geral['X'].value_counts(normalize=True).reset_index()

    for hospital in pd.read_csv('lista_hospitais.csv').hospital:

        n_estimators_for_hospital = int(normalizacao.loc[normalizacao['index'] == hospital, 
                                        'X'].values[0] * session_id)

        dados_hospital = dados_geral.query("X == @hospital") # Use 'dados_hospital' to avoid overwriting 'dados_geral'

        X_train, X_test, y_train, y_test  = prep_iacov.Prep(dados_hospital).executar_prep()

        # --- Local Model Training and Prediction ---
        model_local = RandomForestClassifier(n_estimators=n_estimators_for_hospital, random_state=42) # Added random_state for reproducibility

        model_local.fit(X_train, y_train)

        y_score_local = model_local.predict_proba(X_test)[:, 1]
        auc_local = roc_auc_score(y_test, y_score_local)
        
        # Calculate confidence interval for the local model
        ci_local_lower, ci_local_upper = bootstrap_auc_ci(np.array(y_test), np.array(y_score_local))


        # --- Global Model Prediction ---
        # Assuming 'global_model.pkl' contains the trained global model for this session_id
        with open(f'./envio_server/session_id_{session_id}/global_model.pkl', 'rb') as f:
            model_global = pickle.load(f)

        y_score_global = model_global.predict_proba(X_test)
        # Check if the global model's predict_proba returns a single array or two columns (for binary classification)
        # and adjust if necessary. For binary, it should be)[:, 1].
        if y_score_global.ndim > 1 and y_score_global.shape[1] > 1:
            y_score_global = y_score_global[:, 1]
        
        auc_global = roc_auc_score(y_test, y_score_global)
        
        # Calculate confidence interval for the global model
        ci_global_lower, ci_global_upper = bootstrap_auc_ci(np.array(y_test), np.array(y_score_global))

        # Append results for the current hospital
        results_df_list.append({
            'hospital': hospital,
            'ci_local_lower': ci_local_lower,
            'ci_local_upper': ci_local_upper,
            'ci_global_lower': ci_global_lower,
            'ci_global_upper': ci_global_upper
        })

    # Create final DataFrame from the list of results
    df_results = pd.DataFrame(results_df_list)

    print("Confidence intervals calculated and saved to 'dados_resumo_with_ci.csv'")
    print(df_results)

except FileNotFoundError as e:
    print(f"Error: One of the required files was not found: {e}. Please ensure 'lista_hospitais.csv', 'F_Tabela_Geral_Final.csv', 'prep_iacov.py', and the 'global_model.pkl' path are correct.")
except ImportError as e:
    print(f"Error: A required module could not be imported: {e}. Please ensure 'prep_iacov.py' is correctly placed and accessible.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
