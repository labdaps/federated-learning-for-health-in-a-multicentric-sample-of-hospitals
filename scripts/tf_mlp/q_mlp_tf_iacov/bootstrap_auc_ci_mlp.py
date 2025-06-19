import pandas as pd
import prep_iacov # This module needs to be available in your environment

import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing # For LabelEncoder
import tensorflow as tf # For Keras models
import keras # For keras.metrics.AUC and keras.utils.to_categorical


def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, ci=0.95, seed=42):
    """
    Calculates the bootstrapped confidence interval for AUC-ROC score.

    Args:
        y_true (array-like): True labels (e.g., 0s and 1s).
        y_score (array-like): Predicted scores (probabilities of the positive class).
        n_bootstraps (int): Number of bootstrap samples.
        ci (float): Confidence interval level (e.g., 0.95 for 95%).
        seed (int): Seed for reproducibility.

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval.
    """
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    
    # Ensure y_true and y_score are numpy arrays for consistent indexing
    y_true = np.array(y_true)
    y_score = np.array(y_score)

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
results_df_list = []

# Assuming 'lista_hospitais.csv' and 'F_Tabela_Geral_Final.csv' are in the same directory
# as this script, and 'prep_iacov.py' is importable.
try:
    # Loop through each hospital to process data and calculate AUCs with CIs
    for hospital in pd.read_csv('lista_hospitais.csv').hospital:

        dados = pd.read_csv('F_Tabela_Geral_Final.csv').query("X == @hospital")
        
        # Assuming prep_iacov.Prep(dados).executar_prep() correctly returns splits
        X_train, X_test, y_train_original, y_test_original  = prep_iacov.Prep(dados).executar_prep()

        # --- Data Transformations for Keras ---
        le = preprocessing.LabelEncoder()
        
        # Fit on y_train_original and transform both train and test
        y_train_encoded = le.fit_transform(y_train_original)
        y_test_encoded = le.transform(y_test_original)

        # Convert to one-hot encoding for Keras
        y_train_categorical = keras.utils.to_categorical(y_train_encoded, num_classes=2)
        y_test_categorical = keras.utils.to_categorical(y_test_encoded, num_classes=2)

        # --- Local Model Training and Prediction (MLP) ---
        model_local = tf.keras.models.Sequential()
        # Input layer - assuming X_train.shape[1] is the correct input_shape
        model_local.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
        model_local.add(tf.keras.layers.Dense(20, activation='relu')) # hidden layer 1
        model_local.add(tf.keras.layers.Dense(10, activation='relu')) # hidden layer 2
        model_local.add(tf.keras.layers.Dense(2, activation='softmax')) # output layer

        model_local.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=[keras.metrics.AUC()]) # Use Keras AUC metric for compilation

        # Fit with one-hot encoded labels
        model_local.fit(X_train, y_train_categorical, epochs=5, batch_size=32, verbose=0) # verbose=0 to suppress training output

        # Get predicted probabilities for AUC calculation
        y_score_local = model_local.predict(X_test)[:, 1] # Probability of the positive class

        auc_local = roc_auc_score(y_test_original, y_score_local) # Use original labels for roc_auc_score
        
        # Calculate confidence interval for the local model
        ci_local_lower, ci_local_upper = bootstrap_auc_ci(np.array(y_test_original), np.array(y_score_local))


        # --- Global Model Prediction (MLP) ---
        # Assuming 'model_final.pkl' contains the trained global model
        with open('model_final.pkl', 'rb') as f:
            model_global = pickle.load(f)

        # Get predicted probabilities for AUC calculation
        y_score_global = model_global.predict(X_test)[:, 1] # Probability of the positive class

        auc_global = roc_auc_score(y_test_original, y_score_global) # Use original labels for roc_auc_score
        
        # Calculate confidence interval for the global model
        ci_global_lower, ci_global_upper = bootstrap_auc_ci(np.array(y_test_original), np.array(y_score_global))

        # Append results to the list
        results_df_list.append({
            'hospital': hospital,
            'ci_local_lower': ci_local_lower,
            'ci_local_upper': ci_local_upper,
            'ci_global_lower': ci_global_lower,
            'ci_global_upper': ci_global_upper
        })

    # Create final DataFrame from the list of results
    df_results = pd.DataFrame(results_df_list)

    print("Confidence intervals calculated and saved to 'dados_resumo_mlp_with_ci.csv'")
    print(df_results)

except FileNotFoundError as e:
    print(f"Error: One of the required files was not found: {e}. Please ensure 'lista_hospitais.csv', 'F_Tabela_Geral_Final.csv', 'prep_iacov.py', and 'model_final.pkl' are in the correct directory.")
except ImportError as e:
    print(f"Error: A required module could not be imported: {e}. Please ensure 'prep_iacov.py' is correctly placed and accessible.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
