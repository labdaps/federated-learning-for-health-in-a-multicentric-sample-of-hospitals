# ==============================================================================
# 0. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import pandas as pd
import numpy as np
import pickle

# Bibliotecas de Machine Learning
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Bibliotecas para visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Módulo customizado
import prep_iacov

# ==============================================================================
# 1. FUNÇÕES CENTRAIS DE BOOTSTRAP (sem alterações)
# ==============================================================================

def _calculate_stats(data, alpha=0.05):
    """
    Função auxiliar para calcular a média e o intervalo de confiança (95% por padrão)
    a partir de uma distribuição de dados (ex: AUCs do bootstrap).
    """
    mean = np.mean(data)
    lower_bound = np.percentile(data, 100 * (alpha / 2))
    upper_bound = np.percentile(data, 100 * (1 - alpha / 2))
    return mean, (lower_bound, upper_bound)

def bootstrap_analysis(y_true, y_pred_local, y_pred_federated, n_bootstraps=1000):
    """
    Executa a análise completa de bootstrap. Esta função é agnóstica ao modelo.
    """
    y_true = np.array(y_true)
    y_pred_local = np.array(y_pred_local)
    y_pred_federated = np.array(y_pred_federated)
    n_samples = len(y_true)
    local_aucs, federated_aucs, gains = [], [], []

    for _ in range(n_bootstraps):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc_local_boot = roc_auc_score(y_true[indices], y_pred_local[indices])
        auc_federated_boot = roc_auc_score(y_true[indices], y_pred_federated[indices])
        local_aucs.append(auc_local_boot)
        federated_aucs.append(auc_federated_boot)
        gains.append(auc_federated_boot - auc_local_boot)
        
    mean_local_auc, ci_local_auc = _calculate_stats(local_aucs)
    mean_federated_auc, ci_federated_auc = _calculate_stats(federated_aucs)
    mean_gain, ci_gain = _calculate_stats(gains)

    return {
        'auc_local_mean': mean_local_auc, 'auc_local_ci': ci_local_auc,
        'auc_federated_mean': mean_federated_auc, 'auc_federated_ci': ci_federated_auc,
        'gain_mean': mean_gain, 'gain_ci': ci_gain
    }

# ==============================================================================
# 2. CÓDIGO PRINCIPAL DE EXECUÇÃO (Adaptado para MLP)
# ==============================================================================

all_results = []

for hospital in pd.read_csv('lista_hospitais.csv').hospital:
    print(f"--- Processando hospital: {hospital} ---")

    dados = pd.read_csv('F_Tabela_Geral_Final.csv').query("X == @hospital")
    
    n_amostras = len(dados)
    print(f"Amostras encontradas: {n_amostras}")
    
    prep_module = prep_iacov.Prep(dados) 
    X_train, X_test, y_train, y_test = prep_module.executar_prep()

    # --- Pré-processamento do Alvo (y) para o Keras ---
    y_test_original = y_test.copy() # Guarda o y_test original (1D) para o bootstrap
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test_original)
    
    y_train_cat = keras.utils.to_categorical(y_train_encoded, num_classes=2)
    y_test_cat = keras.utils.to_categorical(y_test_encoded, num_classes=2)
    
    # --- Modelo Local (MLP) ---
    model_local = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)), # Input dinâmico
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model_local.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=[keras.metrics.AUC(name='auc')])

    print("Treinando modelo MLP local...")
    model_local.fit(X_train, y_train_cat, epochs=15, batch_size=32, verbose=0)
    
    # Obtém as probabilidades para o bootstrap (probabilidade da classe 1)
    y_pred_local = model_local.predict(X_test)[:, 1]

    # --- Modelo Federado (MLP) ---
    # Carregando o modelo usando pickle, conforme solicitado.
    print("Carregando modelo MLP federado de 'model_final.pkl'...")
    with open('model_final.pkl', 'rb') as f:
        modelo_federado = pickle.load(f)
    
    # Obtém as probabilidades para o bootstrap
    y_pred_federado = modelo_federado.predict(X_test)[:, 1]

    # Checagem rápida da performance antes do bootstrap
    auc_local_direta = roc_auc_score(y_test_original, y_pred_local)
    auc_federada_direta = roc_auc_score(y_test_original, y_pred_federado)
    print(f"AUC Direta (Local): {auc_local_direta:.4f}")
    print(f"AUC Direta (Federado): {auc_federada_direta:.4f}")

    # Executa a análise estatística
    print("Executando análise de bootstrap...")
    results_dict = bootstrap_analysis(y_test_original, y_pred_local, y_pred_federado)
    results_dict['hospital'] = hospital
    results_dict['quantidade'] = n_amostras
    
    all_results.append(results_dict)
    print(f"Resultados para {hospital} calculados.\n")

# ==============================================================================
# 3. EXIBIÇÃO E EXPORTAÇÃO DOS RESULTADOS FINAIS (sem alterações)
# ==============================================================================

results_df = pd.DataFrame(all_results)

results_df['auc_local_ci_lower'] = results_df['auc_local_ci'].apply(lambda x: x[0])
results_df['auc_local_ci_upper'] = results_df['auc_local_ci'].apply(lambda x: x[1])
results_df['auc_federated_ci_lower'] = results_df['auc_federated_ci'].apply(lambda x: x[0])
results_df['auc_federated_ci_upper'] = results_df['auc_federated_ci'].apply(lambda x: x[1])
results_df['gain_ci_lower'] = results_df['gain_ci'].apply(lambda x: x[0])
results_df['gain_ci_upper'] = results_df['gain_ci'].apply(lambda x: x[1])

display_columns = [
    'hospital', 'quantidade', 
    'auc_local_mean', 'auc_local_ci', 
    'auc_federated_mean', 'auc_federated_ci', 
    'gain_mean', 'gain_ci'
]
results_df_display = results_df[display_columns]

pd.options.display.float_format = '{:,.4f}'.format
print("--- TABELA DE RESULTADOS FINAIS (MLP) ---")
print(results_df_display)

csv_columns = [
    'hospital', 'quantidade', 'auc_local_mean', 'auc_local_ci_lower', 'auc_local_ci_upper',
    'auc_federated_mean', 'auc_federated_ci_lower', 'auc_federated_ci_upper',
    'gain_mean', 'gain_ci_lower', 'gain_ci_upper'
]
df_para_csv = results_df[csv_columns]

df_para_csv.to_csv(
    'resultados_bootstrap_mlp.csv', 
    index=False,
    float_format='%.4f'
)
print("\nArquivo 'resultados_bootstrap_mlp.csv' salvo com sucesso (com 4 casas decimais).")


# ==============================================================================
# 4. GERAÇÃO DE GRÁFICOS (Títulos atualizados para MLP)
# ==============================================================================

print("\nGerando gráficos de performance (estilo híbrido)...")

results_df_sorted = results_df.sort_values(by='quantidade').reset_index(drop=True)
x_values = results_df_sorted['quantidade']

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- Gráfico 1: Comparação de AUCs com ÁREAS DE CONFIANÇA SOMBREADAS ---
ax1 = axes[0]
ax1.plot(x_values, results_df_sorted['auc_local_mean'], marker='o', linestyle='-', 
         label='Local Learning (Mean)', color='royalblue')
ax1.fill_between(x_values, results_df_sorted['auc_local_ci_lower'], results_df_sorted['auc_local_ci_upper'],
                 color='royalblue', alpha=0.2, label='_nolegend_')
ax1.plot(x_values, results_df_sorted['auc_federated_mean'], marker='s', linestyle='--',
         label='Federated Learning (Mean)', color='firebrick')
ax1.fill_between(x_values, results_df_sorted['auc_federated_ci_lower'], results_df_sorted['auc_federated_ci_upper'],
                 color='firebrick', alpha=0.2, label='_nolegend_')
ax1.set_xscale('log')
ax1.set_title('Predictive Performance: Federated vs. Local MLP', fontsize=16) # <-- Título Atualizado
ax1.set_xlabel('Number of Patients (per Hospital, log scale)', fontsize=12)
ax1.set_ylabel('AUC Score', fontsize=12)
ax1.set_ylim(0.45, 1.05)
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- Gráfico 2: Ganho (Delta AUC) com ERRORBARS TRADICIONAIS ---
ax2 = axes[1]
yerr_gain = [
    results_df_sorted['gain_mean'] - results_df_sorted['gain_ci_lower'], 
    results_df_sorted['gain_ci_upper'] - results_df_sorted['gain_mean']
]
ax2.errorbar(x=x_values, y=results_df_sorted['gain_mean'], yerr=yerr_gain,
             fmt='-D', capsize=5, color='darkgreen', label='AUC Gain (95% CI)')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_xscale('log')
ax2.set_title('Performance Gain (Delta AUC) with Federated Learning', fontsize=16)
ax2.set_xlabel('Number of Patients (per Hospital, log scale)', fontsize=12)
ax2.set_ylabel('Delta AUC (Federated - Local)', fontsize=12)
ax2.legend()
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('performance_comparison_plot_mlp.png', dpi=300)
plt.show()

print("\nGráfico híbrido 'performance_comparison_plot_mlp.png' salvo com sucesso.")