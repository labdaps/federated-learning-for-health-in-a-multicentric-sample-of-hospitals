# ==============================================================================
# 0. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
import prep_iacov  # Módulo customizado para pré-processamento

# Bibliotecas para visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

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
    Executa a análise completa de bootstrap.

    Para cada uma das `n_bootstraps` iterações, uma nova amostra do conjunto de teste
    é criada por reamostragem com reposição. A performance (AUC) do modelo local,
    do modelo federado e o ganho (diferença entre eles) são calculados nessa amostra.

    Retorna um dicionário contendo a média e o intervalo de confiança para cada métrica.
    """
    y_true = np.array(y_true)
    y_pred_local = np.array(y_pred_local)
    y_pred_federated = np.array(y_pred_federated)
    n_samples = len(y_true)
    local_aucs, federated_aucs, gains = [], [], []

    for _ in range(n_bootstraps):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        # Garante que a amostra bootstrap tenha as duas classes para o cálculo da AUC
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
# 2. CÓDIGO PRINCIPAL DE EXECUÇÃO (sem alterações)
# ==============================================================================

all_results = []

for hospital in pd.read_csv('lista_hospitais.csv').hospital:
    print(f"--- Processando hospital: {hospital} ---")

    dados = pd.read_csv('F_Tabela_Geral_Final.csv').query("X == @hospital")
    
    n_amostras = len(dados)
    print(f"Amostras encontradas: {n_amostras}")
    
    prep_module = prep_iacov.Prep(dados) 
    X_train, X_test, y_train, y_test = prep_module.executar_prep()

    model_local = LogisticRegression(
        penalty="l2",
        max_iter=1000,
        random_state=42
    )
    model_local.fit(X_train, y_train)
    y_pred_local = model_local.predict_proba(X_test)[:, 1]

    with open('model_final.pkl', 'rb') as f:
        modelo = pickle.load(f)
    
    y_pred_federado = modelo.predict_proba(X_test)[:, 1]

    auc_local_direta = roc_auc_score(y_test, y_pred_local)
    auc_federada_direta = roc_auc_score(y_test, y_pred_federado)
    print(f"AUC Direta (Local): {auc_local_direta:.4f}")
    print(f"AUC Direta (Federado): {auc_federada_direta:.4f}")

    print("Executando análise de bootstrap...")
    results_dict = bootstrap_analysis(y_test, y_pred_local, y_pred_federado)
    results_dict['hospital'] = hospital
    results_dict['quantidade'] = n_amostras
    
    all_results.append(results_dict)
    print(f"Resultados para {hospital} calculados.\n")

# ==============================================================================
# 3. EXIBIÇÃO E EXPORTAÇÃO DOS RESULTADOS FINAIS (COM AJUSTE NO CSV)
# ==============================================================================

results_df = pd.DataFrame(all_results)

# Cria colunas auxiliares para os limites do intervalo de confiança
results_df['auc_local_ci_lower'] = results_df['auc_local_ci'].apply(lambda x: x[0])
results_df['auc_local_ci_upper'] = results_df['auc_local_ci'].apply(lambda x: x[1])
results_df['auc_federated_ci_lower'] = results_df['auc_federated_ci'].apply(lambda x: x[0])
results_df['auc_federated_ci_upper'] = results_df['auc_federated_ci'].apply(lambda x: x[1])
results_df['gain_ci_lower'] = results_df['gain_ci'].apply(lambda x: x[0])
results_df['gain_ci_upper'] = results_df['gain_ci'].apply(lambda x: x[1])

# --- EXIBIÇÃO NO CONSOLE (sem alterações) ---
# Define as colunas a serem exibidas na tabela do console
display_columns = [
    'hospital', 'quantidade', 
    'auc_local_mean', 'auc_local_ci', 
    'auc_federated_mean', 'auc_federated_ci', 
    'gain_mean', 'gain_ci'
]
results_df_display = results_df[display_columns]

# Configura a exibição para 4 casas decimais no print
pd.options.display.float_format = '{:,.4f}'.format
print("--- TABELA DE RESULTADOS FINAIS ---")
print(results_df_display)


# --- EXPORTAÇÃO PARA CSV (MODIFICADO) ---
# 1. Seleciona as colunas desejadas para um CSV limpo
csv_columns = [
    'hospital',
    'quantidade',
    'auc_local_mean',
    'auc_local_ci_lower',
    'auc_local_ci_upper',
    'auc_federated_mean',
    'auc_federated_ci_lower',
    'auc_federated_ci_upper',
    'gain_mean',
    'gain_ci_lower',
    'gain_ci_upper'
]
df_para_csv = results_df[csv_columns]

# 2. Salva o DataFrame selecionado em CSV com formatação de 4 casas decimais
df_para_csv.to_csv(
    'resultados_bootstrap_formatado.csv', 
    index=False,
    float_format='%.4f'  # Argumento chave para formatar os números
)
print("\nArquivo 'resultados_bootstrap_formatado.csv' salvo com sucesso (com 4 casas decimais).")


# ==============================================================================
# 4. GERAÇÃO DE GRÁFICOS (sem alterações)
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
ax1.set_title('Predictive Performance: Federated vs. Local Logistic Regression', fontsize=16)
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
plt.savefig('performance_comparison_plot_hybrid.png', dpi=300)
plt.show()

print("\nGráfico híbrido 'performance_comparison_plot_hybrid.png' salvo com sucesso.")