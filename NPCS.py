import numpy as np
# from scipy.stats import truncnorm
from scipy.stats import poisson
import matplotlib.pyplot as plt
import locale
import time

# Configurações iniciais
locale.setlocale(locale.LC_ALL, '')  # Formatação de moeda brasileira R$
t0 = time.time()

# Constantes
INSS = 908.85  # Desconto máximo INSS
DED_IRPF_SAL = 896  # Parcela dedutível IRPF sobre salário acima de R$4.664,68
DED_IRPF_PLR = 3123.78  # Parcela dedutível sobre PLR para valores acima de R$16.380,38
ALI_IRPF = 0.275  # Alíquota IPRF para renda acima de R$4.664,68
AUX_REFEICAO = 2032.83
CESTA_ALIMENTACAO = 874.78
N_PLR = 3  # Número de salários PLR BNDES
NUM_FUNCIONARIOS = 900
NUM_ANOS = 65
AUMENTO_POR_NIVEL = 186.77
MAX_NIVEIS = 3

# Funções utilitárias
def limitar_num(n, min_val, max_val):
    return max(min(n, max_val), min_val)

def calcular_salario_liquido_from_bruto_mensal(salario_bruto_mensal):
    IRPF = (salario_bruto_mensal - INSS) * ALI_IRPF - DED_IRPF_SAL
    salario_liquido_sem_beneficios = salario_bruto_mensal + AUX_REFEICAO + CESTA_ALIMENTACAO - IRPF - INSS
    PLR_liquida = salario_bruto_mensal * N_PLR - (salario_bruto_mensal * N_PLR * ALI_IRPF - DED_IRPF_PLR)
    DEC_TERCEIRO_LIQ = salario_bruto_mensal - (salario_bruto_mensal * ALI_IRPF - DED_IRPF_SAL)
    FERIAS_LIQ = salario_bruto_mensal / 3 - ((salario_bruto_mensal / 3) * ALI_IRPF - DED_IRPF_SAL)
    salario_liquido = (salario_liquido_sem_beneficios * 12 + PLR_liquida + DEC_TERCEIRO_LIQ + FERIAS_LIQ + CESTA_ALIMENTACAO) / 12
    return salario_liquido

def calcular_salario_bruto_anual(salario_bruto_mensal):
    salario_bruto_anual = salario_bruto_mensal * 13 + salario_bruto_mensal * N_PLR + salario_bruto_mensal / 3 + (AUX_REFEICAO + CESTA_ALIMENTACAO) * 12 + CESTA_ALIMENTACAO
    return salario_bruto_anual

def calcular_folha_salarial_enxuta(salario_bruto_mensal):
    folha_salarial_enxuta = salario_bruto_mensal * 13 + salario_bruto_mensal / 3
    return folha_salarial_enxuta
    

def salario_bruto_mensal_from_anual(salario_bruto_anual):
    salario_bruto_mensal = (salario_bruto_anual - CESTA_ALIMENTACAO - (AUX_REFEICAO + CESTA_ALIMENTACAO) * 12) / (13 + N_PLR + 1/3)
    return salario_bruto_mensal

# Classe para simulação de promoções
class NPCS:
    @staticmethod
    def gerar_promocoes(lambda_, num_funcionarios):
        inferior, superior = 0, 3
        amostras = poisson.rvs(mu=lambda_, size=NUM_FUNCIONARIOS)
        amostras_truncadas = np.clip(amostras, inferior, superior)
        return np.round(amostras_truncadas).astype(int)

    @staticmethod
    def atualizar_salarios_e_impacto(promocoes_ano, salarios_iniciais_mensais, promocoes_acumuladas):
        folha_salarial_enxuta = np.zeros_like(salarios_iniciais_mensais)
        novos_salarios_anual = np.zeros_like(salarios_iniciais_mensais)
        for i, num_niveis in enumerate(promocoes_ano):
            novos_salarios_anual[i] = calcular_salario_bruto_anual(salarios_iniciais_mensais[i]) if promocoes_acumuladas[i] >= 79 else calcular_salario_bruto_anual(salarios_iniciais_mensais[i] + AUMENTO_POR_NIVEL * num_niveis)
            folha_salarial_enxuta[i] = calcular_folha_salarial_enxuta(salarios_iniciais_mensais[i]) if promocoes_acumuladas[i] >=79 else calcular_folha_salarial_enxuta(salarios_iniciais_mensais[i] + AUMENTO_POR_NIVEL * num_niveis)
        impacto_total = np.sum(novos_salarios_anual) - np.sum([calcular_salario_bruto_anual(s) for s in salarios_iniciais_mensais])
        return novos_salarios_anual, impacto_total, folha_salarial_enxuta

# Simulação
def simular(mu, num_funcionarios, num_anos, conservador):
    salario_inicial_1A = 21869.76

    cpgar52_atendida = False
    while not cpgar52_atendida:
        salarios_iniciais_mensais = np.full(num_funcionarios, salario_inicial_1A)
        promocoes_acumuladas = np.zeros(num_funcionarios, dtype=int)
        porcentagem_teto_salarial = np.zeros(num_anos)
        impactos_exercicios = np.zeros(num_anos)
        novos_salarios_anual_exercicios_media = np.zeros(num_anos)
        folhas_pagamentos = np.zeros(num_anos)
        folhas_pagamentos_enxutas = np.zeros(num_anos)
        promocoes_acumuladas_exercicios = np.zeros(num_anos)

        for ano in range(num_anos):
            promocoes_ano = NPCS.gerar_promocoes(mu, num_funcionarios)
            funcionarios_no_teto = promocoes_acumuladas >= 79
            niveis_para_redistribuir = np.sum(promocoes_ano[funcionarios_no_teto])
            if niveis_para_redistribuir > 0 and np.any(~funcionarios_no_teto):
                funcionarios_disponiveis = np.where(~funcionarios_no_teto)[0]
                redistribuicao = np.random.choice(funcionarios_disponiveis, size=niveis_para_redistribuir, replace=True)
                niveis_redistribuidos = np.bincount(redistribuicao, minlength=num_funcionarios)
                promocoes_ano[funcionarios_no_teto] = 0
                promocoes_ano[~funcionarios_no_teto] += niveis_redistribuidos[~funcionarios_no_teto]

            promocoes_acumuladas = np.clip(promocoes_acumuladas + promocoes_ano, 0, 79)
            promocoes_acumuladas_exercicios[ano] = np.mean(promocoes_acumuladas)
            porcentagem_teto_salarial[ano] = np.mean(promocoes_acumuladas >= 79) * 100

            novos_salarios_anual_exercicios_media[ano] = np.mean(NPCS.atualizar_salarios_e_impacto(promocoes_ano, salarios_iniciais_mensais, promocoes_acumuladas)[0])
            folhas_pagamentos[ano] = np.sum(NPCS.atualizar_salarios_e_impacto(promocoes_ano, salarios_iniciais_mensais, promocoes_acumuladas)[0])
            folhas_pagamentos_enxutas[ano] = np.sum(NPCS.atualizar_salarios_e_impacto(promocoes_ano, salarios_iniciais_mensais, promocoes_acumuladas)[2])
            
            salarios_iniciais_mensais = salario_bruto_mensal_from_anual(NPCS.atualizar_salarios_e_impacto(promocoes_ano, salarios_iniciais_mensais, promocoes_acumuladas)[0])
            impactos_exercicios[ano] = NPCS.atualizar_salarios_e_impacto(promocoes_ano, salarios_iniciais_mensais, promocoes_acumuladas)[1]

        tempo_para_topar = int(79 / mu)
        impacto_total = np.sum(impactos_exercicios[:tempo_para_topar])
        limite_impacto = 0.01 * np.sum(folhas_pagamentos[:tempo_para_topar])
        limite_impacto_enxuto = 0.01 * np.sum(folhas_pagamentos_enxutas[:tempo_para_topar])
        if conservador:
            limite = limite_impacto_enxuto
        else:
            limite = limite_impacto
        if impacto_total <= limite:
            cpgar52_atendida = True
            print(f"Impacto total dentro do limite de 1% da folha salarial.")
            print(f"Limite de impacto na folha de pagamento: {locale.currency(limite, grouping=True)}")
            print(f"Impacto total na folha salarial: {locale.currency(impacto_total, grouping=True)}")
            print(f"Média de steps ganhos por funcionário por ano: {mu:.2f}")
            print(f"Salário médio bruto anual final: {locale.currency(np.mean(novos_salarios_anual_exercicios_media[-1]), grouping=True)}")
            print(f"Salário médio bruto mensal final: {locale.currency(np.mean([salario_bruto_mensal_from_anual(novos_salarios_anual_exercicios_media[-1])]), grouping=True)}")
            print(f"Número médio de steps adquiridos em {num_anos} anos: {np.mean(promocoes_acumuladas):.2f}")
            print(f"Aumento percentual médio do salário após {num_anos} anos: {((np.mean(salario_bruto_mensal_from_anual(novos_salarios_anual_exercicios_media[-1])) - salario_inicial_1A) / salario_inicial_1A) * 100:.2f}%")
            for ano in range(num_anos):
                print(f"Após {ano + 1} anos, {porcentagem_teto_salarial[ano]:.2f}% dos funcionários atingiram o teto salarial.")
                print(f"Após {ano + 1} anos, o salário médio bruto mensal é de {locale.currency(salario_bruto_mensal_from_anual(novos_salarios_anual_exercicios_media[ano]), grouping=True)}.")
                
        else:
            print(f"Impacto médio na folha salarial: {locale.currency(impacto_total, grouping=True)}")
            print(f"Limite de impacto na folha de pagamento: {locale.currency(limite, grouping=True)}")
            excesso = limitar_num((impacto_total - limite) / limite, 0, 0.9)
            mu -= excesso
            mu = limitar_num(mu, 0, 3)
            print(f"Ajustando mu para {mu:.2f}")
    return mu, porcentagem_teto_salarial, salario_bruto_mensal_from_anual(novos_salarios_anual_exercicios_media), promocoes_acumuladas_exercicios

# Executar simulação
mu, porcentagem_teto_salarial, salarios_mensal_exercicios_media, promocoes_acumuladas_exercicios = simular(mu=3.0, num_funcionarios=NUM_FUNCIONARIOS, num_anos=NUM_ANOS, conservador=True)

# Parâmetros da distribuição
lambda_ = mu # Média
inferior = 0  # Limite inferior (truncamento)
superior = 3  # Limite superior (truncamento)

# Gerar valores no intervalo [0, 3] para o eixo x
x = np.linspace(inferior, superior, 1000)

# Criar a distribuição normal truncada
dist = poisson.pmf(x, mu=lambda_)
dist /= dist.sum() # Assegura que a distribuição truncada esteja normalizada

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.bar(x, dist, label=fr'Poisson: $\lambda$ = {lambda_:.2F}', color='blue', linewidth=2)
plt.title('Distribuição de Poisson Truncada', fontsize=16)
plt.xlabel('Níveis de Promoção', fontsize=14)
plt.ylabel('Densidade de Probabilidade', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xlim([inferior, superior])
plt.ylim([0, 1.2 * np.max(dist)])  # Ajustar o limite do eixo y para melhor visualização
plt.savefig("Distribuição de Poisson Truncada para Promoções.png", format="png")


# Função para adicionar legendas às barras (a cada duas barras)
def adicionar_legendas_barras(ax, espacamento_legendas, monetario):
    # Iterar sobre as barras no gráfico
    for i, barra in enumerate(ax.patches):
        # Apenas processar barras com índice par (a cada duas barras)
        if i % espacamento_legendas == 0:
            # Obter o índice (ano) correspondente à barra
            indice = int(barra.get_x() + barra.get_width() / 2)
            
            # Somente as barras indicadas devem ser "legendadas"
            if 1 <= indice <= NUM_ANOS:
                # Pegue as posições X e Y da legenda de cada barra
                y_pos = barra.get_height() # valores de Y
                x_pos = barra.get_x() + barra.get_width() / 2 # Valores de X
                
                # Número de pontos entre a barra e a legenda; mude conforme sua preferência
                espaço = 5
                # Alinhamento vertical
                av = 'bottom'
                
                # Usar valores de Y como legenda e formatar números com duas casas decimais:
                if monetario:
                    legenda = locale.currency(y_pos, grouping=True)
                else:
                    legenda = '{:.2f}'.format(y_pos)
                
                # Criar a legenda
                ax.annotate(
                    legenda,                      # Use legenda como legenda
                    (x_pos, y_pos),               # Coloque a legenda no topo da barra
                    xytext=(0, espaço),           # Espaço a legenda verticalmente por espaço
                    textcoords='offset points',   # Interpretar xytext como offset em pontos
                    ha='center',                  # Legenda horizontalmente alinhada no centro
                    va=av,                        # Alinhamento vertical
                    size=10,                      # Tamanho da fonte da legenda
                    rotation=-45                   # Rotação em graus do texto da legenda
                )

# Calcular a média da porcentagem de funcionários que atingiram o teto salarial ao longo dos anos

# Plotar o gráfico de barras
anos = np.arange(1, NUM_ANOS + 1)  # Eixo X: anos de 1 a NUM_ANOS
fig, ax = plt.subplots(figsize=(12, 6))  # Criar figura e eixo
largura = 0.8  #  Largura das barras, valores pequenos faz com que o espaçamento entre elas seja maior
bars = ax.bar(x=anos, height=porcentagem_teto_salarial, color='blue', alpha=0.7, width=largura)  # Plotar as barras
ax.set_ylim(0, 110)
ax.set_title('Porcentagem de Funcionários que Atingiram o Topo da Carreira ao Longo dos Anos', fontsize=16)
ax.set_xlabel('Ano', fontsize=14)
ax.set_ylabel('Porcentagem de Funcionários no Topo (%)', fontsize=14)
ax.set_xticks(anos[::1])  # Mostrar rótulos de 1 em 1 ano1 no eixo X
ax.set_xticklabels(labels=anos[::1], fontsize=6, rotation=0)  # Rotacionar os rótulos do Eixo X e configurar o  tamanho da fonte para melhor legibilidade
ax.grid(True, linestyle='--', alpha=0.7)

              
# Chamar a função para adicionar legendas
adicionar_legendas_barras(ax, espacamento_legendas=2, monetario=False)

# Salvar a figura
plt.savefig("Porcentagem de Funcionários que Atingiram o Topo da Carreira ao Longo dos Anos.png", format="png", bbox_inches='tight')


# Calcular a média dos salários brutos dos funcionários ao longo dos anos

# Plotar o gráfico de barras
anos = np.arange(1, NUM_ANOS + 1)  # Eixo X: anos de 1 a NUM_ANOS
fig, ax = plt.subplots(figsize=(12, 6))  # Criar figura e eixo
largura = 0.8  #  Largura das barras, valores pequenos faz com que o espaçamento entre elas seja maior
bars = ax.bar(x=anos, height=calcular_salario_liquido_from_bruto_mensal(salarios_mensal_exercicios_media), color='blue', alpha=0.7, width=largura)  # Plotar as barras
ax.set_ylim(20000, np.max(calcular_salario_liquido_from_bruto_mensal(salarios_mensal_exercicios_media)) * 1.1)
ax.set_title('Salário Líquido Médio Anualizado dos Funcionários ao Longo dos Anos', fontsize=16)
ax.set_xlabel('Ano', fontsize=14)
ax.set_ylabel('Salário Médio Líquido Anualizado (R$)', fontsize=14)
ax.set_xticks(anos[::1])  # Mostrar rótulos de 1 em 1 ano1 no eixo X
ax.set_xticklabels(labels=anos[::1], fontsize=6, rotation=0)  # Rotacionar os rótulos do Eixo X e configurar o  tamanho da fonte para melhor legibilidade
ax.grid(True, linestyle='--', alpha=0.7)

              
# Chamar a função para adicionar legendas
adicionar_legendas_barras(ax, espacamento_legendas=5, monetario=True)

# Salvar a figura
plt.savefig("Salário Bruto Médio dos Funcionários ao Longo dos Anos.png", format="png", bbox_inches='tight')


# Calcular a média dos steps concedidos ao longo dos anos

# Plotar o gráfico de barras
anos = np.arange(1, NUM_ANOS + 1)  # Eixo X: anos de 1 a NUM_ANOS
fig, ax = plt.subplots(figsize=(12, 6))  # Criar figura e eixo
largura = 0.8  #  Largura das barras, valores pequenos faz com que o espaçamento entre elas seja maior
bars = ax.bar(x=anos, height=promocoes_acumuladas_exercicios, color='blue', alpha=0.7, width=largura)  # Plotar as barras
ax.set_ylim(0, np.max(promocoes_acumuladas_exercicios) * 1.1)
ax.set_title('Número de Steps Médio Concedidos ao Longo dos Anos', fontsize=16)
ax.set_xlabel('Ano', fontsize=14)
ax.set_ylabel('Steps Concedidos', fontsize=14)
ax.set_xticks(anos[::1])  # Mostrar rótulos de 1 em 1 ano1 no eixo X
ax.set_xticklabels(labels=anos[::1], fontsize=6, rotation=0)  # Rotacionar os rótulos do Eixo X e configurar o  tamanho da fonte para melhor legibilidade
ax.grid(True, linestyle='--', alpha=0.7)

              
# Chamar a função para adicionar legendas
adicionar_legendas_barras(ax, espacamento_legendas=3, monetario=False)

# Salvar a figura
plt.savefig("Número de Steps Médio Concedidos ao Longo dos Anos.png", format="png", bbox_inches='tight')


tf = time.time()
print(f'Execução concluída após: {(tf - t0):.2f} segundos')