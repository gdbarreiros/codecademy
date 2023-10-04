# US Medical Insurance

#%%
# Vamos começar importando a biblioteca pandas e abrindo o arquivo usando o métod read_csv()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df_insurance = pd.read_csv('insurance.csv')

#%%
# Verificando as 10 primeiras linhas do conjunto de dados

df_insurance.head(10)

#%%
# Verificando as 10 últimas linhas do conjunto de dados

df_insurance.tail(10)

# As informações contidas no conjunto de dados são: 
    # - age	
    # - sex	
    # - bmi	
    # - children	
    # - smoker	
    # - region	
    # - charges

#%%
# Vamos calcular algumas estatísticas descritivas (média, mediana, máximo, mínimo, assimetria), 
# ao menos para as colunas quantitativas

df_insurance.agg({'age': ['mean', 'median', 'min', 'max', 'skew']
                  ,'bmi': ['mean', 'median', 'min', 'max', 'skew']
                  ,'charges': ['mean', 'median', 'min', 'max', 'skew']
                  ,'children': ['mean', 'median', 'min', 'max', 'skew']})

#%%
# Vamos plotar a distribuição de algumas variáveis, para ver o seu comportamento
# Usamos o método axvline() para incluir uma linha horizontal no gráfico, cuja posição é a mediana 
# - que foi calculada usando o método quantile() da biblioteca numpy 

plt.hist(df_insurance['age'], bins = 15, color = 'blue')
plt.title('Distribuição das idades')
plt.axvline(x = np.quantile(df_insurance['age'], 0.5), color = 'black', label = '2º Quartil (Mediana)')
plt.legend()
plt.show()

plt.hist(df_insurance['bmi'], bins = 10, color = 'gray')
plt.title('Distribuição dos BMIs (IMCs)')
plt.axvline(x = np.quantile(df_insurance['bmi'], 0.5), color = 'black', label = '2º Quartil (Mediana)')
plt.legend()
plt.show()

plt.hist(df_insurance['charges'], bins = 10, color = 'green')
plt.title('Distribuição do valor das cobranças')
plt.axvline(x = np.quantile(df_insurance['charges'], 0.5), color = 'black', label = '2º Quartil (Mediana)')
plt.legend()
plt.show()

plt.hist(df_insurance['children'], bins = 5, color = 'red')
plt.title('Distribuição da quantidade de filhos')
plt.axvline(x = np.quantile(df_insurance['children'], 0.5), color = 'black', label = '2º Quartil (Mediana)')
plt.legend()
plt.show()

# Com base nas distribuições, é possível perceber que os BMIs são os dados que mais se aproximam de uma 
# distribuição normal (gaussiana). As idades praticamente não possuem variação, com exceção da faixa
# entre 20 a 23 anos que tem um número elevado de observações. Por fim, as cobranças e a 
# quantidade de filhos possuem uma distribuição assimétrica à direita, o que significa que a moda < mediana < média

# %%
# Seguindo algumas sugestões propostas no site do Codecademy, vamos realizar algumas análises um pouco 
# mais específicas

# Em primeiro lugar, vamos verificar como está a distribuição especial do nosso dataset
    # - Vamos ver quais são as regiões distintas

print(df_insurance['region'].unique())

# ['southwest' 'southeast' 'northwest' 'northeast']

# Vamos ver qual dessas regiões possui mais pessoas. Podemos fazer isso agrupando o conjunto de dados 
# pela coluna 'region' e depois usando a função de agregação count. Usamos um subconjunto do dataframe, 
# já que se usássemos ele inteiro as informações se repetiriam ao longo das colunas. Depois disso, renomeamos
# a coluna para se chamar 'count' e ordenamos os valores do menos para o maior

df_insurance_grouped = df_insurance[['region', 'age']].groupby(['region']).count()
df_insurance_grouped = df_insurance_grouped.rename(columns = {'age': 'count'})
df_insurance_grouped = df_insurance_grouped.sort_values(by = ['count'], ascending = False)
df_insurance_grouped

# Com isso, podemos ver que a região que possui mais observações nesse conjunto de dados é southeast, 
# com 364 observações

# %%
# Em segundo lugar, vamos avaliar qual a impacto do hábito do cigarro nos custos do seguro médico
# Vamos dividir o dataframe em fumantes x não-fumantes

df_non_smokers = df_insurance[df_insurance['smoker'] != 'yes']
df_smokers = df_insurance[df_insurance['smoker'] == 'yes']

# A quantidade de pessoas não-fumantes nesse conjunto de dados é muito maior (por volta de 5x) que a 
# quantidade de pessoas fumantes, o que pode invalidar nossas conclusões. O ideal seria termos uma 
# distribuição equivalente entre os dois conjuntos. De toda forma, vamos seguir com as análises

df_joined = pd.DataFrame()
df_joined['non_smoker_charges'] = df_non_smokers.agg({'charges': ['mean', 'median', 'min', 'max', 'skew']})
df_joined['smoker_charges'] = df_smokers.agg({'charges': ['mean', 'median', 'min', 'max', 'skew']})
df_joined['diff'] = df_joined['smoker_charges'] - df_joined['non_smoker_charges']
df_joined

# Com base nessa análise, lembrando dos apontamentos anteriores, é notável a diferença no custo do seguro 
# médico de pessoas fumantes x pessoas não-fumantes. Em média, pessoas que não fumam pagam ~U$27k a menos 
# que pessoas fumantes, um número bastante impressionante

#%%
# Além dessas análises, eu também gostaria de calcular qual o coeficiente de correlação de Pearson das variáveis
# qualitativas em relação ao custo ('charges') e também plotar um gráfico de dispersão, para visualiza-la graficamente

