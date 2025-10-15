# FarmTech Solutions - Sistema de Visão Computacional com YOLO

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Sobre o Projeto

Este projeto faz parte da **Fase 6, Capítulo 1** do curso FIAP e tem como objetivo desenvolver um sistema de visão computacional para a **FarmTech Solutions**, uma startup do agronegócio que utiliza tecnologia de ponta para otimizar processos agrícolas.

O sistema implementa detecção de objetos utilizando **YOLOv8 (You Only Look Once)**, comparando diferentes abordagens de treinamento e validando a eficácia do modelo customizado.

### 🎯 Objetivos

- **Entrega 1**: Treinar um modelo YOLO customizado e comparar resultados com 30 e 60 épocas de treinamento
- **Entrega 2**: Comparar três abordagens: YOLO customizado, YOLO padrão e CNN do zero
- Documentar todo o processo com visualizações e análises comparativas
- Criar um repositório completo no GitHub com documentação técnica

## 📊 Dataset

O dataset utilizado contém **82 imagens** divididas em **2 classes**:

- **Gato (Cat)**: 41 imagens
- **Cachorro (Dog)**: 41 imagens

### Estrutura do Dataset

```
dataset/
├── cat/
│   ├── train/          # 32 imagens de treino
│   ├── validation/     # 4 imagens de validação
│   └── test/           # 4 imagens de teste
└── dog/
    ├── train/          # 32 imagens de treino
    ├── validation/     # 4 imagens de validação
    └── test/           # 4 imagens de teste

labels/
└── *.txt              # 41 arquivos de anotações em formato YOLO
```

### Formato das Anotações

As anotações seguem o formato YOLO padrão:
```
class_id x_center y_center width height
```

Todos os valores são normalizados entre 0 e 1.

## 🚀 Tecnologias Utilizadas

- **Python 3.8+**: Linguagem de programação principal
- **YOLOv8 (Ultralytics)**: Framework de detecção de objetos
- **Google Colab**: Ambiente de execução em nuvem com GPU
- **PyTorch**: Framework de deep learning
- **OpenCV**: Processamento de imagens
- **Matplotlib/Seaborn**: Visualização de dados
- **Pandas/NumPy**: Manipulação de dados

## 📁 Estrutura do Projeto

```
fase_6_cap_1/
├── dataset/                    # Dataset original (cat/dog)
│   ├── cat/
│   └── dog/
├── labels/                     # Anotações YOLO existentes
│   └── *.txt
├── ItaloDomingues_RM561787_pbl_fase6.ipynb    # Notebook principal do projeto
├── enunciado.md                # Descrição completa do desafio
└── README.md                   # Este arquivo
```

## 💻 Como Usar

### Pré-requisitos

1. Conta Google (para acesso ao Google Colab)
2. Dataset e labels organizados no Google Drive
3. Conexão com internet para download dos modelos YOLO

### Passo a Passo

1. **Prepare o Google Drive**
   - Faça upload das pastas `dataset/` e `labels/` para o Google Drive
   - Anote o caminho das pastas (ex: `/content/drive/MyDrive/dataset`)

2. **Abra o Notebook no Colab**
   - Faça upload do arquivo `ItaloDomingues_RM561787_pbl_fase6.ipynb` para o Google Colab
   - Ou abra diretamente pelo Google Drive

3. **Configure os Caminhos**
   - Na célula de configuração (Seção 3), ajuste os caminhos:
   ```python
   DATASET_SOURCE = '/content/drive/MyDrive/dataset'  # ⬅️ AJUSTE AQUI!
   LABELS_SOURCE = '/content/drive/MyDrive/labels'    # ⬅️ AJUSTE AQUI!
   ```

4. **Execute o Notebook**
   - Execute as células sequencialmente (Runtime > Run all)
   - O notebook irá:
     - Instalar dependências
     - Conectar ao Google Drive
     - Preparar o dataset no formato YOLO
     - Treinar modelos com 30 e 60 épocas
     - Validar e testar os modelos
     - Gerar visualizações e análises comparativas

5. **Analise os Resultados**
   - Os resultados de treinamento ficam salvos em `/content/runs/detect/`
   - Métricas, gráficos e predições são gerados automaticamente
   - Compare os resultados entre 30 e 60 épocas

## 🔬 Metodologia

### 1. Preparação dos Dados

O notebook implementa um sistema inteligente de conversão de dados:

- **Conversão de Estrutura**: De estrutura baseada em classes (cat/, dog/) para estrutura baseada em splits (train/, val/, test/)
- **Sistema Inteligente de Labels**:
  - Utiliza labels existentes da pasta `labels/` quando disponíveis
  - Cria labels padrão (bounding box centralizado) para imagens sem anotação
  - Reporta estatísticas de labels utilizados vs criados

### 2. Treinamento

Dois modelos são treinados a partir do YOLOv8n pré-treinado:

- **Modelo 1**: 30 épocas de treinamento
- **Modelo 2**: 60 épocas de treinamento

**Parâmetros de Treinamento**:
- Modelo base: `yolov8n.pt` (nano)
- Tamanho da imagem: 640x640
- Batch size: 16
- Otimizador: Adam
- Paciência (early stopping): 50 épocas

### 3. Validação

Durante o treinamento, o modelo é validado a cada época utilizando:
- Conjunto de validação (4 imagens por classe)
- Métricas: Precision, Recall, mAP50, mAP50-95

### 4. Teste

Após o treinamento, os modelos são testados:
- Conjunto de teste independente (4 imagens por classe)
- Geração de predições com bounding boxes
- Análise de confiança e acurácia

### 5. Análise Comparativa

Comparação entre os dois modelos considerando:
- Métricas de performance (mAP, Precision, Recall)
- Tempo de treinamento
- Qualidade das predições
- Trade-off entre acurácia e tempo de treinamento

## 📈 Resultados Esperados

### Métricas de Avaliação

- **mAP50**: Mean Average Precision com IoU threshold de 0.5
- **mAP50-95**: mAP médio com thresholds de 0.5 a 0.95
- **Precision**: Taxa de detecções corretas
- **Recall**: Taxa de objetos detectados

### Visualizações Geradas

1. **Curvas de Treinamento**: Loss, Precision, Recall ao longo das épocas
2. **Matriz de Confusão**: Classificação entre as classes
3. **Predições com Bounding Boxes**: Visualização das detecções
4. **Comparação 30 vs 60 épocas**: Análise de performance

## 📓 Estrutura do Notebook

O notebook `ItaloDomingues_RM561787_pbl_fase6.ipynb` está organizado em 7 seções principais:

1. **Introdução**: Contexto do projeto e objetivos
2. **Instalação e Importações**: Setup do ambiente
3. **Conexão com Google Drive**: Acesso aos dados
4. **Preparação do Dataset**: Conversão para formato YOLO
5. **Treinamento**: Modelos com 30 e 60 épocas
6. **Validação**: Avaliação dos modelos
7. **Testes e Análise**: Comparação e conclusões

Total: **26 células** (código + markdown)
---

**FarmTech Solutions** - Tecnologia a serviço do agronegócio 🌾
