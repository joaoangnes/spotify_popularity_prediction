from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from service.model import ModelService
import pandas as pd
import os

app = Flask(__name__)  # Inicializa a aplicação Flask
model_service = ModelService()  # Instância do serviço responsável por manipular os modelos

# Dicionário de modelo para os modelos disponivels no sistema
availableModels = {
    'RandomForestRegressor': {
        'model': 'RandomForestRegressor',
        'params': {
            'max_depth': 30,
            'min_samples_leaf': 4, 
            'min_samples_split': 2, 
            'n_estimators': 100
        }
    },
    'GradientBoostingRegressor': {
        'model': 'GradientBoostingRegressor',
        'params': {
            'learning_rate': 0.05, 
            'max_depth': 3, 
            'n_estimators': 100
        }
    },
    'LightGBM': {
        'model': 'LightGBM',
        'params': {
            'learning_rate': 0.1, 
            'max_depth': 3, 
            'n_estimators': 50, 
            'num_leaves': 31
        }
    },
    'AdaBoostRegressor': {
        'model': 'AdaBoostRegressor',
        'params': {
            'learning_rate': 0.01, 
            'n_estimators': 100
        }
    }
}

# Carrega o arquivo CSV padrão para ser utilizado inicialmente
default_data = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"datasources/spotify_most_streamed_songs.csv"))

# Variáveis globais para armazenar o estado atual do modelo e dados
current_model = model_service.model_dict['name'] 
current_params = model_service.model_dict['params']
current_file_name = "spotify_most_streamed_songs"
current_data = None

# Função para carregar os dados do arquivo atual
def get_data():
    global current_file_name, current_data
    current_data =  pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"datasources/{current_file_name}.csv"))
    return current_data

# Rota para redefinir os dados para o arquivo padrão
@app.route('/reset_data', methods=["GET"])
def reset_data():
    global current_file_name
    current_file_name = "spotify_most_streamed_songs"
    get_data()
    return redirect(url_for('home'))
    
# Rota para adicionar novos dados carregados pelo usuário
@app.route('/add-data', methods=['POST'])
def add_data():
    global current_file_name, current_data
    
    f = request.files['data']  # Recebe o arquivo enviado pelo usuário
    current_data = pd.read_csv(f.stream)  # Lê os dados do arquivo
    current_file_name = "file_loaded"  # Define um nome temporário para o arquivo carregado
    
    print('add data')
    print('current_data', current_data)
    print('current_file_name', current_file_name)
    
    # Salva os dados carregados em um novo arquivo CSV
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"datasources/{current_file_name}.csv")
    current_data.to_csv(file_path, index=False)
        
    return redirect(url_for('home'))

# Rota para treinar o modelo selecionado com os parâmetros informados
@app.route('/fit-model', methods=['POST'])
def fit_model():
    global current_model, current_params, current_data, current_file_name

    # Define o modelo e seus parâmetros a partir do formulário enviado
    args = request.form.to_dict()
    current_model = args.pop("model")
    current_params = args

    # Converte os valores dos parâmetros para os tipos corretos (float ou int)
    for key, value in current_params.items():
        try:
            if '.' in value:
                current_params[key] = float(value)
            else:
                current_params[key] = int(value)
        except ValueError:
            pass 
        
    # Atualiza os parâmetros do modelo selecionado no dicionário global
    availableModels[current_model]['params'] = current_params
    
    # Carrega os dados e treina o modelo
    data = get_data()
    model_service.train(data, current_model, current_params)

    return redirect(url_for('change_model'))

# Rota para trocar o modelo atual
@app.route('/change-model', methods=["GET"])
def change_model():
    global current_model, current_params, model_service
    
    print('change model', current_model)
    print('model_param', current_params)

    # Cria um dicionário representando o modelo atual
    model = {
        'model': current_model,
        'params': current_params
    }

    # print('change model', model)
    
    # Verifica se um novo modelo foi selecionado na URL
    modelName = request.args.get('model', None)
    if modelName: model = availableModels[modelName]

    return render_template(template_name_or_list='change_model.html', data=model)

# Rota principal que exibe um resumo dos dados
@app.route('/', methods=["GET"])
def home():
    # Busca os dados atuais do sistema
    current_data = get_data()
    
    # Gera um resumo dos streams por ano de lançamento
    dic = {}
    for _, val in current_data.iterrows():
        if not str(val["released_year"]) in dic.keys():
            dic[str(val["released_year"])] = 0
        try:
            dic[str(val["released_year"])] += int(val["streams"])
        except:
            continue
        
    dic = dict(sorted(dic.items()))
    
    # Prepara os rótulos (anos) e os dados (stream por anos) para exibição
    labels = list(dic.keys()) 
    data = list(dic.values()) 
    
    return render_template(template_name_or_list='home.html', labels=labels, data=data)

# Rota para visualizar dados filtrados por artista
@app.route('/artist_select', methods=["GET"])
def artist_select():
    return render_template(template_name_or_list='artist_select.html')

# Filtra dados pelo artista informado na URL
@app.route('/a', methods=["GET"])
def artist():
    name = request.args['name'].replace('+', ' ')
    current_data = get_data()
        
    dic = {}
    for _, val in current_data.iterrows():
        if val["artist(s)_name"] == name:
            if not str(val["released_year"]) in dic.keys():
                dic[str(val["released_year"])] = 0
            try:
                dic[str(val["released_year"])] += 1
            except:
                continue
    
    # Ordena o dicionário por ano
    dic = dict(sorted(dic.items()))
    
    # Prepara os rótulos (anos) e os dados (quantidade de lançamentos) para exibição
    labels = list(dic.keys())
    data = list(dic.values())
    
    return render_template(template_name_or_list='artist.html', name=name, labels=labels, data=data)

# Rota para exibir dados por plataforma
@app.route('/p', methods=["GET"])
def platform():
    # Busca os dados atuais do sistema
    current_data = get_data()
    
    # Agrega os dados por playlists e charts de diferentes plataformas
    playlists = {
        "spotify": 0,
        "apple": 0,
        "deezer": 0
    }
    charts = {
        "spotify": 0,
        "apple": 0,
        "deezer": 0
    }
    
    for _, val in current_data.iterrows():
        try:
            playlists["spotify"] += int(val["in_spotify_playlists"])
            playlists["apple"] += int(val["in_apple_playlists"])
            playlists["deezer"] += int(val["in_deezer_playlists"])
            
            charts["spotify"] += int(val["in_spotify_charts"])
            charts["apple"] += int(val["in_apple_charts"])
            charts["deezer"] += int(val["in_deezer_charts"])
        except:
            continue
    
    # Ordena os dicionários por chave (nome da plataforma)
    playlists = dict(sorted(playlists.items()))
    charts = dict(sorted(charts.items()))
    
    # Prepara os rótulos (nomes das plataformas) e os dados para exibição
    labels = list(playlists.keys())
    playlists = list(playlists.values())
    charts = list(charts.values())
    
    return render_template(template_name_or_list='platforms.html', labels=labels, playlists=playlists, charts=charts)

# Rota para exibir as previsões dos dados utilizando o modelo atual
@app.route('/predict-data-view', methods=["GET"])
def predict_data_view():
    current_data = get_data()  
    
    # Faz a previsão e reorganiza as colunas
    predictions = model_service.predict(current_data)
    predictions.rename(columns={'streams': 'Previsão de Streams'}, inplace=True)
    
    columns_order = ['Previsão de Streams'] + [' '] + [col for col in predictions.columns if col != 'Previsão de Streams']
    predictions[' '] = '=>'  # Coluna de espaçamento
    predictions = predictions[columns_order]

    # Gerar HTML da tabela
    table_html = predictions.to_html(classes='data-table', index=False, justify='left', border=0, escape=False)
    
    return render_template('predict_data_view.html', table_html=table_html)
