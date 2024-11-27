from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)
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

currentModel = 'RandomForestRegressor'
currentParams = {
    'max_depth': 30,
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'n_estimators': 100,
}

def getCsv():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("../datasource/spotify_most_streamed_songs.csv")

# ao ser chamado treina o novo modelo selecionado usando os parametros definidos pelo usuário
# também define o modelo atual sendo utilizado
@app.route('/fit-model', methods=['POST'])
def fit_model():
    global currentModel, currentParams

    # define o novo modelo selecionado
    args = request.form.to_dict()
    currentModel = args.pop("model")
    currentParams = args

    # salva o modelo caso queira retornar ao mesmo
    availableModels[currentModel]['params'] = currentParams

    return redirect(url_for('change_model'))

# tela para trocar de modelo
@app.route('/change-model', methods=["GET"])
def change_model():
    global currentModel, currentParams

    # modelo usando as configuraçoes atuais
    model = {
        'model': currentModel,
        'params': currentParams
    }

    # busca o modelo selecionado
    # definindo uma configuração padrão para o mesmo
    modelName = request.args.get('model', None)
    if modelName: model = availableModels[modelName]

    return render_template(template_name_or_list='change_model.html', data=model)


@app.route('/', methods=["GET"])
def home():
    df = getCsv()
    for col in df:
        print(col)
        
    dic = {}
    for _, val in df.iterrows():
        if not str(val["released_year"]) in dic.keys():
            dic[str(val["released_year"])] = 0
        try:
            dic[str(val["released_year"])] += int(val["streams"])
        except:
            continue
        
    dic = dict(sorted(dic.items()))
        
    labels = list(dic.keys())
    data = list(dic.values())
    
    return render_template(template_name_or_list='home.html', labels=labels, data=data)

@app.route('/artist_select', methods=["GET"])
def artist_select():
    return render_template(template_name_or_list='artist_select.html')

@app.route('/a', methods=["GET"])
def artist():
    name = request.args['name'].replace('+', ' ')
    df = getCsv()
    for col in df:
        print(col)
        
    dic = {}
    for _, val in df.iterrows():
        if val["artist(s)_name"] == name:
            if not str(val["released_year"]) in dic.keys():
                dic[str(val["released_year"])] = 0
            try:
                dic[str(val["released_year"])] += 1
            except:
                continue
        
    dic = dict(sorted(dic.items()))
        
    labels = list(dic.keys())
    data = list(dic.values())
    
    return render_template(template_name_or_list='artist.html', name=name, labels=labels, data=data)

@app.route('/p', methods=["GET"])
def platform():
    df = getCsv()
    for col in df:
        print(col)
        
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
    i = 0
    for _, val in df.iterrows():
        try:
            playlists["spotify"] += int(val["in_spotify_playlists"])
            playlists["apple"] += int(val["in_apple_playlists"])
            playlists["deezer"] += int(val["in_deezer_playlists"])
            
            charts["spotify"] += int(val["in_spotify_charts"])
            charts["apple"] += int(val["in_apple_charts"])
            charts["deezer"] += int(val["in_deezer_charts"])
        except:
            continue
        
    playlists = dict(sorted(playlists.items()))
    charts = dict(sorted(charts.items()))
        
    labels = list(playlists.keys())
    playlists = list(playlists.values())
    charts = list(charts.values())
    
    return render_template(template_name_or_list='platforms.html', labels=labels, playlists=playlists, charts=charts)


