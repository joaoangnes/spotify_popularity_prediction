from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)
currentModel = 'RandomForestRegressor'

def getCsv():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("../datasource/spotify_most_streamed_songs.csv")

@app.route('/fit-model', methods=['POST'])
def fit_model():
    global currentModel
    
    currentModel = request.form['model']

    print(request.form)
    return redirect(url_for('change_model'))

@app.route('/change-model', methods=["GET"])
def change_model():
    global currentModel

    data = {
        'model': "RandomForestRegressor",
        'params': {
            'max_depth': 30,
            'min_samples_leaf': 4,
            'min_samples_split': 2,
            'n_estimators': 100,
        }
    }

    match request.args.get('model', currentModel):
        case 'RandomForestRegressor':
            data = {
                'model': 'RandomForestRegressor',
                'params': {
                    'max_depth': 0,
                    'min_samples_leaf': 0, 
                    'min_samples_split': 0, 
                    'n_estimators': 0
                }
            }
        case 'GradientBoostingRegressor':
            data = {
                'model': 'GradientBoostingRegressor',
                'params': {
                    'learning_rate': 0.05, 
                    'max_depth': 3, 
                    'n_estimators': 100
                }
            }
        case 'LightGBM':
            data = {
                'model': 'LightGBM',
                'params': {
                    'learning_rate': 0.1, 
                    'max_depth': 3, 
                    'n_estimators': 50, 
                    'num_leaves': 31
                }
            }
        case 'AdaBoostRegressor':
            data = {
                'model': 'AdaBoostRegressor',
                'params': {
                    'learning_rate': 0.01, 
                    'n_estimators': 100
                }
            }

    return render_template(template_name_or_list='change_model.html', data=data)


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

@app.route('/a/<string:name>', methods=["GET"])
def artist(name: str):
    name = name.replace('+', ' ')
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


