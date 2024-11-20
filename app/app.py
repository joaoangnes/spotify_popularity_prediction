from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

def getCsv():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("../datasource/spotify_most_streamed_songs.csv")

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
