#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask, request, url_for, jsonify, session, render_template, flash, redirect, abort
import os
from flask.json import JSONEncoder
from database import LogLoader
app = Flask(__name__)
db = LogLoader("ailoganalyzer_db")
from datetime import datetime
from pandas import date_range

user = user ={"username":"admin", "password":"admin"}
print(user)

# Change le format de la date (pour inclure les microsecondes)
class CustomJSONEncoder(JSONEncoder):

    def default(self, obj):
        try:
            if isinstance(obj, datetime):
                return obj.isoformat()
        except TypeError:
            pass
        return JSONEncoder.default(self, obj)

app.json_encoder = CustomJSONEncoder

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        param = dict(request.args)
        if "filters" in param:
            filters = param["filters"].split(" ")
            filters = "&".join(filters)
            filters = "&" + filters
        else:
            filters = "&end_time=" + datetime.strftime(datetime.now(),"%Y-%m-%d-%H.%M.%S.%f")
        return render_template("test.html", filters = filters)

@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == user['password'] and request.form['username'] == user['username']:
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return home()

@app.route('/api/logs/<action>')
def api_log(action):
    if session.get('logged_in'):
        action = str(action)
        param = dict(request.args)
        fields = None
        limit = None

        if "fields" in param:
            fields = param["fields"].split(",")
            param.pop("fields")
        if "limit" in param:
            limit = int(param["limit"])
            param.pop("limit")
        if "start_time" in param:
            param["start_time"] = datetime.strptime(param["start_time"],"%Y-%m-%d-%H.%M.%S.%f")
        if "end_time" in param:
            param["end_time"] = datetime.strptime(param["end_time"],"%Y-%m-%d-%H.%M.%S.%f")
        if "abnormal" in param:
            param["abnormal"] = (param["abnormal"] == "True")


        if action == "count":
            if "number" in param:
                number = int(param["number"])
                param.pop("number")
            else:
                return "bad parameters"
            dates = date_range(start = param["start_time"], end = param["end_time"], periods = number)
            time_serie = db.time_serie("logs", dates, param)

            return jsonify(time_serie)

        if action == "get":
            print("hello")
            log_ls = db.find("logs", param,fields, limit=limit)
            return jsonify(log_ls)
    return home()

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()

@app.route("/visualisation.html")
def visualisation():
    return render_template("visualisation.html")


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=4000)
#test : http://localhost:4000/api/logs/get?abnormal=True&start_time=2005-06-03-16.12.34.557453&end_time=2005-06-04-03.17.40.435081
