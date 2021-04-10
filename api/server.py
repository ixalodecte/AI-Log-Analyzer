#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask, request, url_for, jsonify
from flask.json import JSONEncoder
import calendar
from database import *
app = Flask(__name__)
db = LogLoader("ailoganalyzer_db")
from datetime import datetime
from pandas import date_range

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

@app.route('/api/logs/<action>')
def api_log(action):
    action = str(action)
    param = dict(request.args)
    fields = None
    count = False

    if "fields" in param:
        field = param["fields"].split(",")
        param.pop("fields")
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
        log_ls = db.find("logs", param,fields)
        return jsonify(log_ls)


if __name__ == '__main__':
    app.run(debug=True)
#test : http://localhost:5000/api/logs/get?abnormal=True&start_time=2005-06-03-16.12.34.557453&end_time=2005-06-04-03.17.40.435081
