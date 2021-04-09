#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask, request, url_for, jsonify
from database import *
app = Flask(__name__)
db = LogLoader("ailoganalyzer_db")
from datetime import datetime


@app.route('/api/log')
def api_log():
    param = dict(request.args)
    fields = None
    if fields in param:
        field = param["fields"].split(",")
        param.pop("fields")
    if "start_time" in param:
        param["start_time"] = datetime.strptime(param["start_time"],"%Y-%m-%d-%H.%M.%S.%f")
    if "end_time" in param:
        param["end_time"] = datetime.strptime(param["end_time"],"%Y-%m-%d-%H.%M.%S.%f")
    if "abnormal" in param:
        param["abnormal"] = (param["abnormal"] == "True")
    log_ls = db.find("logs", param,fields)
    return jsonify(log_ls)

    return "Helloe !"

@app.route('/api/count')
def api_count_log():
    param = dict(request.args)
    sec_shift = param["sec"]
    filters = {
        "end_time" : datetime.now(),
        "start_time":
    }


if __name__ == '__main__':
    app.run(debug=True)
