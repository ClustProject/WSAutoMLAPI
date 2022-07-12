import json
from fastapi import Request
from fastapi import Response
from fastapi import status
from typing import Union, List
from pydantic import BaseModel
import joblib
import config

import pandas as pd
import numpy as np
import os

config = config.preprocessing

@app.get("/get_file")
async def get_load_file(req_file:Request):
    
    # 비동기 형태로 데이터를 json형태로 전송받는다
    input_data = await req_file.json()
    
    # input_data convert to dataframe
    df = pd.read_json(input_data)

    return df


@app.post("/preprocessing")    # set router for handling data to web server
async def preprocessing(df):
    
    '''
    Reading user uploaded file_data which data is json type
    convert to dataframe then starting preprocessing
    
    importent parameter key in preprocessing
    conzon_id = "콘존ID"
    vds_id = "VDS_ID"
    '''
    df = get_load_file()
    
    
    # start data preprocessing--------------
    # checking for models key and user file_data is enough or not
    
    checking_features = set(joblib.load(config['best_model_path'] + config['model_name']).keys()) - set(df)
    
    # cheking is enough features
    '''
    conzon_id: '콘존ID'
    vds_id: 'VDS_ID'
    기준으로 해당 컬럼 존재 시 자동 merge
    '''
    
    if len(checking_features) > 0:
        print(checking_features)

        if conzon_id in checking_features:
            # join conzon master file code
            print("merged_df")
            combinede_df = Preprocessing.combineded_df(df)

        elif vds_id in checking_features:
            # join vds master file code
            print("merged_df")
            combinede_df = df.Preprocessing.combineded_df(df)

    '''
    call the preprocessing module
    start preprocessing
    remove_missing_values: remove nan value and ''
    '''
    removed_df = Preprocessing.remove_missing_values(combinede_df)
    labeled_df = Preprocessing.label_encode(removed_df)
    LSTM_df = Preprocessing.convert_to_datetime(labeled_df)
    LSTM_df = LSTM_df.set_index(config['index'])
    LSTM_df.drop(config['drop_col'], axis=1, inplace=True)
    
    
    # feature_data 와 label_data 기반으로 window_size 길이의 input으로 미래의 데이터를 예측하는 데이터 생성
    def make_dataset(data, label, window_size=72):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
    
    
    # response finished preprocessing data to web 
    # the data_type is json
    # ensure_ascii = False == 문자들이 있는 그대로 출력, True == escape
    res_data = jsonable_encoder(json.dumps(LSTM_df, ensure_ascii=False))
    
    # json 형태로 web server에 데이터 통신
    return Response(content=res_data, media_type="application/json") 