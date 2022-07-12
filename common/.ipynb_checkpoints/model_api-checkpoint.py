import json
from fastapi import Request
from fastapi import Response
from typing import Union, List
from pydantic import BaseModel
import joblib
import config

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# config test
lstm_config_data = config.lstm_model_parameter



# get user`s input data in web
class config(BaseModel):
    input_size: Union[int, None] = None
    output_size: Union[int, None] = None
    window_size: Union[str, None] = None
    lstm_layers: Union[int, None] = None
    hidden_size: Union[float, None] = None
    dropout: Union[float, None] = None
    epochs: Union[int, None] = None
    batch_size: Union[int, None] = None
    device: Union[str, None] = None
    
    
@app.get("/model_config")    # get model_config parameter data to web
async def config(parameter:BaseModel):
    
     # 비동기 형태로 데이터를 json형태로 전송받는다
    input_param = await parameter.json()
    
    return input_param
    
    
@app.post("/model")    # set router for handling data to web server
async def lstm_model(parameter: Request):
    
    '''
    if user load file data and then we can get user`s loaded data after that 
    Continue preprocessing then model learning, model distribution
    
    get model parameter who selected in our web site
    
    get data for lstm_model parameter
    "input_size" : 1,  # 데이터 변수 개수, int
    "output_size" : 1, # 데이터 변수 개수, int
    "window_size" : 48,  # input sequence의 길이, int
    "lstm_layers" : 3, # layer 깊이 수
    "hidden_size" : 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
    "dropout" : 0.2,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
    "epochs" : 200,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
    "batch_size" : 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
    "device" : 'cuda'  # 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
    '''
    
    
    def get_keras_model(config):
    input1 = Input(shape=(config['time_step'], config['input_size']))
    lstm = input1
    for i in range(config['lstm_layers']):
        lstm = LSTM(units=config['hidden_size'],dropout=config['dropout'],return_sequences=True)(lstm)
    output = Dense(config['output_size'])(lstm)
    model = Model(input1, output)
    model.compile(loss='mse', optimizer='adam', metrics=["mae"])     # metrics=["mae"]
    return model


    def gpu_train_init(config):
        if config['device'] == 'cuda'
            sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 최대 70% gpu 사용
            sess_config.gpu_options.allow_growth=True   # gpu 사용량이 부족할 시 자동으로 분배
            sess = tf.Session(config = sess_config)
            set_session(sess)
        else:
            pass
    

    def train(config, train_and_valid_data):
        if config.use_cuda: gpu_train_init()
        train_X, train_Y, valid_X, valid_Y = train_and_valid_data
        model = get_keras_model(config)
        model.summary()
        if config.add_train:
            model.load_weights(config['best_model_path'] + config['model_name'])

        callbacks = [
        TensorBoard(log_dir=logdir),
        EarlyStopping(monitor="val_loss", patience=config['patience'], mode='auto'),
        ModelCheckpoint(filepath=config['best_model_path'] + config['model_name'],
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='auto')]


        await model.fit(train_X, train_Y, batch_size=config['batch_size'], epochs=config['epochs'], verbose=1,
                  validation_data=(valid_X, valid_Y), callbacks=callbacks)


    def predict(config, test_X):
        '''
        get the best_model param and path then starting predict
        config is user`s input param
        '''
        model = get_keras_model(config)
        model.load_weights(config['best_model_path'] + config['model_name'])
        result = model.predict(test_X, batch_size=1)
        result = result.reshape((-1, config['output_size']))
        return result
    
     
    res_data = jsonable_encoder(json.dumps(model, ensure_ascii=False))
    return Response(content=res_data, media_type="application/json") 