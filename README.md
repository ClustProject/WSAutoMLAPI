# WSAutoMLAPI

교통 시계열 데이터를 입력으로 받아 미래 데이터 평균속도 값을 예측하는 time series forecasting 대한 설명
입력 데이터 형태 : 다변량 시계열 데이터

웹페이지 구현 후 API 형태로 사용자에게 input_data_file을 입력받아 모델학습 및 데이터 전처리 진행

lstm_model_parameter = {
    "best_model_path": './check',  # 학습 완료 모델 저장 경로
    "model_name": '/lstm.h5'
    "parameter": {
        "input_size" : 1,  # 데이터 변수 개수, int
        "output_size" : 1, # 데이터 변수 개수, int
        "window_size" : 48,  # input sequence의 길이, int
        "time_step" : 1, # 예측할 길이, int
        "patience" : 50, # 학습 시 참는 범위 설정
        "lstm_layers" : 3, # layer 깊이 수
        "forecast_step" : 1,  # 예측할 미래 시점의 길이, int
        "hidden_size" : 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
        "dropout" : 0.2,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        "epochs" : 200,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
        "batch_size" : 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
        "device" : 'cuda'  # 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
    }
}

preprocessing = {
    "conzon_id" = "콘존ID"
    "vds_id" = "VDS_ID"
    "drop_col" = ['집계일자', '집계일', '집계시분']
    "index" = "집계날"
}

