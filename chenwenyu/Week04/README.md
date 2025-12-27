
[TOC]

### Project Organization
bert_tune_proj/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 主文件
│   ├── bert_model.py    # 训练代码
│   ├── predictor.py     # 预测服务类
│   ├── schemas.py       # Pydantic模型
│   └── models/          # 保存训练好的模型
├── assets
│   ├──models/           #google-bert-chinese model
│   ├──waimai_10k.csv     #数据集
│   ├──post_data_multi.json  #AB测试用内容
│   └──post_data.json        #AB测试用内容
├── README.md             #使用说明，存在的问题等
└── train.py            # 训练入口脚本 

### start train

cd ~/work/bert_tune_proj
python train.py
......
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at ./assets/models/google-bert/bert-base-chinese and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Using device: cpu
------------Epoch: 0 ----------------
Epoch: 0, Average training loss: 1.5514
Accuracy: 0.7768
Average testing loss: 0.8642
-------------------------------
Model saved to ./app/models/bert-finetuned-epoch0
🎉 新的最佳模型保存，准确率: 0.7768
------------Epoch: 1 ----------------
Epoch: 1, Average training loss: 0.5954
Accuracy: 0.8929
Average testing loss: 0.3957
-------------------------------
Model saved to ./app/models/bert-finetuned-epoch1
🎉 新的最佳模型保存，准确率: 0.8929
------------Epoch: 2 ----------------
Epoch: 2, Average training loss: 0.2657
Accuracy: 0.8929
Average testing loss: 0.4232
-------------------------------
------------Epoch: 3 ----------------
Epoch: 3, Average training loss: 0.1442
Accuracy: 0.8750
Average testing loss: 0.6480
-------------------------------

训练完成！最佳准确率: 0.8929

### run service
cd ~/work/bert_tune_proj
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
[1] 21599
(pytorch_d2l) 192:bert_tune_proj wenyuc$ INFO:     Will watch for changes in these directories: ['/Users/wenyuc/work/bert_tune_proj']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [21599] using StatReload
INFO:     Started server process [21602]
INFO:     Waiting for application startup.
尝试加载模型从: app/models/bert-finetuned-epoch0
路径是否存在: True
INFO:app.predictor:Using device: cpu
INFO:app.predictor:模型目录文件: ['model.safetensors', 'label_encoder.pkl', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json', 'vocab.txt']
INFO:app.predictor:开始加载模型...
INFO:app.predictor:模型加载成功
INFO:app.predictor:开始加载分词器...
INFO:app.predictor:分词器加载成功
INFO:app.predictor:标签编码器加载成功，类别: [0, 1]
✅ Model loaded successfully
Predictor type: <class 'app.predictor.BertPredictor'>
Predictor device: cpu
INFO:     Application startup complete.
#### Tips 
查找占用8000端口的进程
lsof -i :8000
或者使用
netstat -anv | grep 8000
### Stop Service
pkill -f uvicorn

### test api
#### 1. 健康检查
curl http://localhost:8000/health
Health check - predictor: <app.predictor.BertPredictor object at 0x12c06b220>
INFO:     127.0.0.1:56039 - "GET /health HTTP/1.1" 200 OK
{"status":"healthy","model_loaded":true,"device":"cpu"}

#### 2. Debug predict return value
curl http://localhost:8000/debug-predict
INFO:     127.0.0.1:56101 - "GET /debug-predict HTTP/1.1" 200 OK
{"type":"<class 'list'>","value":"[{'text': '这', 'predicted_label': 1, 'predicted_class': 1, 'confidence': 0.900057315826416}]","is_numpy":false,"converted":[{"text":"这","predicted_label":1,"predicted_class":1,"confidence":0.900057315826416}],"converted_type":"<class 'list'>"}

#### 2. 单个预测
curl -X POST "http://localhost:8000/predict/这个产品很好用"
INFO:     127.0.0.1:56169 - "POST /predict/%E8%BF%99%E4%B8%AA%E4%BA%A7%E5%93%81%E5%BE%88%E5%A5%BD%E7%94%A8 HTTP/1.1" 200 OK
{"text":"这个产品很好用","predicted_label":"1","predicted_class":1,"confidence":0.9001}

#### 3. 批量预测(还在调试阶段)
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"texts": ["这个很好", "那个不好"]}'

INFO:     127.0.0.1:58910 - "POST /predict HTTP/1.1" 200 OK
INFO:     127.0.0.1:51000 - "POST /predict HTTP/1.1" 200 OK
[{"text":"这个很好","predicted_label":"1","predicted_class":1,"confidence":0.7766},{"text":"那个不好","predicted_label":"0","predicted_class":0,"confidence":0.7985}]

### AB测试结果

ab -n 100 -c 1 -p ./assets/post_data_multi.json -T "application/json" http://127.0.0.1:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient)...predict batch: ['这个产品很好用', '好吃得没话说', '这个产品简直没法用', '还是等下一个新产品吧', '不太好吃']
..done

Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /predict
Document Length:        478 bytes

Concurrency Level:      1
Time taken for tests:   6.173 seconds
Complete requests:      100
Failed requests:        0
Total transferred:      62300 bytes
Total body sent:        29000
HTML transferred:       47800 bytes
Requests per second:    16.20 [#/sec] (mean)
Time per request:       61.726 [ms] (mean)
Time per request:       61.726 [ms] (mean, across all concurrent requests)
Transfer rate:          9.86 [Kbytes/sec] received
                        4.59 kb/s sent
                        14.44 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       0
Processing:    57   62   6.2     60     109
Waiting:       57   61   6.2     60     109
Total:         58   62   6.2     60     109

Percentage of the requests served within a certain time (ms)
  50%     60
  66%     61
  75%     62
  80%     62
  90%     64
  95%     73
  98%     82
  99%    109
 100%    109 (longest request)

$ab -n 100 -c 5 -p ./assets/post_data_multi.json -T "application/json" http://127.0.0.1:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient)...INFO:     127.0.0.1:52141 - "POST /predict HTTP/1.0" 200 OK
..done

Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /predict
Document Length:        478 bytes

Concurrency Level:      5
Time taken for tests:   6.702 seconds
Complete requests:      100
Failed requests:        0
Total transferred:      62300 bytes
Total body sent:        29000
HTML transferred:       47800 bytes
Requests per second:    14.92 [#/sec] (mean)
Time per request:       335.118 [ms] (mean)
Time per request:       67.024 [ms] (mean, across all concurrent requests)
Transfer rate:          9.08 [Kbytes/sec] received
                        4.23 kb/s sent
                        13.30 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.1      0       0
Processing:    85  325  61.7    320     463
Waiting:       61  235  88.3    252     434
Total:         85  326  61.7    320     463

Percentage of the requests served within a certain time (ms)
  50%    320
  66%    336
  75%    345
  80%    348
  90%    412
  95%    434
  98%    463
  99%    463
 100%    463 (longest request)

(pytorch_d2l) 192:bert_tune_proj wenyuc$ ab -n 100 -c 10 -p ./assets/post_data.json -T "application/json" http://127.0.0.1:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient)...INFO:     127.0.0.1:52306 - "POST /predict HTTP/1.0" 200 OK
......
..done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /predict
Document Length:        478 bytes

Concurrency Level:      10
Time taken for tests:   6.481 seconds
Complete requests:      100
Failed requests:        0
Total transferred:      62300 bytes
Total body sent:        29000
HTML transferred:       47800 bytes
Requests per second:    15.43 [#/sec] (mean)
Time per request:       648.147 [ms] (mean)
Time per request:       64.815 [ms] (mean, across all concurrent requests)
Transfer rate:          9.39 [Kbytes/sec] received
                        4.37 kb/s sent
                        13.76 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.1      0       0
Processing:    86  634  95.2    623     808
Waiting:       61  360 186.9    369     778
Total:         86  635  95.2    624     808

Percentage of the requests served within a certain time (ms)
  50%    624
  66%    628
  75%    643
  80%    682
  90%    779
  95%    808
  98%    808
  99%    808
 100%    808 (longest request)

### 存在的问题
1. 模型增大训练样本，保存最高精度的模型还要调整
2. API接口继续完善
