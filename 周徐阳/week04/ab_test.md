# 使用ab测试一下，1/5/10 总耗时
```cmd
ab -n 100 -c 1 -T "application/json" -p data.json http://localhost:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking localhost (be patient).....done


Server Software:        
Server Hostname:        localhost
Server Port:            8000

Document Path:          /predict
Document Length:        9 bytes

Concurrency Level:      1
Time taken for tests:   0.074 seconds
Complete requests:      100
Failed requests:        0
Non-2xx responses:      100
Total transferred:      13200 bytes
Total body sent:        22100
HTML transferred:       900 bytes
Requests per second:    1349.87 [#/sec] (mean)
Time per request:       0.741 [ms] (mean)
Time per request:       0.741 [ms] (mean, across all concurrent requests)
Transfer rate:          174.01 [Kbytes/sec] received
                        291.33 kb/s sent
                        465.34 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       0
Processing:     0    1   3.1      0      31
Waiting:        0    1   3.1      0      31
Total:          0    1   3.1      0      31

Percentage of the requests served within a certain time (ms)
  50%      0
  66%      0
  75%      0
  80%      0
  90%      1
  95%      1
  98%      1
  99%     31
 100%     31 (longest request)


ab -n 100 -c 5 -T "application/json" -p data.json http://localhost:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking localhost (be patient).....done


Server Software:        
Server Hostname:        localhost
Server Port:            8000

Document Path:          /predict
Document Length:        9 bytes

Concurrency Level:      5
Time taken for tests:   0.052 seconds
Complete requests:      100
Failed requests:        0
Non-2xx responses:      100
Total transferred:      13200 bytes
Total body sent:        22100
HTML transferred:       900 bytes
Requests per second:    1932.96 [#/sec] (mean)
Time per request:       2.587 [ms] (mean)
Time per request:       0.517 [ms] (mean, across all concurrent requests)
Transfer rate:          249.17 [Kbytes/sec] received
                        417.17 kb/s sent
                        666.34 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.1      0       1
Processing:     0    2   2.2      1      13
Waiting:        0    2   2.2      1      13
Total:          1    2   2.2      1      13

Percentage of the requests served within a certain time (ms)
  50%      1
  66%      2
  75%      2
  80%      2
  90%      5
  95%      8
  98%     11
  99%     13
 100%     13 (longest request)

ab -n 100 -c 10 -T "application/json" -p data.json http://localhost:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking localhost (be patient).....done


Server Software:        
Server Hostname:        localhost
Server Port:            8000

Document Path:          /predict
Document Length:        9 bytes

Concurrency Level:      10
Time taken for tests:   0.046 seconds
Complete requests:      100
Failed requests:        0
Non-2xx responses:      100
Total transferred:      13200 bytes
Total body sent:        22100
HTML transferred:       900 bytes
Requests per second:    2188.57 [#/sec] (mean)
Time per request:       4.569 [ms] (mean)
Time per request:       0.457 [ms] (mean, across all concurrent requests)
Transfer rate:          282.12 [Kbytes/sec] received
                        472.34 kb/s sent
                        754.46 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.3      0       1
Processing:     1    2   2.3      1      23
Waiting:        1    2   2.3      1      22
Total:          1    2   2.3      2      23

Percentage of the requests served within a certain time (ms)
  50%      2
  66%      2
  75%      3
  80%      3
  90%      4
  95%      4
  98%      5
  99%     23
 100%     23 (longest request)

```
