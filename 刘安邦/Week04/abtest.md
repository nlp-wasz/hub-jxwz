ab -n 100 -c 1 -p data.json -T 'application/json' -H 'accept: application/json' 'http://127.0.0.1:8000/v1/text-cls/bert'
This is ApacheBench, Version 2.3 <$Revision: 1923142 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient).....done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /v1/text-cls/bert
Document Length:        131 bytes

Concurrency Level:      1
Time taken for tests:   2.400 seconds
Complete requests:      100
Failed requests:        15
   (Connect: 0, Receive: 0, Length: 15, Exceptions: 0)
Total transferred:      25685 bytes
Total body sent:        24100
HTML transferred:       13085 bytes
Requests per second:    41.66 [#/sec] (mean)
Time per request:       24.003 [ms] (mean)
Time per request:       24.003 [ms] (mean, across all concurrent requests)
Transfer rate:          10.45 [Kbytes/sec] received
                        9.80 kb/s sent
                        20.25 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.3      0       1
Processing:    20   24   2.4     24      40
Waiting:       20   24   2.4     23      40
Total:         20   24   2.4     24      40

Percentage of the requests served within a certain time (ms)
  50%     24
  66%     24
  75%     25
  80%     25
  90%     27
  95%     28
  98%     30
  99%     40
 100%     40 (longest request)

ab -n 100 -c 5 -p data.json -T 'application/json' -H 'accept: application/json' 'http://127.0.0.1:8000/v1/text-cls/bert'
This is ApacheBench, Version 2.3 <$Revision: 1923142 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient).....done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /v1/text-cls/bert
Document Length:        131 bytes

Concurrency Level:      5
Time taken for tests:   1.260 seconds
Complete requests:      100
Failed requests:        10
   (Connect: 0, Receive: 0, Length: 10, Exceptions: 0)
Total transferred:      25690 bytes
Total body sent:        24100
HTML transferred:       13090 bytes
Requests per second:    79.35 [#/sec] (mean)
Time per request:       63.013 [ms] (mean)
Time per request:       12.603 [ms] (mean, across all concurrent requests)
Transfer rate:          19.91 [Kbytes/sec] received
                        18.67 kb/s sent
                        38.58 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.4      0       2
Processing:    39   61   5.2     60      79
Waiting:       39   60   5.2     60      78
Total:         40   61   5.2     61      79

Percentage of the requests served within a certain time (ms)
  50%     61
  66%     63
  75%     63
  80%     64
  90%     66
  95%     67
  98%     78
  99%     79
 100%     79 (longest request)

ab -n 100 -c 10 -p data.json -T 'application/json' -H 'accept: application/json' 'http://127.0.0.1:8000/v1/text-cls/bert'
This is ApacheBench, Version 2.3 <$Revision: 1923142 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient).....done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /v1/text-cls/bert
Document Length:        131 bytes

Concurrency Level:      10
Time taken for tests:   1.270 seconds
Complete requests:      100
Failed requests:        12
   (Connect: 0, Receive: 0, Length: 12, Exceptions: 0)
Total transferred:      25688 bytes
Total body sent:        24100
HTML transferred:       13088 bytes
Requests per second:    78.74 [#/sec] (mean)
Time per request:       127.006 [ms] (mean)
Time per request:       12.701 [ms] (mean, across all concurrent requests)
Transfer rate:          19.75 [Kbytes/sec] received
                        18.53 kb/s sent
                        38.28 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.5      0       2
Processing:    42  120  13.8    122     146
Waiting:       42  119  13.7    121     144
Waiting:       42  119  13.7    121     144
Total:         43  121  13.8    122     146

Percentage of the requests served within a certain time (ms)
  50%    122
  66%    126
  75%    129
  80%    130
  90%    132
  95%    137
  98%    146
  99%    146
 100%    146 (longest request)
