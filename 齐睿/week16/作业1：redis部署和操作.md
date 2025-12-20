# 部署
```
C:\Users\QR>redis-server
1518:C 18 Dec 2025 11:27:57.644 * oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
1518:C 18 Dec 2025 11:27:57.651 * Redis version=8.2.1, bits=64, commit=00000000, modified=0, pid=1518, just started
1518:C 18 Dec 2025 11:27:57.652 # Warning: no config file specified, using the default config. In order to specify a config file use redis-server /path/to/redis.conf
1518:M 18 Dec 2025 11:27:57.668 # You requested maxclients of 10000 requiring at least 10032 max file descriptors.
1518:M 18 Dec 2025 11:27:57.681 # Server can't set maximum open files to 10032 because of OS error: Operation not permitted.
1518:M 18 Dec 2025 11:27:57.696 # Current maximum open files is 3200. maxclients has been reduced to 3168 to compensate for low ulimit. If you need higher maxclients increase 'ulimit -n'.
1518:M 18 Dec 2025 11:27:57.698 * monotonic clock: POSIX clock_gettime
                _._
           _.-``__ ''-._
      _.-``    `.  `_.  ''-._           Redis Open Source
  .-`` .-```.  ```\/    _.,_ ''-._      8.2.1 (00000000/0) 64 bit
 (    '      ,       .-`  | `,    )     Running in standalone mode
 |`-._`-...-` __...-.``-._|'` _.-'|     Port: 6379
 |    `-._   `._    /     _.-'    |     PID: 1518
  `-._    `-._  `-./  _.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |           https://redis.io
  `-._    `-._`-.__.-'_.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |
  `-._    `-._`-.__.-'_.-'    _.-'
      `-._    `-.__.-'    _.-'
          `-._        _.-'
              `-.__.-'

 Redis for Windows Project Home
 https://github.com/redis-windows/redis-windows

1518:M 18 Dec 2025 11:27:57.720 * Server initialized
1518:M 18 Dec 2025 11:27:57.721 * Loading RDB produced by version 8.2.1
1518:M 18 Dec 2025 11:27:57.722 * RDB age 4227399 seconds
1518:M 18 Dec 2025 11:27:57.722 * RDB memory usage when created 0.53 Mb
1518:M 18 Dec 2025 11:27:57.722 * Done loading RDB, keys loaded: 1, keys expired: 0.
1518:M 18 Dec 2025 11:27:57.722 * DB loaded from disk: 0.002 seconds
1518:M 18 Dec 2025 11:27:57.722 * Ready to accept connections tcp
```

# 简单操作
```cmd
PS C:\Users\QR> redis-cli
127.0.0.1:6379> SELECT 0
OK
127.0.0.1:6379> DBSIZE
(integer) 1
127.0.0.1:6379> SET name "张三"
OK
127.0.0.1:6379> GET name
"\xe5\xbc\xa0\xe4\xb8\x89"
127.0.0.1:6379> LPUSH mylist "item1"
(integer) 1
127.0.0.1:6379> LRANGE mylist 0 -1
1) "item1"
127.0.0.1:6379> HSET user:1000 name "李四" age 30
(integer) 2
127.0.0.1:6379> HGETALL user:1000
2) "name"
3) "\xe6\x9d\x8e\xe5\x9b\x9b"
4) "age"
5) "30"
127.0.0.1:6379> KEYS *
6) "name"
7) "user:1000"
8) "mylist"
9) "myKey"
127.0.0.1:6379> SCAN 0 COUNT 100
10) "0"
11) 1) "name"
   12) "user:1000"
   13) "mylist"
   14) "myKey"
127.0.0.1:6379>
```