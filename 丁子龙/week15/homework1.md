# 【08-financial-report-generator】中包含的方案

## 方案-天池三轮车
### 数据源
#### 政务 
国家统计局
【中国国家统计局数据】
1. 获取分类

https://data.stats.gov.cn/easyquery.htm

2. 获取数据

https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=hgyd&rowcode=zb&colcode=sj&wds=%5B%5D&dfwds=%5B%7B%22wdcode%22%3A%22zb%22%2C%22valuecode%22%3A%22A0801%22%7D%5D&k1=1752137782720&h=1

人民银行
'http://www.pbc.gov.cn'

政策库
国务院政策文件搜索'https://sousuo.www.gov.cn/search-gov/data?t=zhengcelibrary_gw_bm_gb&q={key}&timetype=timezd&mintime={start}&maxtime={end}&sort=pubtime&sortType=1&searchfield=title&pcodeJiguan=&childtype=&subchildtype=&tsbq=&pubtimeyear=&puborg=&pcodeYear=&pcodeNum=&filetype=&p=1&n=5&inpro=&bmfl=&dup=&orpro=&bmpubyear='

政府网
人民政府网信息搜索'https://sousuoht.www.gov.cn/athena/forward/2B22E8E39E850E17F95A016A74FCB6B673336FA8B6FEC0E2955907EF9AEE06BE'


#### 交易所
上交所
获取恒生指数
1. 获取目录，seriesName指数名称，seriesCode指数编号
https://www.hsi.com.hk/data/schi/index-series/directory.json
2. 获取指数ID，indexSeriesList[0].indexList[0].indexCode
https://www.hsi.com.hk/data/schi/rt/index-series/hsi/performance.do?7602
3. 获取指数数据，
https://www.hsi.com.hk/data/schi/indexes/{指数ID}/chart.json

深交所

港交所

期货交易所

#### 权威平台
中国新闻网

东方财富：Akshare外汇数据采集，Akshare期货数据数据采集，Akshare宏观经济数据采集

同花顺

百度股市通：http://finance.pae.baidu.com/selfselect/sug?wd={0}&skip_login=1&finClientType=pc

金十数据：https://finnews.cubenlp.com

新浪财经：https://sou.chinanews.com/search/news

雪球股票：https://xueqiu.com/hq

## 方案-好像成为人类
### 数据源
#### 搜索引擎
 搜狗搜素
 bing
 duckduckgo


## 方案-队伍名字不能为空
### 数据源
#### 搜索引擎
 搜狗搜素
 bing
 duckduckgo
 谷歌搜索
baidusearch

## 方案-乘风破浪的弟弟
### 数据源
Wind
Akshare
沪深300上证指数
websearch

## 方案-清水河小分队
### 数据源
AKShare
东方财富
同花顺
官方网站
金融数据
实时信息
实时信息
政策数据
香港交易所
上海交易所
世界银行
深圳交易所
Google搜索
websearch



