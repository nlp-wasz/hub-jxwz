# 作业1:

阅读【08-financial-report-generator】中包含的方案，总结下其中使用了哪些外部数据源？

## 天池三轮车方案

1. **网络搜索数据源**：
    -
    百度搜索 ([baidu_search.py](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week15/08-financial-report-generator/方案-天池三轮车/docker_image/docker_image/app/tools/search/baidu_search.py))
    -
    必应搜索 ([bing_search.py](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week15/08-financial-report-generator/方案-天池三轮车/docker_image/docker_image/app/tools/search/bing_search.py))
    -
    DuckDuckGo搜索 ([duckduckgo_search.py](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week15/08-financial-report-generator/方案-天池三轮车/docker_image/docker_image/app/src/tool/search/duckduckgo_search.py))
    -
    谷歌搜索 ([google_search.py](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week15/08-financial-report-generator/方案-天池三轮车/docker_image/docker_image/app/src/tool/search/google_search.py))

2. **PDF文档处理**：
    - 使用`mineru`工具进行PDF解析
    - 使用`PyMuPDF`处理PDF文件

3. **财经数据源**：
    - 通过`akshare`库获取财经数据
    - 使用`stock_zh_a_spot_em()`获取东方财富实时数据

## 好想成为人类方案

1. **宏观数据源**：
    -
    中国国家统计局数据 ([data_gjtjj.py](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week15/08-financial-report-generator/方案-队伍名字不能为空/docker_image/app/mcps/spider/data_gjtjj.py))
        - 包含月度数据（价格指数、工业、能源、固定资产投资等）
        - 季度数据（国民经济核算、农业、工业、建筑业等）

2. **证券交易所数据**：
    -
    上海证券交易所公开报告 ([report_sh.py](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week15/08-financial-report-generator/方案-队伍名字不能为空/docker_image/app/mcps/spider/report_sh.py))
    - 通过AkShare获取股票相关数据

3. **网络资源**：
    - 使用`selenium`和`requests`爬取网页数据
    - PDF文件下载和解析功能

## 队伍名字不能为空方案

1. **财经数据**：
    - AkShare库提供各类财经数据
    - 东方财富实时数据
    - 雪球网数据 (`pysnowball`)

2. **宏观统计数据**：
    - 国家统计局宏观经济数据
    - 月度和季度统计数据

3. **网络资源访问**：
    -
    HTTP请求工具 ([http_utils.py](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week15/08-financial-report-generator/方案-队伍名字不能为空/docker_image/app/mcps/common/http_utils.py))
    - 浏览器自动化工具 (Selenium)

# 作业2:

本地尝试部署下mineru，进行pdf文档解析；如果本地无法部署，则使用云端的环境api； 使用mineru 工具对一个pdf/word进行解析，然后对这个文档进行rag问答；

详见mineru_rag.py

# 作业3:

将 09_DeepResearch.py 改为不同章节同时生成，并且加入 方式/react 的机制，大模型判断这个这个章节的生成效果，有反馈建议，逐步生成。

详见09_DeepResearch.py
