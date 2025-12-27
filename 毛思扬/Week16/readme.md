# 作业1:

本地安装一下redis （或使用 http://onecompiler.com/redis/），学习redis的使用，学会操作list、set。

详见作业1.py

![截屏2025-12-17 17.46.07.png](%E4%BD%9C%E4%B8%9A/%E6%88%AA%E5%B1%8F2025-12-17%2017.46.07.png)

# 作业2:

阅读下 https://github.com/redis/redis-vl-python 的项目和已有的代码，阅读 SemanticCache.py，找出其中存在bug。

- 没有使用连接池：代码中直接创建了redis.Redis实例，而没有使用连接池
- 拼写错误：在call方法中，变量名pormpts应该是prompts
- 数据一致性问题：在store方法中，同时使用了Redis的列表和键值对存储，但没有确保它们的一致性
- 异常处理不完善：在store方法中，捕获了所有异常但只是打印traceback，没有适当的错误恢复或处理
- clear_cache方法健壮性不足：当某些键不存在时可能会出现问题

# 作业3:

基于已有的代码，完成SemanticRouter （快速实现意图识别）的逻辑和框架，可以写伪代码（运行不起来也没问题也可以）。

详见09-llm-smart-cache/llm_cache/SemanticRouter.py