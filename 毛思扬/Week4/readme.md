# 第四周
1、 单独给一个外卖评价数据集，需要大家使用bert模型做微调，使用fastapi部署为服务。

​		详见01-intent-classify.py

2、 部署完服务后，使用ab测试一下，1/5/10 并发下总耗时；
​		
​		bert
​		![ab_results_bert.png](01-intent-classify/test/ab_results_bert.png)
​		
​		gpt
​		![ab_results_gpt.png](01-intent-classify/test/ab_results_gpt.png)
​		
​		regex
​		![ab_results_regex.png](01-intent-classify/test/ab_results_regex.png)
​		
​		tfidf
​		![ab_results_tfidf.png](01-intent-classify/test/ab_results_tfidf.png)
