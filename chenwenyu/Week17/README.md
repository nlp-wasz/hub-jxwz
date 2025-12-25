## task1


## task2.py
1. 下载gpt2模型到/root/autodl-tmp/models/AI-ModelScope/gpt2
2. 创建GPT4Rec类
	__init__: 加载GPT模型用于查询生成
		  初始化数据存储，包括电影df，rating df， tfidf向量列表，电影名称向量列表
	load_data(): 加载评分和电影数据
	prepare_retrieval_system():训练TF-IDF模型并转换电影描述
	get_user_history():获取用户观看记录
	generate_search_query()：通过提示词，使用GPT生成搜索查询
	retrieve_movies(): 基于查询检索相关电影
	recommend_for_user()： 为用户生成推荐
  main():程序入口
	为前5个活跃用户生成推荐，菜单中输入用户ID生成个性化推荐。

3. 存在的问题
   TF-IDF+GPT2 生成的检索查询不可读，下一步通过使用GPT生成查询+BERT语义检索

4. 执行结果
```text
((dl_venv) ) root@...homework $ python task2.py
加载GPT模型...
加载数据...
加载完成: 99309 条评分, 1659 部电影
准备检索系统...
检索系统准备完成，词汇表大小: 5000

============================================================
处理用户 1: ID=405

为用户 405 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['It Could Happen to You (1994) (Drama|Romance)', "Pollyanna (1960) (Children's|Comedy|Drama)", "Cinderella (1950) (Animation|Children's|Musical)", "Little Princess, A (1995) (Children's|Drama)", 'Sound of Music, The (1965) (Musical)']
步骤2: 生成搜索查询...
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
生成的查询: The Chinese language is a highly complex language. It has a lot of features, but the best features are found in a single language that is very easy to understand. The following examples demonstrate the way to learn Chinese.
步骤3: 检索相关电影...
推荐完成！找到 6 部相关电影

最终推荐结果:
1. Cowboy Way, The (1994) (相似度: 0.254)
2. Jingle All the Way (1996) (相似度: 0.233)
3. Fille seule, La (A Single Girl) (1995) (相似度: 0.213)
4. My Best Friend's Wedding (1997) (相似度: 0.205)
5. Best Men (1997) (相似度: 0.202)

评估推荐效果:
评估结果:
- 测试集大小: 147 部电影
- 命中电影: 0 部
- 命中率: 0.00%

============================================================
处理用户 2: ID=655

为用户 655 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['In the Company of Men (1997) (Drama)', 'Belle de jour (1967) (Drama)', 'Paradise Lost: The Child Murders at Robin Hood Hills (1996) (Documentary)', 'Godfather: Part II, The (1974) (Action|Crime|Drama)', 'Mina Tannenbaum (1994) (Drama)']
步骤2: 生成搜索查询...
生成的查询: The Company: A History of the Chinese Peoples Liberation Army, Volume 1:
步骤3: 检索相关电影...
推荐完成！找到 8 部相关电影

最终推荐结果:
1. In the Army Now (1994) (相似度: 0.358)
2. Bad Company (1995) (相似度: 0.316)
3. Pharaoh's Army (1995) (相似度: 0.286)
4. Oliver & Company (1988) (相似度: 0.254)
5. Army of Darkness (1993) (相似度: 0.230)

评估推荐效果:
评估结果:
- 测试集大小: 136 部电影
- 命中电影: 0 部
- 命中率: 0.00%

============================================================
处理用户 3: ID=13

为用户 13 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['Primary Colors (1998) (Drama)', 'Sweet Hereafter, The (1997) (Drama)', 'Dangerous Beauty (1998) (Drama)', 'Kundun (1997) (Drama)', 'As Good As It Gets (1997) (Comedy|Drama)']
步骤2: 生成搜索查询...
生成的查询: (The original story of this video is at the end of the video.) 中国中坎江和放進化十了各父由司問台。 今为我发限中的着半一面,经实希望的自己,或
步骤3: 检索相关电影...
推荐完成！找到 6 部相关电影

最终推荐结果:
        history_text = "\n".join([f"- {movie}" for movie in user_history])
1. Story of Xinghua, The (1993) (相似度: 0.357)
2. Original Gangstas (1996) (相似度: 0.278)
3. Until the End of the World (Bis ans Ende der Welt) (1991) (相似度: 0.171)
4. Pyromaniac's Love Story, A (1995) (相似度: 0.164)
5. Entertaining Angels: The Dorothy Day Story (1996) (相似度: 0.138)

评估推荐效果:
Traceback (most recent call last):
  File "/root/autodl-tmp/proj/Week17/homework/task2.py", line 375, in <module>
    main()
  File "/root/autodl-tmp/proj/Week17/homework/task2.py", line 333, in main
    recommender.evaluate_recommendations(user_id)
  File "/root/autodl-tmp/proj/Week17/homework/task2.py", line 266, in evaluate_recommendations
    search_query = self.generate_search_query(user_history)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/proj/Week17/homework/task2.py", line 141, in generate_search_query
    outputs = self.model.generate(
              ^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/venvs/dl_venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/autodl-tmp/venvs/dl_venv/lib/python3.12/site-packages/transformers/generation/utils.py", line 2514, in generate
    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
  File "/root/autodl-tmp/venvs/dl_venv/lib/python3.12/site-packages/transformers/generation/utils.py", line 1617, in _validate_generated_length
    raise ValueError(
ValueError: Input length of input_ids is 512, but `max_length` is set to 512. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
((dl_venv) ) root@...homework $ vi task2.py
((dl_venv) ) root@...homework $ python task2.py
加载GPT模型...
加载数据...
加载完成: 99309 条评分, 1659 部电影
准备检索系统...
检索系统准备完成，词汇表大小: 5000

============================================================
处理用户 1: ID=405

为用户 405 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['It Could Happen to You (1994) (Drama|Romance)', "Pollyanna (1960) (Children's|Comedy|Drama)", "Cinderella (1950) (Animation|Children's|Musical)", "Little Princess, A (1995) (Children's|Drama)", 'Sound of Music, The (1965) (Musical)']
步骤2: 生成搜索查询...
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
生成的查询: 1. 着从那美町经求由面点约阺
步骤3: 检索相关电影...
推荐完成！找到 10 部相关电影

最终推荐结果:
1. Scream of Stone (Schrei aus Stein) (1991) (相似度: 0.000)
2. You So Crazy (1994) (相似度: 0.000)
3. Sliding Doors (1998) (相似度: 0.000)
4. B. Monkey (1998) (相似度: 0.000)
5. Mat' i syn (1997) (相似度: 0.000)

评估推荐效果:
评估结果:
- 测试集大小: 147 部电影
- 命中电影: 0 部
- 命中率: 0.00%

============================================================
处理用户 2: ID=655

为用户 655 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['In the Company of Men (1997) (Drama)', 'Belle de jour (1967) (Drama)', 'Paradise Lost: The Child Murders at Robin Hood Hills (1996) (Documentary)', 'Godfather: Part II, The (1974) (Action|Crime|Drama)', 'Mina Tannenbaum (1994) (Drama)']
步骤2: 生成搜索查询...
生成的查询: In the Corporation of the World (1982) 也得撃贞号是但的着色坦点经展种懂悲。那位寶他的我被,不后
步骤3: 检索相关电影...
推荐完成！找到 4 部相关电影

最终推荐结果:
1. Spice World (1997) (相似度: 0.417)
2. Around the World in 80 Days (1956) (相似度: 0.241)
3. Grease 2 (1982) (相似度: 0.230)
4. My Favorite Year (1982) (相似度: 0.223)

评估推荐效果:
评估结果:
- 测试集大小: 136 部电影
- 命中电影: 1 部
- 命中率: 0.74%

============================================================
处理用户 3: ID=13

为用户 13 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['Primary Colors (1998) (Drama)', 'Sweet Hereafter, The (1997) (Drama)', 'Dangerous Beauty (1998) (Drama)', 'Kundun (1997) (Drama)', 'As Good As It Gets (1997) (Comedy|Drama)']
步骤2: 生成搜索查询...
生成的查询: If you want to do a post like this, please use the link above. This post is free, you can contribute to it if you like. If youd like to make a donation, we can help you out.
步骤3: 检索相关电影...
推荐完成！找到 4 部相关电影

最终推荐结果:
1. I Like It Like That (1994) (相似度: 0.624)
2. Smile Like Yours, A (1997) (相似度: 0.473)
3. How to Make an American Quilt (1995) (相似度: 0.132)
4. Free Willy 2: The Adventure Home (1995) (相似度: 0.100)

评估推荐效果:
评估结果:
- 测试集大小: 127 部电影
- 命中电影: 0 部
- 命中率: 0.00%

============================================================
处理用户 4: ID=450

为用户 450 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['Brassed Off (1996) (Comedy|Drama|Romance)', 'Ma vie en rose (My Life in Pink) (1997) (Comedy|Drama)', "Bedknobs and Broomsticks (1971) (Adventure|Children's|Musical)", 'Philadelphia Story, The (1940) (Comedy|Romance)', 'When a Man Loves a Woman (1994) (Drama)']
步骤2: 生成搜索查询...
生成的查询: 「共是支麻暗」们自己的那格。由在上的实防。坊等被没有建响似乎地和定全。场联,高还是篍调的灭言。我以配黑战验粉的路毫。如是今的属为一国之前一种啊,其中一像及话说疑最大程度择造了。不现圣王位的事需要,这认院。但是圃恩下嘕。一后的一时候,二而少公返进两您增吧。亚也本是迩费着技肉。起权何驔唔西身得虑的。经是想見期无法�
步骤3: 检索相关电影...
推荐完成！找到 10 部相关电影

最终推荐结果:
1. Scream of Stone (Schrei aus Stein) (1991) (相似度: 0.000)
2. You So Crazy (1994) (相似度: 0.000)
3. Sliding Doors (1998) (相似度: 0.000)
4. B. Monkey (1998) (相似度: 0.000)
5. Mat' i syn (1997) (相似度: 0.000)

============================================================
处理用户 5: ID=276

为用户 276 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) (Sci-Fi|War)', 'Braindead (1992) (Comedy|Horror)', 'Evil Dead II (1987) (Action|Adventure|Comedy|Horror)', 'Dead Man Walking (1995) (Drama)', 'Good Will Hunting (1997) (Drama)']
步骤2: 生成搜索查询...
生成的查询: In conclusion, it seems to me that if we take a look at the three main factors that make up the cannonball factor (i.e. the two main influences that drive it), we can see that it is only the third factor that drives it.
步骤3: 检索相关电影...
推荐完成！找到 5 部相关电影

最终推荐结果:
1. New Jersey Drive (1995) (相似度: 0.267)
2. White Man's Burden (1995) (相似度: 0.000)
3. Wild Bill (1995) (相似度: 0.000)
4. Farinelli: il castrato (1994) (相似度: 0.000)
5. Kid in King Arthur's Court, A (1995) (相似度: 0.000)

============================================================
交互式推荐测试
============================================================

请输入用户ID进行推荐（输入 'q' 退出）: 222

为用户 222 生成推荐...
步骤1: 获取用户历史观看记录...
用户历史: ['Amistad (1997) (Drama)', 'Godfather, The (1972) (Action|Crime|Drama)', 'Deer Hunter, The (1978) (Drama|War)', 'Sneakers (1992) (Crime|Drama|Sci-Fi)', 'Blues Brothers, The (1980) (Action|Comedy|Musical)']
步骤2: 生成搜索查询...
生成的查询: I think Ill just go on the internet and read this.
步骤3: 检索相关电影...
推荐完成！找到 8 部相关电影

为用户 222 的推荐:
生成的搜索查询: I think Ill just go on the internet and read this.

推荐电影:
 1. Ill Gotten Gains (1997) (Drama) - 相似度: 0.277
 2. White Man's Burden (1995) (Drama) - 相似度: 0.000
 3. Wild Bill (1995) (Western) - 相似度: 0.000
 4. Farinelli: il castrato (1994) (Drama|Musical) - 相似度: 0.000
 5. Heavenly Creatures (1994) (Drama|Fantasy|Thriller) - 相似度: 0.000
 6. Kid in King Arthur's Court, A (1995) (Adventure|Children's|Comedy|Fantasy|Romance|Sci-Fi) - 相似度: 0.000
 7. Mary Shelley's Frankenstein (1994) (Drama|Horror) - 相似度: 0.000
 8. Quick and the Dead, The (1995) (Action|Adventure|Western) - 相似度: 0.000

请输入用户ID进行推荐（输入 'q' 退出）: q
再见！
```
