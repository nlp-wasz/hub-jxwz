import os
import asyncio
from typing import Dict, List, Optional
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

# 设置OpenAI API
# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-e512f31a96454eaf871605cc0d440220"   #my test key
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents import Agent, Runner, handoff, set_tracing_disabled
set_tracing_disabled(True)

# 模拟股票数据库类
class StockDatabase:
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_database()
        self._populate_sample_data()
    
    def _init_database(self):
        """初始化股票数据库表"""
        cursor = self.conn.cursor()
        
        # 创建股票基本信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                symbol TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                industry TEXT,
                market_cap REAL
            )
        ''')
        
        # 创建股票价格表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                FOREIGN KEY (symbol) REFERENCES stocks (symbol)
            )
        ''')
        
        self.conn.commit()
    
    def _populate_sample_data(self):
        """填充示例股票数据"""
        cursor = self.conn.cursor()
        
        # 检查是否已有数据
        cursor.execute("SELECT COUNT(*) FROM stocks")
        if cursor.fetchone()[0] > 0:
            return
        
        # 添加示例股票
        sample_stocks = [
            ('AAPL', 'Apple Inc.', 'Technology', 2800000000000),
            ('GOOGL', 'Alphabet Inc.', 'Technology', 1800000000000),
            ('MSFT', 'Microsoft Corporation', 'Technology', 2200000000000),
            ('TSLA', 'Tesla Inc.', 'Automotive', 600000000000),
            ('AMZN', 'Amazon.com Inc.', 'E-commerce', 1500000000000),
            ('BABA', 'Alibaba Group', 'E-commerce', 250000000000),
            ('00700', 'Tencent Holdings', 'Technology', 400000000000),
            ('601318', 'Ping An Insurance', 'Finance', 150000000000),
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO stocks VALUES (?, ?, ?, ?)",
            sample_stocks
        )
        
        # 生成最近30天的价格数据
        end_date = datetime.now()
        for symbol, _, _, _ in sample_stocks:
            base_price = random.uniform(100, 500)
            for i in range(30):
                date = (end_date - timedelta(days=29-i)).strftime('%Y-%m-%d')
                change = random.uniform(-0.05, 0.05)
                close_price = base_price * (1 + change)
                open_price = close_price * random.uniform(0.98, 1.02)
                high_price = max(open_price, close_price) * random.uniform(1.01, 1.05)
                low_price = min(open_price, close_price) * random.uniform(0.95, 0.99)
                volume = random.randint(1000000, 50000000)
                
                cursor.execute('''
                    INSERT INTO stock_prices 
                    (symbol, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, date, open_price, high_price, low_price, close_price, volume))
                
                base_price = close_price
        
        self.conn.commit()
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """获取股票基本信息"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT symbol, name, industry, market_cap FROM stocks WHERE symbol = ?", 
            (symbol,)
        )
        result = cursor.fetchone()
        if result:
            return {
                'symbol': result[0],
                'name': result[1],
                'industry': result[2],
                'market_cap': result[3]
            }
        return None
    
    def get_stock_price(self, symbol: str, days: int = 30) -> List[Dict]:
        """获取股票价格历史"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT date, open_price, high_price, low_price, close_price, volume
            FROM stock_prices 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT ?
        ''', (symbol, days))
        
        prices = []
        for row in cursor.fetchall():
            prices.append({
                'date': row[0],
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5]
            })
        
        return prices
    
    def search_stocks(self, keyword: str) -> List[Dict]:
        """搜索股票"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT symbol, name, industry, market_cap 
            FROM stocks 
            WHERE symbol LIKE ? OR name LIKE ? OR industry LIKE ?
        ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))
        
        stocks = []
        for row in cursor.fetchall():
            stocks.append({
                'symbol': row[0],
                'name': row[1],
                'industry': row[2],
                'market_cap': row[3]
            })
        
        return stocks
    
    def get_industry_stocks(self, industry: str) -> List[Dict]:
        """获取特定行业的股票"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT symbol, name, industry, market_cap 
            FROM stocks 
            WHERE industry LIKE ?
        ''', (f'%{industry}%',))
        
        stocks = []
        for row in cursor.fetchall():
            stocks.append({
                'symbol': row[0],
                'name': row[1],
                'industry': row[2],
                'market_cap': row[3]
            })
        
        return stocks

# 创建股票数据库实例
stock_db = StockDatabase()

# 定义股票工具函数
def get_stock_info(symbol: str) -> str:
    """获取股票基本信息"""
    try:
        info = stock_db.get_stock_info(symbol.upper())
        if info:
            return f"""
股票 {symbol} 信息：
- 公司名称：{info['name']}
- 行业：{info['industry']}
- 市值：${info['market_cap']:,.0f}
"""
        else:
            return f"未找到股票 {symbol} 的信息"
    except Exception as e:
        return f"查询股票信息时出错：{str(e)}"

def get_stock_price_history(symbol: str, days: int = 7) -> str:
    """获取股票价格历史"""
    try:
        prices = stock_db.get_stock_price(symbol.upper(), days)
        if not prices:
            return f"未找到股票 {symbol} 的价格数据"
        
        latest = prices[0]
        price_change = latest['close'] - prices[-1]['close']
        change_percent = (price_change / prices[-1]['close']) * 100
        
        result = f"""
股票 {symbol} 最近{days}天价格：
最新价格：${latest['close']:.2f}
涨跌幅：{price_change:+.2f} ({change_percent:+.2f}%)
日期范围：{prices[-1]['date']} 到 {latest['date']}

最近5个交易日：
"""
        for price in prices[:5]:
            result += f"- {price['date']}: ${price['close']:.2f} (成交量: {price['volume']:,})\n"
        
        return result
    except Exception as e:
        return f"查询股票价格时出错：{str(e)}"

def search_stocks(keyword: str) -> str:
    """搜索股票"""
    try:
        stocks = stock_db.search_stocks(keyword)
        if not stocks:
            return f"未找到包含 '{keyword}' 的股票"
        
        result = f"找到 {len(stocks)} 只相关股票：\n"
        for stock in stocks:
            result += f"- {stock['symbol']}: {stock['name']} ({stock['industry']})\n"
        
        return result
    except Exception as e:
        return f"搜索股票时出错：{str(e)}"

def get_industry_overview(industry: str) -> str:
    """获取行业概览"""
    try:
        stocks = stock_db.get_industry_stocks(industry)
        if not stocks:
            return f"未找到 {industry} 行业的股票"
        
        result = f"{industry} 行业股票概览：\n"
        total_market_cap = sum(stock['market_cap'] for stock in stocks)
        
        for stock in stocks:
            market_share = (stock['market_cap'] / total_market_cap) * 100
            result += f"- {stock['symbol']} ({stock['name']}): ${stock['market_cap']:,.0f} ({market_share:.1f}%)\n"
        
        result += f"\n行业总市值：${total_market_cap:,.0f}"
        return result
    except Exception as e:
        return f"获取行业概览时出错：{str(e)}"

# 创建股票Agent
stock_agent = Agent(
    name="StockExpert",
    instructions="""你是一个专业的股票分析师助手。你专门处理股票相关的查询，包括：
1. 查询股票基本信息
2. 获取股票价格历史
3. 搜索特定股票
4. 分析行业趋势

你可以访问实时的股票数据库来获取准确信息。请以专业、准确的方式回答用户问题。
如果用户的问题与股票无关，请礼貌地建议他们与闲聊助手交流。

可用工具：
- get_stock_info: 获取股票基本信息
- get_stock_price_history: 获取股票价格历史
- search_stocks: 搜索相关股票
- get_industry_overview: 获取行业概览

请根据问题选择合适的工具来获取最新数据。""",
    tools=[
        get_stock_info,
        get_stock_price_history,
        search_stocks,
        get_industry_overview
    ],
    handoffs=[],  # 可以切换到其他Agent
)

# 创建闲聊Agent
chat_agent = Agent(
    name="ChatAssistant",
    instructions="""你是一个友好、热情的闲聊助手。你擅长：
1. 日常对话和闲聊
2. 回答一般知识问题
3. 提供生活建议
4. 进行轻松的对话

如果用户询问股票、投资、金融相关的问题，请礼貌地将对话转交给股票专家。
保持对话自然、友好、有趣。""",
    tools=[],  # 闲聊Agent不需要工具
    handoffs=[stock_agent],  # 可以切换到股票Agent
)

# 主协调Agent
main_agent = Agent(
    name="MainCoordinator",
    instructions="""你是对话协调员，负责将用户请求路由到合适的专家：
- 如果用户询问股票、投资、价格、市场等金融相关话题，转交给StockExpert
- 如果是一般对话、闲聊、日常问题，转交给ChatAssistant
- 如果用户明确说要切换助手，按照要求处理

请快速准确地判断用户意图并进行路由。""",
    handoffs=[chat_agent, stock_agent],
)

# 对话运行函数
async def run_conversation():
    print("=== 多Agent对话系统 ===")
    print("支持话题：股票查询、日常闲聊")
    print("输入 'quit' 退出\n")
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            # 运行主协调Agent
            result = await Runner.run(main_agent, input=user_input)
            
            print(f"\n🤖 助手: {result.final_output}")
            
            # 显示实际处理Agent的信息
            if hasattr(result, 'final_agent'):
                print(f"   [由 {result.final_agent.name} 处理]")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误：{e}")

# 测试函数
async def test_agents():
    """测试各个Agent的功能"""
    test_cases = [
        "你好，今天天气怎么样？",  # 应该由闲聊Agent处理
        "苹果公司的股票代码是什么？",  # 应该由股票Agent处理
        "告诉我AAPL的股价",  # 应该由股票Agent处理
        "科技行业有哪些主要公司？",  # 应该由股票Agent处理
        "讲个笑话吧",  # 应该由闲聊Agent处理
    ]
    
    print("=== Agent测试 ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case}")
        result = await Runner.run(main_agent, input=test_case)
        print(f"回复: {result.final_output}")
        if hasattr(result, 'final_agent'):
            print(f"处理Agent: {result.final_agent.name}")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_agents())
    
    print("\n" + "="*50)
    
    # 运行交互式对话
    asyncio.run(run_conversation())