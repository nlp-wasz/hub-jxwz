from elasticsearch import Elasticsearch
import json
from datetime import datetime
import traceback


class ESCRUDTest:
    """Elasticsearch增删改查测试类"""
    ELASTICSEARCH_URL = "http://192.168.1.130:9200"

    def __init__(self, es_host=None):
        """初始化Elasticsearch连接"""
        if es_host is None:
            es_host = self.ELASTICSEARCH_URL
        self.es = Elasticsearch(es_host)
        self.index_name = "my_job_test"
        
        # 检查连接
        if self.es.ping():
            print("✅ 成功连接到 Elasticsearch！")
        else:
            print("❌ 无法连接到 Elasticsearch，请检查服务是否运行。")
            return
        
        # 初始化索引
        self._setup_index()
    
    def _setup_index(self):
        """设置索引和映射"""
        print(f"\n=== 设置索引 {self.index_name} ===")
        
        # 如果索引存在，先删除
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            print(f"🗑️ 已删除现有索引 {self.index_name}")
        
        # 创建新索引
        index_mapping = {
            "mappings": {
                "properties": {
                    "employee_id": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "ik_max_word"},
                    "department": {"type": "keyword"},
                    "position": {"type": "text", "analyzer": "ik_smart"},
                    "salary": {"type": "float"},
                    "age": {"type": "integer"},
                    "skills": {"type": "text", "analyzer": "ik_max_word"},
                    "is_active": {"type": "boolean"},
                    "hire_date": {"type": "date"},
                    "created_at": {"type": "date"}
                }
            }
        }
        
        try:
            self.es.indices.create(index=self.index_name, body=index_mapping)
            print(f"✅ 索引 '{self.index_name}' 创建成功")
        except Exception as e:
            print(f"❌ 创建索引失败: {e}")
    
    def print_search_results(self, response, title="搜索结果"):
        """格式化打印搜索结果"""
        print(f"\n--- {title} ---")
        total = response['hits']['total']['value']
        print(f"找到 {total} 条文档：")
        
        if total > 0:
            for hit in response['hits']['hits']:
                print(f"📄 ID: {hit['_id']}, 得分: {hit['_score']:.2f}")
                print(f"   内容: {json.dumps(hit['_source'], ensure_ascii=False, indent=6)}")
        else:
            print("   无匹配结果")
    
    def create_test_data(self):
        """创建测试数据 (Create)"""
        print("\n=== 创建测试数据 ===")
        
        # 准备测试文档
        test_documents = [
            {
                "employee_id": "EMP001",
                "name": "张三",
                "department": "技术部",
                "position": "高级软件工程师",
                "salary": 15000.0,
                "age": 28,
                "skills": "Python, Java, 机器学习, 数据分析",
                "is_active": True,
                "hire_date": "2022-01-15",
                "created_at": datetime.now().isoformat()
            },
            {
                "employee_id": "EMP002",
                "name": "李四",
                "department": "产品部",
                "position": "产品经理",
                "salary": 18000.0,
                "age": 32,
                "skills": "产品设计, 用户体验, 项目管理",
                "is_active": True,
                "hire_date": "2021-06-20",
                "created_at": datetime.now().isoformat()
            },
            {
                "employee_id": "EMP003",
                "name": "王五",
                "department": "技术部",
                "position": "数据工程师",
                "salary": 12000.0,
                "age": 26,
                "skills": "SQL, Python, Elasticsearch, 大数据处理",
                "is_active": True,
                "hire_date": "2023-03-10",
                "created_at": datetime.now().isoformat()
            },
            {
                "employee_id": "EMP004",
                "name": "赵六",
                "department": "人事部",
                "position": "人事专员",
                "salary": 8000.0,
                "age": 24,
                "skills": "招聘, 培训, 员工关系管理",
                "is_active": False,
                "hire_date": "2023-08-01",
                "created_at": datetime.now().isoformat()
            },
            {
                "employee_id": "EMP005",
                "name": "孙七",
                "department": "技术部",
                "position": "前端开发工程师",
                "salary": 11000.0,
                "age": 25,
                "skills": "JavaScript, React, Vue.js, HTML, CSS",
                "is_active": True,
                "hire_date": "2023-05-15",
                "created_at": datetime.now().isoformat()
            }
        ]
        
        try:
            # 批量插入文档
            for doc in test_documents:
                response = self.es.index(
                    index=self.index_name,
                    id=doc["employee_id"],
                    document=doc
                )
                print(f"✅ 文档 {doc['employee_id']} ({doc['name']}) 已插入")
            
            # 刷新索引确保数据可被搜索
            self.es.indices.refresh(index=self.index_name)
            print("✅ 所有测试数据创建完成")
            
        except Exception as e:
            print(f"❌ 创建数据失败: {e}")
    
    def read_test_data(self):
        """读取测试数据 (Read)"""
        print("\n=== 读取测试数据 ===")
        
        try:
            # 1. 查询所有文档
            print("\n--- 查询所有员工 ---")
            all_docs = self.es.search(
                index=self.index_name,
                body={
                    "query": {"match_all": {}},
                    "size": 10
                }
            )
            self.print_search_results(all_docs, "所有员工")
            
            # 2. 精确查询
            print("\n--- 精确查询：技术部员工 ---")
            dept_query = self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "term": {"department": "技术部"}
                    }
                }
            )
            self.print_search_results(dept_query, "技术部员工")
            
            # 3. 全文检索
            print("\n--- 全文检索：技能包含'Python'的员工 ---")
            skill_query = self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "match": {"skills": "Python"}
                    }
                }
            )
            self.print_search_results(skill_query, "Python技能员工")
            
            # 4. 范围查询
            print("\n--- 范围查询：薪资在10000-16000之间的员工 ---")
            range_query = self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "range": {
                            "salary": {
                                "gte": 10000,
                                "lte": 16000
                            }
                        }
                    }
                }
            )
            self.print_search_results(range_query, "薪资范围查询")
            
            # 5. 复合查询
            print("\n--- 复合查询：技术部且在职的员工 ---")
            bool_query = self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"department": "技术部"}},
                                {"term": {"is_active": True}}
                            ]
                        }
                    }
                }
            )
            self.print_search_results(bool_query, "技术部在职员工")
            
            # 6. 聚合查询
            print("\n--- 聚合查询：按部门统计员工数量 ---")
            agg_query = self.es.search(
                index=self.index_name,
                body={
                    "aggs": {
                        "employees_by_department": {
                            "terms": {
                                "field": "department",
                                "size": 10
                            }
                        }
                    },
                    "size": 0
                }
            )
            
            print("部门员工统计:")
            for bucket in agg_query['aggregations']['employees_by_department']['buckets']:
                print(f"  {bucket['key']}: {bucket['doc_count']} 人")
            
        except Exception as e:
            print(f"❌ 读取数据失败: {e}")
    
    def update_test_data(self):
        """更新测试数据 (Update)"""
        print("\n=== 更新测试数据 ===")
        
        try:
            # 1. 更新单个文档
            print("\n--- 更新员工薪资 ---")
            update_doc = {
                "doc": {
                    "salary": 16000.0,
                    "position": "资深软件工程师"
                }
            }
            
            response = self.es.update(
                index=self.index_name,
                id="EMP001",
                body=update_doc
            )
            print(f"✅ 员工 EMP001 信息已更新")
            
            # 2. 部分更新（使用脚本）
            print("\n--- 使用脚本批量调整薪资 ---")
            script_update = {
                "script": {
                    "source": "ctx._source.salary = ctx._source.salary * 1.1",
                    "lang": "painless"
                },
                "query": {
                    "term": {"department": "技术部"}
                }
            }
            
            response = self.es.update_by_query(
                index=self.index_name,
                body=script_update
            )
            print(f"✅ 已为技术部员工调整薪资，更新了 {response['updated']} 条记录")
            
            # 3. 验证更新结果
            print("\n--- 验证更新结果 ---")
            updated_emp = self.es.get(index=self.index_name, id="EMP001")
            print(f"EMP001更新后信息: {json.dumps(updated_emp['_source'], ensure_ascii=False, indent=2)}")
            
            # 刷新索引
            self.es.indices.refresh(index=self.index_name)
            
        except Exception as e:
            print(f"❌ 更新数据失败: {e}")
    
    def delete_test_data(self):
        """删除测试数据 (Delete)"""
        print("\n=== 删除测试数据 ===")
        
        try:
            # 1. 按查询删除薪资低于9000的员工（包括EMP004）
            print("\n--- 删除薪资低于9000的员工 ---")
            delete_query = {
                "query": {
                    "range": {
                        "salary": {"lt": 9000}
                    }
                }
            }
            
            response = self.es.delete_by_query(
                index=self.index_name,
                body=delete_query
            )
            print(f"✅ 已删除 {response['deleted']} 名薪资低于9000的员工")
            
            # 刷新索引确保删除生效
            self.es.indices.refresh(index=self.index_name)
            
            # 2. 删除另一个单个文档（演示单个删除）
            print("\n--- 删除指定员工 ---")
            try:
                response = self.es.delete(index=self.index_name, id="EMP002")
                print(f"✅ 员工 EMP002 已被删除")
            except Exception as e:
                if "not_found" in str(e).lower():
                    print(f"ℹ️ 员工 EMP002 不存在或已被删除")
                else:
                    print(f"❌ 删除员工 EMP002 失败: {e}")
            
            # 3. 验证删除结果
            print("\n--- 验证删除结果：剩余员工 ---")
            remaining_docs = self.es.search(
                index=self.index_name,
                body={
                    "query": {"match_all": {}},
                    "size": 10
                }
            )
            self.print_search_results(remaining_docs, "剩余员工")
            
        except Exception as e:
            print(f"❌ 删除数据失败: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始 Elasticsearch CRUD 测试")
        try:
            self.create_test_data()
            self.read_test_data()
            self.update_test_data()
            self.delete_test_data()
            print("\n✅ 所有 Elasticsearch CRUD 测试完成")
        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")
            print(traceback.format_exc())
    
    def cleanup(self):
        """清理测试索引"""
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            print(f"🗑️ 测试索引 {self.index_name} 已清理")


if __name__ == "__main__":
    es_test = ESCRUDTest()

    try:
        es_test.run_all_tests()
    finally:
        # 可选：测试完成后清理索引
        # es_test.cleanup()
        pass

