from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, or_, and_
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, joinedload, selectinload, contains_eager

from KnowledgeClass import KnowledgeDocument, KnowledgeDatabase  # 映射实体类对象

# 通过 Session 对象，对数据库进行 crud 操作

# 创建 Engine 和 Session
engine = create_engine("sqlite:///C:\\ai\\sqlite\\database\\localhost.db", echo=True)
Session = sessionmaker(bind=engine)
session = Session()

print("-" * 200)

# 通过 session 操作数据库
# 一：查询
# 1.query(Model) 查询（通过映射的实体类 查询对应表中的数据信息，返回的是 可迭代对象）
# ①all()：获取所有查询数据
query_all = session.query(KnowledgeDatabase).all()
for query in query_all:
    print(query)

# ②first()：获取第一行数据
query_first = session.query(KnowledgeDatabase).first()
print(query_first)

# ③one()：只查询一条，如果存在多条则报错
query_one = session.query(KnowledgeDatabase).one()
query_one = session.query(KnowledgeDocument).one()  # 查询到多条，报错
print(query_one)

# ④获取查询数量
query_count = session.query(KnowledgeDatabase).count()
print(query_count)

# 2.filter() 和 filter_by()  条件查询
# filter() 使用复杂场景（支持 ==  !=  >  >=  <  <=  like  in_  and_  or_  between 等）
# filter_by 仅支持 ==
# ① == 等值条件
query_eq = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_id == 1) \
    .all()
for query in query_eq:
    print(query)

# ② !=
query_ne = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_id != 1) \
    .all()
for query in query_ne:
    print(query)

# ③ >  >=  <  <=  一样
query_ge = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_id >= 3) \
    .all()
for query in query_ge:
    print(query)

# ④ like() 模糊查询（ _ 单个字符占位符， % 对个字符占位符）
query_like = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_title.like("_周杰伦%")) \
    .all()
for query in query_like:
    print(query)

# ⑤ or_(if1, if2)  and_(if1, if2)  多条件查询（需要单独导入）
query_or = session.query(KnowledgeDocument) \
    .filter(or_(KnowledgeDocument.document_title.like("_周杰伦%"), KnowledgeDocument.document_title.like("%稻香%"))) \
    .all()
for query in query_or:
    print(query)
print("-" * 200)
query_and = session.query(KnowledgeDocument) \
    .filter(and_(KnowledgeDocument.document_title.like("_周杰伦%"), KnowledgeDocument.document_title.like("%稻香%"))) \
    .all()
for query in query_and:
    print(query)

# ⑥ in()
query_in = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_id.in_([1, 2, 3])) \
    .all()
for query in query_in:
    print(query)

# ⑥ is_() 和  is_not  判断是否为None（或者特定的值）
query_is = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_category.is_(None)) \
    .all()
for query in query_is:
    print(query)
print("-" * 200)
query_is = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_category.is_("歌曲1")) \
    .all()
for query in query_is:
    print(query)

# ⑦排序 asc()：升序，默认  desc()：降序
query_desc = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_title.is_not(None)) \
    .order_by(KnowledgeDocument.document_id.desc()) \
    .all()
for query in query_desc:
    print(query)

# ⑧ 分页 查询
curr_page = 2
page_size = 2
query_limit = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_title.is_not(None)) \
    .order_by(KnowledgeDocument.document_id) \
    .offset((curr_page - 1) * page_size) \
    .limit(page_size) \
    .all()
for query in query_limit:
    print(query)

# ⑨ join() 内连接查询
query_join = session.query(KnowledgeDatabase, KnowledgeDocument) \
    .join(KnowledgeDocument, KnowledgeDatabase.knowledge_id == KnowledgeDocument.knowledge_id) \
    .filter(KnowledgeDocument.document_title.not_like("%周杰伦%")) \
    .all()
for knowledge, document in query_join:
    print(knowledge)
    print(document)
    print("-" * 200)

# ⭐ 利用 relationship 直接将对应的文档信息 存储到 KnowledgeDatabase.documents 属性中（关联属性）
join_relationship = session.query(KnowledgeDatabase) \
    .join(KnowledgeDatabase.documents) \
    .filter(KnowledgeDocument.document_title.not_like("%周杰伦%")) \
    .all()
for knowledge in join_relationship:
    print(knowledge)
    for document in knowledge.documents:
        print(document)
    print("-" * 200)

# ⭐ contains_eager()：在进行 关联查询时，可以在 获取documents时 也遵循 filter() 中的过滤规则
query_join = session.query(KnowledgeDatabase) \
    .join(KnowledgeDatabase.documents) \
    .filter(KnowledgeDocument.document_title.not_like("%周杰伦%")) \
    .options(contains_eager(KnowledgeDatabase.documents)) \
    .all()
for knowledge in query_join:
    print(knowledge)
    for document in knowledge.documents:
        print(document)
    print("-" * 200)

# ⑩ outerjoin() 左外连接
query_outerjoin = session.query(KnowledgeDatabase, KnowledgeDocument) \
    .outerjoin(KnowledgeDocument, KnowledgeDatabase.knowledge_id == KnowledgeDocument.knowledge_id) \
    .filter(KnowledgeDocument.document_title.not_like("%周杰伦%")) \
    .all()
for knowledge, document in query_outerjoin:
    print(knowledge)
    print(document)
    print("-" * 200)

# ⭐ 利用 relationship 直接将对应的文档信息 存储到 KnowledgeDatabase.documents 属性中（关联属性）
outerjoin_relationship = session.query(KnowledgeDatabase) \
    .outerjoin(KnowledgeDatabase.documents) \
    .filter(KnowledgeDocument.document_title.not_like("%周杰伦%")) \
    .options(contains_eager(KnowledgeDatabase.documents)) \
    .all()
for knowledge in outerjoin_relationship:
    print(knowledge)
    for document in knowledge.documents:
        print(document)
    print("-" * 200)

# 二：options() 关联表加载 关联字段 数据信息
# 对两个关联表（通过 relationship 关联），在查询时同时将对应数据加载到关联属性上（documents，knowledge）
# 这样 无需调用属性时 再执行查询sql（避免N+1次查询，现在一次性查询来，而无需每次调用属性都要查询一次）

# ⭐ joinedload 和 selectinload 不会受 .filter(KnowledgeDocument.xxx) 影响，它会把所有文档都加载进来。
# ① joinedload()  一条sql（关联查询，并且把数据加载到 对应的 关联属性）
query_joinedload = session.query(KnowledgeDatabase) \
    .options(joinedload(KnowledgeDatabase.documents)) \
    .all()
for knowledge in query_joinedload:
    print(knowledge)
    for document in knowledge.documents:
        if "周杰伦" not in document.document_title:
            print(document)
    print("-" * 200)

# ② selectinload()  两条sql（先查询 主表，再根据主表id 查询从表）
query_selectinload = session.query(KnowledgeDatabase) \
    .options(selectinload(KnowledgeDatabase.documents)) \
    .all()
for knowledge in query_selectinload:
    print(knowledge)
    for document in knowledge.documents:
        if "周杰伦" not in document.document_title:
            print(document)
    print("-" * 200)

#
# 三：增加数据（add）
# 1.添加一行数据
# 创建要添加的 对象信息
document1 = KnowledgeDocument(document_title="《七里香》", document_category="歌曲5", knowledge_id=1)
session.add(document1)
session.commit()
print(document1)  # 包含了自动生成的 knowledge_id 和 create_dt  update_dt

# 2.添加多行数据
document1 = KnowledgeDocument(document_title="《菊花台》", document_category="电影《满城尽带黄金甲》主题曲", knowledge_id=1)
document2 = KnowledgeDocument(document_title="《东风破》", document_category="开创“三古三新”中国风典范", knowledge_id=1)
document3 = KnowledgeDocument(document_title="《发如雪》", document_category="“你发如雪，凄美了离别…”", knowledge_id=1)
session.add_all([document1, document2, document3])
session.commit()
print(f"document1 = {document1}")  # 包含了自动生成的 knowledge_id 和 create_dt  update_dt
print(f"document2 = {document2}")  # 包含了自动生成的 knowledge_id 和 create_dt  update_dt
print(f"document3 = {document3}")  # 包含了自动生成的 knowledge_id 和 create_dt  update_dt

# 四：update()  修改
# 1. 先查询 后修改
update_sel = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_id == 1) \
    .one()
print(f"要修改的数据：{update_sel}")
if update_sel:
    update_sel.file_path = "/file_path"
    session.commit()
print(f"修改后的数据：{update_sel}")

# 2.直接修改（不触发ORM事件，不会将数据加载到内存）
update_notsel = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_id == 2) \
    .update({"file_path": "/file", KnowledgeDocument.file_type: "pdf"})
session.commit()

# 五：delete()  删除
# 1. 先查询 后删除
delete_sel = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_id.is_(8)) \
    .one()
print(f"要删除的数据：{delete_sel}")
if delete_sel:
    session.delete(delete_sel)
    session.commit()

# 2.直接删除（不触发ORM事件，不会将数据加载到内存）
delete_notsel = session.query(KnowledgeDocument) \
    .filter(KnowledgeDocument.document_id == 7) \
    .delete()
# print(delete_notsel) # 删除的行数
session.commit()
