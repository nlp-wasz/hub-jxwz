from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
import traceback

# 创建数据库引擎
engine = create_engine('sqlite:///my_job_orm.db', echo=True)

# 创建 ORM 模型的基类
Base = declarative_base()


# --- 定义 ORM 模型 ---

class User(Base):
    """用户表"""
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    age = Column(Integer)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 定义与 Order 表的关系
    orders = relationship("Order", back_populates="user")

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}', age={self.age})>"


class Product(Base):
    """商品表"""
    __tablename__ = 'products'

    product_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    category = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    # 定义与 Order 表的关系
    orders = relationship("Order", back_populates="product")

    def __repr__(self):
        return f"<Product(name='{self.name}', price={self.price}, stock={self.stock})>"


class Order(Base):
    """订单表"""
    __tablename__ = 'orders'

    order_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    product_id = Column(Integer, ForeignKey('products.product_id'), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    total_price = Column(Float, nullable=False)
    order_date = Column(DateTime, default=datetime.utcnow)

    # 定义关系
    user = relationship("User", back_populates="orders")
    product = relationship("Product", back_populates="orders")

    def __repr__(self):
        return f"<Order(user_id={self.user_id}, product_id={self.product_id}, quantity={self.quantity})>"


class ORMCRUDTest:
    """ORM增删改查测试类"""
    
    def __init__(self):
        # 创建数据库和表
        Base.metadata.create_all(engine)
        print("数据库和表已成功创建。")
        
        # 创建会话
        Session = sessionmaker(bind=engine)
        self.session = Session()
    
    def create_test_data(self):
        """创建测试数据 (Create)"""
        print("\n=== 创建测试数据 ===")
        try:
            # 创建用户
            user1 = User(username='alice', email='alice@test.com', age=25)
            user2 = User(username='bob', email='bob@test.com', age=30)
            user3 = User(username='charlie', email='charlie@test.com', age=28)
            
            # 创建商品
            product1 = Product(name='笔记本电脑', price=5999.99, stock=10, category='电子产品')
            product2 = Product(name='无线鼠标', price=89.99, stock=50, category='电子产品')
            product3 = Product(name='咖啡杯', price=29.99, stock=100, category='生活用品')
            
            # 添加到会话
            self.session.add_all([user1, user2, user3, product1, product2, product3])
            self.session.commit()
            
            # 创建订单
            order1 = Order(user=user1, product=product1, quantity=1, total_price=5999.99)
            order2 = Order(user=user2, product=product2, quantity=2, total_price=179.98)
            order3 = Order(user=user1, product=product3, quantity=3, total_price=89.97)
            
            self.session.add_all([order1, order2, order3])
            self.session.commit()
            
            print("✅ 测试数据创建成功")
            
        except Exception as e:
            print(f"❌ 创建数据失败: {e}")
            self.session.rollback()
    
    def read_test_data(self):
        """读取测试数据 (Read)"""
        print("\n=== 读取测试数据 ===")
        try:
            # 1. 查询所有用户
            print("\n--- 所有用户 ---")
            users = self.session.query(User).all()
            for user in users:
                print(f"用户: {user}")
            
            # 2. 条件查询
            print("\n--- 年龄大于25的用户 ---")
            young_users = self.session.query(User).filter(User.age > 25).all()
            for user in young_users:
                print(f"用户: {user}")
            
            # 3. 关联查询
            print("\n--- 所有订单及其用户和商品信息 ---")
            orders = self.session.query(Order).join(User).join(Product).all()
            for order in orders:
                print(f"订单: {order.order_id}, 用户: {order.user.username}, "
                      f"商品: {order.product.name}, 数量: {order.quantity}")
            
            # 4. 聚合查询
            print("\n--- 商品库存统计 ---")
            products = self.session.query(Product).filter(Product.stock > 0).all()
            for product in products:
                print(f"商品: {product.name}, 库存: {product.stock}")
                
        except Exception as e:
            print(f"❌ 读取数据失败: {e}")
    
    def update_test_data(self):
        """更新测试数据 (Update)"""
        print("\n=== 更新测试数据 ===")
        try:
            # 1. 更新单个字段
            user_to_update = self.session.query(User).filter_by(username='alice').first()
            if user_to_update:
                user_to_update.age = 26
                self.session.commit()
                print(f"✅ 用户 {user_to_update.username} 的年龄已更新为 {user_to_update.age}")
            
            # 2. 批量更新
            self.session.query(Product).filter(Product.category == '电子产品').update(
                {Product.stock: Product.stock + 5}
            )
            self.session.commit()
            print("✅ 所有电子产品库存已增加5")
            
            # 3. 验证更新结果
            print("\n--- 更新后的数据 ---")
            updated_user = self.session.query(User).filter_by(username='alice').first()
            print(f"用户信息: {updated_user}")
            
            updated_products = self.session.query(Product).filter(Product.category == '电子产品').all()
            for product in updated_products:
                print(f"商品: {product.name}, 库存: {product.stock}")
                
        except Exception as e:
            print(f"❌ 更新数据失败: {e}")
            self.session.rollback()
    
    def delete_test_data(self):
        """删除测试数据 (Delete)"""
        print("\n=== 删除测试数据 ===")
        try:
            # 1. 删除单个记录
            user_to_delete = self.session.query(User).filter_by(username='charlie').first()
            if user_to_delete:
                self.session.delete(user_to_delete)
                self.session.commit()
                print(f"✅ 用户 charlie 已被删除")
            
            # 2. 批量删除
            deleted_count = self.session.query(Product).filter(Product.stock < 20).delete()
            self.session.commit()
            print(f"✅ 已删除 {deleted_count} 个库存不足的商品")
            
            # 3. 验证删除结果
            print("\n--- 删除后剩余的用户 ---")
            remaining_users = self.session.query(User).all()
            for user in remaining_users:
                print(f"用户: {user}")
            
            print("\n--- 删除后剩余的商品 ---")
            remaining_products = self.session.query(Product).all()
            for product in remaining_products:
                print(f"商品: {product}")
                
        except Exception as e:
            print(f"❌ 删除数据失败: {e}")
            self.session.rollback()
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始 ORM CRUD 测试")
        try:
            self.create_test_data()
            self.read_test_data()
            self.update_test_data()
            self.delete_test_data()
            print("\n✅ 所有 ORM CRUD 测试完成")
        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")
            print(traceback.format_exc())
        finally:
            self.close()
    
    def close(self):
        """关闭会话"""
        self.session.close()
        print("📝 数据库会话已关闭")


if __name__ == "__main__":
    # 创建测试实例并运行测试
    crud_test = ORMCRUDTest()
    crud_test.run_all_tests()

