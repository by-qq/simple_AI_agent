# mysql_pool.py
import pymysql
from pymysql import cursors
from dbutils.pooled_db import PooledDB  # 使用 DBUtils 的连接池
from contextlib import contextmanager
from app.config import settings

# 创建全局连接池
_pool = None


def init_pool():
    """初始化连接池"""
    global _pool
    if _pool is None:
        _pool = PooledDB(
            creator=pymysql,  # 使用 pymysql 作为连接创建器
            mincached=settings.MYSQL_POOL_MIN_SIZE if hasattr(settings, 'MYSQL_POOL_MIN_SIZE') else 1,
            maxcached=settings.MYSQL_POOL_MAX_SIZE if hasattr(settings, 'MYSQL_POOL_MAX_SIZE') else 10,
            maxconnections=settings.MYSQL_POOL_MAX_SIZE if hasattr(settings, 'MYSQL_POOL_MAX_SIZE') else 10,
            blocking=True,  # 连接池耗尽时阻塞等待
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DB,
            charset='utf8mb4',
            cursorclass=cursors.DictCursor,
            autocommit=True,
        )
    return _pool


@contextmanager
def get_conn():
    """从连接池获取连接"""
    if _pool is None:
        init_pool()

    conn = _pool.connection()  # DBUtils 使用 .connection() 方法
    try:
        yield conn
    finally:
        conn.close()  # 将连接归还给连接池


def close_pool():
    """关闭连接池"""
    global _pool
    if _pool:
        _pool.close()  # DBUtils 的关闭方法
        _pool = None