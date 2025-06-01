"""
Database configuration and connection management.
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import redis
import structlog
from typing import Generator

from .config import settings

logger = structlog.get_logger()

# SQLAlchemy setup
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Redis setup
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

# Metadata for reflection
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error("Database session error", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """
    Get Redis client.
    
    Returns:
        redis.Redis: Redis client instance
    """
    return redis_client


async def check_database_connection() -> bool:
    """
    Check if database connection is healthy.
    
    Returns:
        bool: True if connection is healthy, False otherwise
    """
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False


async def check_redis_connection() -> bool:
    """
    Check if Redis connection is healthy.
    
    Returns:
        bool: True if connection is healthy, False otherwise
    """
    try:
        redis_client.ping()
        return True
    except Exception as e:
        logger.error("Redis connection check failed", error=str(e))
        return False


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise


def drop_tables():
    """Drop all database tables."""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error("Failed to drop database tables", error=str(e))
        raise 