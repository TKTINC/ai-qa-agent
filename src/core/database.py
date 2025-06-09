"""
AI QA Agent - Database Models and Management
Comprehensive database setup with SQLAlchemy models and session management.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Float, Boolean, JSON, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from sqlalchemy.pool import StaticPool
from typing import Generator, Optional, Dict, Any, List
import logging
import uuid
from datetime import datetime

from .config import get_settings

logger = logging.getLogger(__name__)

# Base class for all database models
Base = declarative_base()


class AnalysisSession(Base):
    """Stores repository analysis sessions."""
    __tablename__ = "analysis_sessions"
    
    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    repository_name = Column(String, nullable=False, index=True)
    repository_url = Column(String, nullable=True)
    repository_hash = Column(String, nullable=True, index=True)
    
    # Analysis metadata
    total_files = Column(Integer, default=0)
    analyzed_files = Column(Integer, default=0)
    skipped_files = Column(Integer, default=0)
    language_distribution = Column(JSON, default=dict)
    complexity_metrics = Column(JSON, default=dict)
    
    # Processing status
    status = Column(String, default="pending", index=True)
    progress_percentage = Column(Integer, default=0)
    current_step = Column(String, nullable=True)
    
    # Timing information
    created_at = Column(DateTime, server_default=func.now(), index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Analysis configuration
    analysis_config = Column(JSON, default=dict)
    
    # Relationships
    components = relationship("CodeComponent", back_populates="session", cascade="all, delete-orphan")
    generated_tests = relationship("GeneratedTest", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_session_status_created', 'status', 'created_at'),
        Index('idx_session_repo_name', 'repository_name'),
    )
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate analysis duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if analysis is complete."""
        return self.status in ["completed", "failed"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "repository_name": self.repository_name,
            "repository_url": self.repository_url,
            "total_files": self.total_files,
            "analyzed_files": self.analyzed_files,
            "language_distribution": self.language_distribution,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


class CodeComponent(Base):
    """Stores analyzed code components."""
    __tablename__ = "code_components"
    
    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey('analysis_sessions.id'), nullable=False, index=True)
    
    # Component identification
    name = Column(String, nullable=False, index=True)
    component_type = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False, index=True)
    full_name = Column(String, nullable=False)
    
    # Location information
    line_start = Column(Integer, nullable=False)
    line_end = Column(Integer, nullable=False)
    column_start = Column(Integer, nullable=True)
    column_end = Column(Integer, nullable=True)
    
    # Code analysis metrics
    complexity = Column(Integer, default=1)
    loc = Column(Integer, default=0)
    maintainability_index = Column(Float, nullable=True)
    
    # Component metadata
    parameters = Column(JSON, default=list)
    return_type = Column(String, nullable=True)
    dependencies = Column(JSON, default=list)
    decorators = Column(JSON, default=list)
    
    # Code content
    source_code = Column(Text, nullable=False)
    docstring = Column(Text, nullable=True)
    comments = Column(JSON, default=list)
    
    # Analysis metadata
    language = Column(String, nullable=False)
    encoding = Column(String, default="utf-8")
    ast_data = Column(JSON, nullable=True)
    
    # Processing information
    created_at = Column(DateTime, server_default=func.now())
    analysis_duration = Column(Float, nullable=True)
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="components")
    generated_tests = relationship("GeneratedTest", back_populates="component", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_component_session_type', 'session_id', 'component_type'),
        Index('idx_component_name_type', 'name', 'component_type'),
        Index('idx_component_file_path', 'file_path'),
        Index('idx_component_complexity', 'complexity'),
    )
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified component name."""
        return f"{self.file_path}::{self.full_name}"
    
    @property
    def is_testable(self) -> bool:
        """Determine if component is suitable for test generation."""
        if self.complexity < 2 and self.component_type == "function":
            return False
        if self.component_type in ["import", "variable"]:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "component_type": self.component_type,
            "file_path": self.file_path,
            "full_name": self.full_name,
            "complexity": self.complexity,
            "loc": self.loc,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "dependencies": self.dependencies,
            "language": self.language,
            "is_testable": self.is_testable,
        }


class GeneratedTest(Base):
    """Stores AI-generated tests for code components."""
    __tablename__ = "generated_tests"
    
    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey('analysis_sessions.id'), nullable=False, index=True)
    component_id = Column(String, ForeignKey('code_components.id'), nullable=False, index=True)
    
    # Test identification
    test_name = Column(String, nullable=False)
    test_type = Column(String, nullable=False, index=True)
    test_framework = Column(String, default="pytest")
    
    # Generated content
    test_code = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)
    test_description = Column(Text, nullable=True)
    
    # AI generation metadata
    ai_model = Column(String, nullable=False)
    ai_provider = Column(String, nullable=False)
    generation_prompt = Column(Text, nullable=True)
    generation_config = Column(JSON, default=dict)
    
    # Quality metrics
    confidence_score = Column(Float, default=0.0)
    quality_score = Column(Float, default=0.0)
    complexity_score = Column(Float, nullable=True)
    coverage_estimate = Column(Float, nullable=True)
    
    # Validation status
    is_valid = Column(Boolean, default=False)
    is_executable = Column(Boolean, default=False)
    validation_errors = Column(JSON, default=list)
    execution_result = Column(JSON, nullable=True)
    
    # Test characteristics
    test_inputs = Column(JSON, default=list)
    expected_outputs = Column(JSON, default=list)
    mocks_required = Column(JSON, default=list)
    dependencies = Column(JSON, default=list)
    
    # Processing timestamps
    created_at = Column(DateTime, server_default=func.now())
    validated_at = Column(DateTime, nullable=True)
    last_executed_at = Column(DateTime, nullable=True)
    
    # Performance metrics
    generation_time = Column(Float, nullable=True)
    validation_time = Column(Float, nullable=True)
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="generated_tests")
    component = relationship("CodeComponent", back_populates="generated_tests")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_test_session_type', 'session_id', 'test_type'),
        Index('idx_test_component_valid', 'component_id', 'is_valid'),
        Index('idx_test_quality_score', 'quality_score'),
        Index('idx_test_ai_model', 'ai_model'),
    )
    
    @property
    def is_ready_for_execution(self) -> bool:
        """Check if test is ready for execution."""
        return self.is_valid and not self.validation_errors
    
    @property
    def overall_score(self) -> float:
        """Calculate overall test score."""
        scores = [s for s in [self.confidence_score, self.quality_score] if s is not None]
        return sum(scores) / len(scores) if scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "test_name": self.test_name,
            "test_type": self.test_type,
            "test_framework": self.test_framework,
            "explanation": self.explanation,
            "ai_model": self.ai_model,
            "confidence_score": self.confidence_score,
            "quality_score": self.quality_score,
            "is_valid": self.is_valid,
            "is_executable": self.is_executable,
            "overall_score": self.overall_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TaskStatus(Base):
    """Tracks background task execution status."""
    __tablename__ = "task_status"
    
    # Primary identification
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_type = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    
    # Task identification
    task_name = Column(String, nullable=False)
    task_description = Column(String, nullable=True)
    parent_task_id = Column(String, nullable=True, index=True)
    
    # Status tracking
    status = Column(String, default="pending", index=True)
    progress = Column(Integer, default=0)
    current_step = Column(String, nullable=True)
    total_steps = Column(Integer, nullable=True)
    current_step_number = Column(Integer, default=0)
    
    # Results and errors
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    error_traceback = Column(Text, nullable=True)
    
    # Timing information
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Task configuration
    task_config = Column(JSON, default=dict)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Resource usage
    memory_usage_mb = Column(Float, nullable=True)
    cpu_time_seconds = Column(Float, nullable=True)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_task_status_type', 'status', 'task_type'),
        Index('idx_task_session_created', 'session_id', 'created_at'),
        Index('idx_task_parent_child', 'parent_task_id'),
    )
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == "running"
    
    @property
    def is_complete(self) -> bool:
        """Check if task is complete (success or failure)."""
        return self.status in ["completed", "failed", "cancelled"]
    
    def update_progress(self, progress: int, step: str = None, step_number: int = None):
        """Update task progress."""
        self.progress = max(0, min(100, progress))
        if step:
            self.current_step = step
        if step_number is not None:
            self.current_step_number = step_number
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "task_name": self.task_name,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_step_number": self.current_step_number,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal = None
        self._setup_database()
    
    def _setup_database(self):
        """Initialize database engine and session factory."""
        database_url = self.settings.get_database_url()
        
        # Configure engine based on database type
        if database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
            if ":memory:" in database_url:
                connect_args.update({
                    "poolclass": StaticPool,
                    "echo": self.settings.database_echo
                })
            
            self.engine = create_engine(
                database_url,
                echo=self.settings.database_echo,
                connect_args=connect_args
            )
        else:
            # PostgreSQL or other databases
            self.engine = create_engine(
                database_url,
                echo=self.settings.database_echo,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        
        logger.info(f"Database engine created for: {database_url}")
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (useful for testing)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session for dependency injection."""
        db = self.SessionLocal()
        try:
            yield db
        except Exception as e:
            logger.error(f"Database session error: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def health_check(self) -> bool:
        """Check database connection health."""
        try:
            db = self.SessionLocal()
            db.execute("SELECT 1")
            db.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            db = self.SessionLocal()
            
            stats = {
                "analysis_sessions": db.query(AnalysisSession).count(),
                "code_components": db.query(CodeComponent).count(),
                "generated_tests": db.query(GeneratedTest).count(),
                "active_tasks": db.query(TaskStatus).filter(
                    TaskStatus.status.in_(["pending", "running"])
                ).count(),
                "completed_sessions": db.query(AnalysisSession).filter(
                    AnalysisSession.status == "completed"
                ).count(),
                "valid_tests": db.query(GeneratedTest).filter(
                    GeneratedTest.is_valid == True
                ).count(),
            }
            
            db.close()
            return stats
        except Exception as e:
            logger.error(f"Error collecting database stats: {e}")
            return {}


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    yield from db_manager.get_session()


def create_tables():
    """Initialize database tables."""
    db_manager.create_tables()


def drop_tables():
    """Drop all database tables."""
    db_manager.drop_tables()


def get_db_stats() -> Dict[str, Any]:
    """Get database statistics."""
    return db_manager.get_stats()
