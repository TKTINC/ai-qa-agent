"""Tests for database models and management."""

import pytest
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.core.database import Base, AnalysisSession, CodeComponent, GeneratedTest, TaskStatus


@pytest.fixture
def test_db():
    """Create a test database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    session.close()


class TestAnalysisSession:
    """Test AnalysisSession model."""
    
    def test_create_analysis_session(self, test_db):
        """Test creating an analysis session."""
        session = AnalysisSession(
            repository_name="test-repo",
            repository_url="https://github.com/test/repo"
        )
        
        test_db.add(session)
        test_db.commit()
        
        assert session.id is not None
        assert session.total_files == 0
        assert session.status == "pending"
        assert session.created_at is not None
    
    def test_analysis_session_properties(self, test_db):
        """Test analysis session properties."""
        session = AnalysisSession(repository_name="test-repo", status="completed")
        assert session.is_complete == True
        
        session.status = "pending"
        assert session.is_complete == False


class TestCodeComponent:
    """Test CodeComponent model."""
    
    def test_create_code_component(self, test_db):
        """Test creating a code component."""
        analysis_session = AnalysisSession(repository_name="test-repo")
        test_db.add(analysis_session)
        test_db.commit()
        
        component = CodeComponent(
            session_id=analysis_session.id,
            name="test_function",
            component_type="function",
            file_path="src/test.py",
            full_name="src.test.test_function",
            line_start=1,
            line_end=10,
            source_code="def test_function(): pass",
            language="python"
        )
        
        test_db.add(component)
        test_db.commit()
        
        assert component.id is not None
        assert component.complexity == 1
        assert component.is_testable == False  # Complexity too low
