#!/usr/bin/env python3
"""
Enhanced Production Multi-Agent AI System
A robust, scalable, and secure multi-agent system with advanced features.
"""

import os
import json
import uuid
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
import aiohttp
import yaml
from pathlib import Path
import hashlib
import traceback
from contextlib import asynccontextmanager
import signal
import sys
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import queue
import secrets
import aiosqlite
from collections import defaultdict, deque
import weakref
import resource
import psutil
from typing_extensions import TypeAlias
import pydantic
from pydantic import BaseModel, Field, validator
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential


# ============================================================================
# ENHANCED CONFIGURATION AND MODELS
# ============================================================================

class SystemConfig(BaseModel):
    """Pydantic-based configuration with validation"""
    
    class DatabaseConfig(BaseModel):
        path: str = "data/system.db"
        max_connections: int = Field(default=20, ge=1, le=100)
        connection_timeout: int = Field(default=30, ge=1)
        pragma_settings: Dict[str, Any] = Field(default_factory=lambda: {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": 10000,
            "foreign_keys": True,
            "temp_store": "memory"
        })
    
    class ModelsConfig(BaseModel):
        default: str = "mock-model"
        timeout: int = Field(default=120, ge=1)
        max_tokens: int = Field(default=4000, ge=1)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        max_retries: int = Field(default=3, ge=0)
        rate_limit_per_minute: int = Field(default=60, ge=1)
    
    class AgentsConfig(BaseModel):
        max_count: int = Field(default=100, ge=1)
        response_timeout: int = Field(default=300, ge=1)
        health_check_interval: int = Field(default=60, ge=10)
        max_concurrent_tasks: int = Field(default=10, ge=1)
    
    class SecurityConfig(BaseModel):
        api_key_length: int = Field(default=32, ge=16)
        session_timeout: int = Field(default=3600, ge=300)
        max_task_size: int = Field(default=1048576, ge=1024)  # 1MB
        allowed_models: List[str] = Field(default_factory=lambda: ["mock-model", "gpt-3.5-turbo", "gpt-4", "claude-3"])
        enable_audit_log: bool = True
    
    class ProcessingConfig(BaseModel):
        auto_process: bool = True
        worker_count: int = Field(default=4, ge=1, le=20)
        batch_size: int = Field(default=5, ge=1)
        priority_queue_enabled: bool = True
        task_timeout: int = Field(default=600, ge=60)
    
    class MonitoringConfig(BaseModel):
        metrics_enabled: bool = True
        performance_tracking: bool = True
        alert_thresholds: Dict[str, float] = Field(default_factory=lambda: {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 5.0,
            "response_time": 30.0
        })
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from file with validation"""
        config_file = Path(config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                return cls(**config_data)
            except Exception as e:
                logging.error(f"Failed to load config from {config_path}: {e}")
        
        return cls()


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class AgentRole(Enum):
    """Enhanced agent roles"""
    ASSISTANT = "assistant"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    OPTIMIZER = "optimizer"


class TaskStatus(Enum):
    """Task status with more granular states"""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class SystemEvent(Enum):
    """System events for monitoring"""
    AGENT_CREATED = auto()
    AGENT_ACTIVATED = auto()
    AGENT_DEACTIVATED = auto()
    TASK_SUBMITTED = auto()
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    SYSTEM_OVERLOAD = auto()
    ERROR_THRESHOLD_EXCEEDED = auto()


# ============================================================================
# ENHANCED DATA MODELS
# ============================================================================

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    level: int  # 1-10 skill level
    description: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class TaskMetrics:
    """Task execution metrics"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: Optional[float] = None
    tokens_used: int = 0
    retries: int = 0
    memory_peak: float = 0.0  # MB
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class Agent:
    """Enhanced agent with capabilities and health monitoring"""
    id: str
    name: str
    role: AgentRole
    description: str
    model: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    last_activity: Optional[datetime] = None
    health_score: float = 1.0
    current_load: int = 0
    max_concurrent_tasks: int = 5
    average_response_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    @property
    def is_available(self) -> bool:
        return self.is_active and self.current_load < self.max_concurrent_tasks and self.health_score > 0.5
    
    def can_handle_task(self, task_description: str, required_capabilities: List[str] = None) -> float:
        """Calculate agent's suitability for a task (0.0 to 1.0)"""
        if not self.is_available:
            return 0.0
        
        score = 0.5  # Base score
        
        # Role-based scoring
        task_lower = task_description.lower()
        role_keywords = {
            AgentRole.RESEARCHER: ['research', 'find', 'investigate', 'analyze', 'study'],
            AgentRole.ANALYST: ['analyze', 'data', 'statistics', 'report', 'interpret'],
            AgentRole.COORDINATOR: ['coordinate', 'manage', 'organize', 'schedule'],
            AgentRole.SPECIALIST: ['expert', 'specialized', 'technical', 'advanced'],
            AgentRole.VALIDATOR: ['validate', 'verify', 'check', 'review', 'quality'],
            AgentRole.OPTIMIZER: ['optimize', 'improve', 'enhance', 'performance']
        }
        
        if self.role in role_keywords:
            keyword_matches = sum(1 for keyword in role_keywords[self.role] if keyword in task_lower)
            score += min(keyword_matches * 0.1, 0.3)
        
        # Capability scoring
        if required_capabilities:
            matching_caps = [cap for cap in self.capabilities if cap.name in required_capabilities]
            if matching_caps:
                avg_level = sum(cap.level for cap in matching_caps) / len(matching_caps)
                score += (avg_level / 10) * 0.2
        
        # Health and performance scoring
        score *= self.health_score
        score *= (1.0 - (self.current_load / self.max_concurrent_tasks) * 0.2)
        
        return min(score, 1.0)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'role': self.role.value,
            'description': self.description,
            'model': self.model,
            'capabilities': [asdict(cap) for cap in self.capabilities],
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.success_rate,
            'health_score': self.health_score,
            'current_load': self.current_load,
            'average_response_time': self.average_response_time,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }


@dataclass
class Task:
    """Enhanced task with priority and dependencies"""
    id: str
    description: str
    status: TaskStatus
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_for: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    required_capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to be executed (dependencies met)"""
        return self.status == TaskStatus.PENDING and (not self.scheduled_for or self.scheduled_for <= datetime.now())
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.value,
            'assigned_agent_id': self.assigned_agent_id,
            'created_at': self.created_at.isoformat(),
            'scheduled_for': self.scheduled_for.isoformat() if self.scheduled_for else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error,
            'dependencies': self.dependencies,
            'required_capabilities': self.required_capabilities,
            'metadata': self.metadata,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds
        }


# ============================================================================
# ENHANCED DATABASE LAYER WITH CONNECTION POOLING
# ============================================================================

class DatabasePool:
    """Async database connection pool"""
    
    def __init__(self, db_path: str, config: SystemConfig.DatabaseConfig):
        self.db_path = Path(db_path)
        self.config = config
        self.pool: List[aiosqlite.Connection] = []
        self.pool_lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__ + '.DatabasePool')
        
    async def initialize(self):
        """Initialize the database pool"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create initial connections
        for _ in range(min(5, self.config.max_connections)):
            conn = await self._create_connection()
            self.pool.append(conn)
        
        # Initialize schema
        async with self.get_connection() as conn:
            await self._init_schema(conn)
        
        self.logger.info(f"Database pool initialized with {len(self.pool)} connections")
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection"""
        conn = await aiosqlite.connect(str(self.db_path), timeout=self.config.connection_timeout)
        
        # Apply pragma settings
        for pragma, value in self.config.pragma_settings.items():
            if isinstance(value, bool):
                value = "ON" if value else "OFF"
            await conn.execute(f"PRAGMA {pragma} = {value}")
        
        return conn
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        async with self.pool_lock:
            if self.pool:
                conn = self.pool.pop()
            else:
                conn = await self._create_connection()
        
        try:
            yield conn
        finally:
            async with self.pool_lock:
                if len(self.pool) < self.config.max_connections:
                    self.pool.append(conn)
                else:
                    await conn.close()
    
    async def _init_schema(self, conn: aiosqlite.Connection):
        """Initialize enhanced database schema"""
        schema_queries = [
            """
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                description TEXT,
                model TEXT NOT NULL,
                capabilities TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                total_tasks INTEGER DEFAULT 0,
                successful_tasks INTEGER DEFAULT 0,
                failed_tasks INTEGER DEFAULT 0,
                health_score REAL DEFAULT 1.0,
                current_load INTEGER DEFAULT 0,
                max_concurrent_tasks INTEGER DEFAULT 5,
                average_response_time REAL DEFAULT 0.0,
                last_activity TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 2,
                assigned_agent_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scheduled_for TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT,
                error TEXT,
                dependencies TEXT,  -- JSON
                required_capabilities TEXT,  -- JSON
                metadata TEXT,  -- JSON
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                timeout_seconds INTEGER DEFAULT 300,
                FOREIGN KEY (assigned_agent_id) REFERENCES agents (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS task_metrics (
                task_id TEXT PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                processing_time REAL,
                tokens_used INTEGER DEFAULT 0,
                retries INTEGER DEFAULT 0,
                memory_peak REAL DEFAULT 0.0,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_id TEXT,
                task_id TEXT,
                data TEXT  -- JSON
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                agent_id TEXT,
                session_start TIMESTAMP,
                session_end TIMESTAMP,
                tasks_completed INTEGER DEFAULT 0
            )
            """
        ]
        
        for query in schema_queries:
            await conn.execute(query)
        
        # Create indexes separately
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_event_type ON system_events (event_type)",
            "CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent_id ON agent_sessions (agent_id, session_start)"
        ]
        
        for query in index_queries:
            await conn.execute(query)
        
        await conn.commit()
    
    async def close(self):
        """Close all connections in the pool"""
        async with self.pool_lock:
            for conn in self.pool:
                await conn.close()
            self.pool.clear()


class EnhancedDatabaseManager:
    """Enhanced database manager with better performance and features"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.pool = DatabasePool(config.database.path, config.database)
        self.logger = logging.getLogger(__name__ + '.DatabaseManager')
        self._cache = {}
        self._cache_ttl = {}
        
    async def initialize(self):
        """Initialize the database"""
        await self.pool.initialize()
    
    async def close(self):
        """Close database connections"""
        await self.pool.close()
    
    def _cache_key(self, table: str, key: str) -> str:
        """Generate cache key"""
        return f"{table}:{key}"
    
    def _is_cache_valid(self, cache_key: str, ttl_seconds: int = 60) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_ttl:
            return False
        return (datetime.now() - self._cache_ttl[cache_key]).total_seconds() < ttl_seconds
    
    async def save_agent(self, agent: Agent):
        """Save agent with caching"""
        async with self.pool.get_connection() as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO agents 
                (id, name, role, description, model, capabilities, created_at, is_active,
                 total_tasks, successful_tasks, failed_tasks, health_score, current_load,
                 max_concurrent_tasks, average_response_time, last_activity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent.id, agent.name, agent.role.value, agent.description, agent.model,
                json.dumps([asdict(cap) for cap in agent.capabilities]),
                agent.created_at, agent.is_active, agent.total_tasks, agent.successful_tasks,
                agent.failed_tasks, agent.health_score, agent.current_load,
                agent.max_concurrent_tasks, agent.average_response_time, agent.last_activity
            ))
            await conn.commit()
        
        # Update cache
        cache_key = self._cache_key("agents", agent.id)
        self._cache[cache_key] = agent
        self._cache_ttl[cache_key] = datetime.now()
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent with caching"""
        cache_key = self._cache_key("agents", agent_id)
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        async with self.pool.get_connection() as conn:
            async with conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    agent = self._row_to_agent(row)
                    self._cache[cache_key] = agent
                    self._cache_ttl[cache_key] = datetime.now()
                    return agent
        
        return None
    
    async def list_agents(self, active_only: bool = True, role: Optional[AgentRole] = None) -> List[Agent]:
        """List agents with filtering"""
        query = "SELECT * FROM agents WHERE 1=1"
        params = []
        
        if active_only:
            query += " AND is_active = 1"
        
        if role:
            query += " AND role = ?"
            params.append(role.value)
        
        query += " ORDER BY health_score DESC, success_rate DESC"
        
        agents = []
        async with self.pool.get_connection() as conn:
            async with conn.execute(query, params) as cursor:
                async for row in cursor:
                    agents.append(self._row_to_agent(row))
        
        return agents
    
    def _row_to_agent(self, row) -> Agent:
        """Convert database row to Agent object"""
        capabilities = []
        if row[5]:  # capabilities JSON
            try:
                cap_data = json.loads(row[5])
                capabilities = [AgentCapability(**cap) for cap in cap_data]
            except (json.JSONDecodeError, TypeError):
                pass
        
        return Agent(
            id=row[0],
            name=row[1],
            role=AgentRole(row[2]),
            description=row[3],
            model=row[4],
            capabilities=capabilities,
            created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
            is_active=bool(row[7]),
            total_tasks=row[8],
            successful_tasks=row[9],
            failed_tasks=row[10],
            health_score=row[11],
            current_load=row[12],
            max_concurrent_tasks=row[13],
            average_response_time=row[14],
            last_activity=datetime.fromisoformat(row[15]) if row[15] else None
        )
    
    async def save_task(self, task: Task):
        """Save task and metrics"""
        async with self.pool.get_connection() as conn:
            # Save main task
            await conn.execute("""
                INSERT OR REPLACE INTO tasks
                (id, description, status, priority, assigned_agent_id, created_at,
                 scheduled_for, started_at, completed_at, result, error, dependencies,
                 required_capabilities, metadata, retry_count, max_retries, timeout_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id, task.description, task.status.value, task.priority.value,
                task.assigned_agent_id, task.created_at, task.scheduled_for,
                task.started_at, task.completed_at, task.result, task.error,
                json.dumps(task.dependencies), json.dumps(task.required_capabilities),
                json.dumps(task.metadata), task.retry_count, task.max_retries, task.timeout_seconds
            ))
            
            # Save metrics if available
            if task.metrics:
                await conn.execute("""
                    INSERT OR REPLACE INTO task_metrics
                    (task_id, start_time, end_time, processing_time, tokens_used, retries, memory_peak)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id, task.metrics.start_time, task.metrics.end_time,
                    task.metrics.processing_time, task.metrics.tokens_used,
                    task.metrics.retries, task.metrics.memory_peak
                ))
            
            await conn.commit()
    
    async def get_ready_tasks(self, limit: int = 10) -> List[Task]:
        """Get tasks ready for processing, ordered by priority"""
        tasks = []
        async with self.pool.get_connection() as conn:
            query = """
                SELECT * FROM tasks 
                WHERE status = 'pending' 
                AND (scheduled_for IS NULL OR scheduled_for <= datetime('now'))
                ORDER BY priority DESC, created_at ASC 
                LIMIT ?
            """
            async with conn.execute(query, (limit,)) as cursor:
                async for row in cursor:
                    task = self._row_to_task(row)
                    if task.is_ready:
                        tasks.append(task)
        
        return tasks
    
    def _row_to_task(self, row) -> Task:
        """Convert database row to Task object"""
        return Task(
            id=row[0],
            description=row[1],
            status=TaskStatus(row[2]),
            priority=TaskPriority(row[3]),
            assigned_agent_id=row[4],
            created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
            scheduled_for=datetime.fromisoformat(row[6]) if row[6] else None,
            started_at=datetime.fromisoformat(row[7]) if row[7] else None,
            completed_at=datetime.fromisoformat(row[8]) if row[8] else None,
            result=row[9],
            error=row[10],
            dependencies=json.loads(row[11]) if row[11] else [],
            required_capabilities=json.loads(row[12]) if row[12] else [],
            metadata=json.loads(row[13]) if row[13] else {},
            retry_count=row[14],
            max_retries=row[15],
            timeout_seconds=row[16]
        )
    
    async def log_system_event(self, event_type: SystemEvent, agent_id: str = None, 
                             task_id: str = None, data: Dict[str, Any] = None):
        """Log system events for monitoring"""
        event_id = str(uuid.uuid4())
        async with self.pool.get_connection() as conn:
            await conn.execute("""
                INSERT INTO system_events (id, event_type, agent_id, task_id, data)
                VALUES (?, ?, ?, ?, ?)
            """, (event_id, event_type.name, agent_id, task_id, json.dumps(data) if data else None))
            await conn.commit()


# ============================================================================
# ENHANCED LLM PROVIDER WITH RETRY AND RATE LIMITING
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int, per: int = 60):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Try to acquire a token"""
        async with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            
            self.allowance += time_passed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                return False
            
            self.allowance -= 1.0
            return True


class EnhancedLLMProvider:
    """Enhanced LLM provider with advanced features"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.LLMProvider')
        self.rate_limiter = RateLimiter(config.models.rate_limit_per_minute)
        self.session = None
        
        # Response cache
        self.response_cache = {}
        self.cache_ttl = {}
        
        # Model-specific configurations
        self.model_configs = {
            'gpt-3.5-turbo': {'max_tokens': 4096, 'temperature': 0.7},
            'gpt-4': {'max_tokens': 8192, 'temperature': 0.7},
            'claude-3': {'max_tokens': 4000, 'temperature': 0.7}
        }
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.models.timeout),
            connector=aiohttp.TCPConnector(limit=100)
        )
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, messages: List[Dict], model: str) -> str:
        """Generate cache key for response"""
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.md5(content.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              model: Optional[str] = None,
                              agent_id: Optional[str] = None,
                              max_tokens: int = None,
                              temperature: float = None) -> Dict[str, Any]:
        """Generate response with caching and retry logic"""
        model = model or self.config.models.default
        
        # Validate model
        if model not in self.config.security.allowed_models:
            raise ValueError(f"Model {model} not allowed")
        
        # Check cache
        cache_key = self._get_cache_key(messages, model)
        if cache_key in self.response_cache and self._is_cache_valid(cache_key):
            self.logger.debug(f"Cache hit for model {model}")
            return self.response_cache[cache_key]
        
        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise Exception("Rate limit exceeded")
        
        try:
            # Try real API if available, otherwise use mock
            if model.startswith('gpt-') and os.getenv('OPENAI_API_KEY'):
                response = await self._call_openai(messages, model, max_tokens, temperature)
            elif model.startswith('claude') and os.getenv('ANTHROPIC_API_KEY'):
                response = await self._call_anthropic(messages, model, max_tokens, temperature)
            else:
                # Use enhanced mock response
                response = await self._enhanced_mock_response(messages, model, agent_id)
            
            # Cache successful responses
            self.response_cache[cache_key] = response
            self.cache_ttl[cache_key] = datetime.now()
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            # Fallback to mock on error
            return await self._enhanced_mock_response(messages, model, agent_id)
    
    def _is_cache_valid(self, cache_key: str, ttl_minutes: int = 30) -> bool:
        """Check if cached response is still valid"""
        if cache_key not in self.cache_ttl:
            return False
        age = (datetime.now() - self.cache_ttl[cache_key]).total_seconds() / 60
        return age < ttl_minutes
    
    async def _call_openai(self, messages: List[Dict], model: str, max_tokens: int, temperature: float) -> Dict:
        """Call OpenAI API with proper error handling"""
        headers = {
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
            'Content-Type': 'application/json'
        }
        
        config = self.model_configs.get(model, {})
        payload = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens or config.get('max_tokens', self.config.models.max_tokens),
            'temperature': temperature or config.get('temperature', self.config.models.temperature)
        }
        
        async with self.session.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    'content': data['choices'][0]['message']['content'],
                    'model': model,
                    'usage': data.get('usage', {})
                }
            else:
                error_text = await response.text()
                raise Exception(f"OpenAI API error {response.status}: {error_text}")
    
    async def _call_anthropic(self, messages: List[Dict], model: str, max_tokens: int, temperature: float) -> Dict:
        """Call Anthropic API with proper error handling"""
        headers = {
            'x-api-key': os.getenv('ANTHROPIC_API_KEY'),
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        # Convert messages format for Anthropic
        system_message = ""
        human_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                human_messages.append(msg)
        
        payload = {
            'model': model,
            'messages': human_messages,
            'max_tokens': max_tokens or self.config.models.max_tokens,
            'temperature': temperature or self.config.models.temperature
        }
        
        if system_message:
            payload['system'] = system_message
        
        async with self.session.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    'content': data['content'][0]['text'],
                    'model': model,
                    'usage': data.get('usage', {})
                }
            else:
                error_text = await response.text()
                raise Exception(f"Anthropic API error {response.status}: {error_text}")
    
    async def _enhanced_mock_response(self, messages: List[Dict], model: str, agent_id: str = None) -> Dict:
        """Generate enhanced context-aware mock responses"""
        # Simulate realistic processing time
        await asyncio.sleep(0.8 + (len(str(messages)) / 1000))
        
        user_message = ""
        system_context = ""
        
        for msg in messages:
            if msg['role'] == 'user':
                user_message = msg['content']
            elif msg['role'] == 'system':
                system_context = msg['content']
        
        user_lower = user_message.lower()
        
        # Advanced response generation based on context
        if 'python' in user_lower and ('code' in user_lower or 'implement' in user_lower):
            if 'machine learning' in user_lower or 'ml' in user_lower:
                response = self._generate_ml_code_response(user_message)
            elif 'web' in user_lower or 'api' in user_lower:
                response = self._generate_web_api_response(user_message)
            elif 'data' in user_lower and 'analysis' in user_lower:
                response = self._generate_data_analysis_response(user_message)
            else:
                response = self._generate_general_code_response(user_message)
        
        elif 'research' in user_lower:
            response = self._generate_research_response(user_message)
        
        elif 'analyze' in user_lower or 'analysis' in user_lower:
            response = self._generate_analysis_response(user_message)
        
        elif 'optimize' in user_lower or 'improve' in user_lower:
            response = self._generate_optimization_response(user_message)
        
        elif 'coordinate' in user_lower or 'manage' in user_lower:
            response = self._generate_coordination_response(user_message)
        
        else:
            response = self._generate_assistant_response(user_message, system_context)
        
        # Add agent-specific context if available
        if agent_id and 'specialist' in system_context.lower():
            response = f"[Specialist Analysis] {response}"
        elif agent_id and 'researcher' in system_context.lower():
            response = f"[Research Findings] {response}"
        
        return {
            'content': response,
            'model': model,
            'usage': {'total_tokens': len(response) + len(user_message)}
        }
    
    def _generate_ml_code_response(self, query: str) -> str:
        return """Here's a comprehensive machine learning solution:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLPipeline:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
    
    def load_data(self, file_path):
        \"\"\"Load and preprocess data\"\"\"
        self.data = pd.read_csv(file_path)
        print(f"Loaded {len(self.data)} samples with {len(self.data.columns)} features")
        return self.data
    
    def preprocess(self, target_column):
        \"\"\"Preprocess the data\"\"\"
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return self.X_train.shape, self.X_test.shape
    
    def train(self, **kwargs):
        \"\"\"Train the model\"\"\"
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                **kwargs
            )
        
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate(self):
        \"\"\"Evaluate model performance\"\"\"
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        predictions = self.model.predict(self.X_test)
        
        # Generate report
        report = classification_report(self.y_test, predictions)
        cm = confusion_matrix(self.y_test, predictions)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': self.model.score(self.X_test, self.y_test)
        }
    
    def plot_feature_importance(self, top_n=10):
        \"\"\"Plot feature importance\"\"\"
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    ml = MLPipeline('random_forest')
    
    # Load and preprocess data
    # ml.load_data('your_dataset.csv')
    # train_shape, test_shape = ml.preprocess('target_column')
    
    # Train model
    # ml.train(n_estimators=200, max_depth=10)
    
    # Evaluate
    # results = ml.evaluate()
    # print("Accuracy:", results['accuracy'])
    
    # Plot feature importance
    # ml.plot_feature_importance()
    
    print("ML Pipeline ready for your data!")
```

This pipeline provides a complete machine learning workflow with preprocessing, training, evaluation, and visualization capabilities."""
    
    def _generate_web_api_response(self, query: str) -> str:
        return """Here's a robust web API implementation:

```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import httpx
import redis
import json
from datetime import datetime, timedelta
import logging

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced API Service",
    description="Production-ready API with authentication, caching, and monitoring",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Models
class TaskRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=1000)
    priority: int = Field(default=2, ge=1, le=4)
    metadata: Optional[dict] = None

class TaskResponse(BaseModel):
    id: str
    description: str
    status: str
    created_at: datetime
    estimated_completion: Optional[datetime]

class APIResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # In production, verify against your auth service
    if token != "your-api-key":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

# Caching decorator
def cached(expiration: int = 300):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result, default=str))
            return result
        return wrapper
    return decorator

# Background task processor
async def process_task_background(task_id: str, description: str):
    \"\"\"Simulate background task processing\"\"\"
    await asyncio.sleep(5)  # Simulate work
    
    # Update task status in database/cache
    result = {
        "task_id": task_id,
        "status": "completed",
        "result": f"Processed: {description}",
        "completed_at": datetime.now()
    }
    
    redis_client.setex(f"task:{task_id}", 3600, json.dumps(result, default=str))
    logging.info(f"Task {task_id} completed")

# API Endpoints
@app.get("/", response_model=APIResponse)
async def health_check():
    return APIResponse(
        success=True,
        message="API is healthy and running",
        data={"version": "2.0.0", "uptime": "active"}
    )

@app.post("/tasks", response_model=APIResponse)
async def create_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    task_id = f"task_{int(datetime.now().timestamp())}"
    
    # Start background processing
    background_tasks.add_task(process_task_background, task_id, request.description)
    
    task_response = TaskResponse(
        id=task_id,
        description=request.description,
        status="processing",
        created_at=datetime.now(),
        estimated_completion=datetime.now() + timedelta(minutes=5)
    )
    
    return APIResponse(
        success=True,
        message="Task created successfully",
        data=task_response.dict()
    )

@app.get("/tasks/{task_id}", response_model=APIResponse)
@cached(expiration=60)
async def get_task_status(task_id: str, token: str = Depends(verify_token)):
    # Check cache for completed task
    cached_task = redis_client.get(f"task:{task_id}")
    
    if cached_task:
        task_data = json.loads(cached_task)
        return APIResponse(
            success=True,
            message="Task found",
            data=task_data
        )
    
    # Return processing status if not completed
    return APIResponse(
        success=True,
        message="Task is still processing",
        data={
            "task_id": task_id,
            "status": "processing",
            "progress": "In progress..."
        }
    )

@app.get("/metrics", response_model=APIResponse)
async def get_metrics(token: str = Depends(verify_token)):
    # In production, integrate with proper monitoring
    metrics = {
        "active_tasks": len(redis_client.keys("task:*")),
        "api_calls_today": 1250,
        "average_response_time": "0.3s",
        "error_rate": "0.1%"
    }
    
    return APIResponse(
        success=True,
        message="Metrics retrieved",
        data=metrics
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return APIResponse(
        success=False,
        message=str(exc.detail),
        data={"status_code": exc.status_code}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

This API includes authentication, caching, background processing, error handling, and monitoring capabilities for production use."""
    
    def _generate_data_analysis_response(self, query: str) -> str:
        return """Here's a comprehensive data analysis solution:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, data_path=None, df=None):
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Provide either data_path or DataFrame")
        
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Numeric columns: {len(self.numeric_cols)}")
        print(f"Categorical columns: {len(self.categorical_cols)}")
    
    def data_overview(self):
        \"\"\"Generate comprehensive data overview\"\"\"
        overview = {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'dtypes': self.df.dtypes.value_counts().to_dict()
        }
        
        # Missing values by column
        missing_by_col = self.df.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
        
        print("=" * 50)
        print("DATA OVERVIEW")
        print("=" * 50)
        print(f"Shape: {overview['shape']}")
        print(f"Memory usage: {overview['memory_usage']:.2f} MB")
        print(f"Missing values: {overview['missing_values']}")
        print(f"Duplicate rows: {overview['duplicate_rows']}")
        
        if not missing_by_col.empty:
            print("\\nMissing values by column:")
            for col, count in missing_by_col.items():
                pct = (count / len(self.df)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        
        return overview
    
    def statistical_summary(self):
        \"\"\"Generate statistical summary\"\"\"
        print("\\n" + "=" * 50)
        print("STATISTICAL SUMMARY")
        print("=" * 50)
        
        # Numeric columns summary
        if self.numeric_cols:
            print("\\nNumeric columns:")
            desc = self.df[self.numeric_cols].describe()
            print(desc.round(3))
            
            # Additional statistics
            print("\\nAdditional statistics:")
            for col in self.numeric_cols:
                skewness = stats.skew(self.df[col].dropna())
                kurtosis = stats.kurtosis(self.df[col].dropna())
                print(f"  {col}: skew={skewness:.3f}, kurtosis={kurtosis:.3f}")
        
        # Categorical columns summary
        if self.categorical_cols:
            print("\\nCategorical columns:")
            for col in self.categorical_cols:
                unique_count = self.df[col].nunique()
                most_common = self.df[col].value_counts().iloc[0]
                most_common_pct = (most_common / len(self.df)) * 100
                print(f"  {col}: {unique_count} unique values, "
                      f"most common: {most_common} ({most_common_pct:.1f}%)")
    
    def correlation_analysis(self, method='pearson', threshold=0.5):
        \"\"\"Analyze correlations between numeric variables\"\"\"
        if len(self.numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis")
            return None
        
        corr_matrix = self.df[self.numeric_cols].corr(method=method)
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        print(f"\\nStrong correlations (|r| >= {threshold}):")
        for corr in sorted(strong_corr, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  {corr['var1']} <-> {corr['var2']}: {corr['correlation']:.3f}")
        
        return corr_matrix
    
    def create_visualizations(self):
        \"\"\"Create comprehensive visualizations\"\"\"
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution Overview', 'Correlation Heatmap', 
                          'Missing Values', 'Box Plots'),
            specs=[[{"secondary_y": True}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # 1. Distribution overview for first numeric column
        if self.numeric_cols:
            col = self.numeric_cols[0]
            fig.add_trace(
                go.Histogram(x=self.df[col], name=f'{col} Distribution'),
                row=1, col=1
            )
        
        # 2. Correlation heatmap
        if len(self.numeric_cols) >= 2:
            corr_matrix = self.df[self.numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, 
                          x=corr_matrix.columns, 
                          y=corr_matrix.columns,
                          colorscale='RdBu',
                          name='Correlation'),
                row=1, col=2
            )
        
        # 3. Missing values
        missing_counts = self.df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            fig.add_trace(
                go.Bar(x=missing_counts.index, y=missing_counts.values,
                      name='Missing Values'),
                row=2, col=1
            )
        
        # 4. Box plots for numeric columns
        if self.numeric_cols:
            for col in self.numeric_cols[:3]:  # Limit to first 3 columns
                fig.add_trace(
                    go.Box(y=self.df[col], name=col),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Comprehensive Data Analysis Dashboard")
        fig.show()
    
    def outlier_detection(self, method='iqr'):
        \"\"\"Detect outliers using IQR or Z-score method\"\"\"
        outliers_summary = {}
        
        for col in self.numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = self.df[z_scores > 3]
            
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'indices': outliers.index.tolist()
            }
        
        print(f"\\nOutlier detection using {method.upper()} method:")
        for col, info in outliers_summary.items():
            if info['count'] > 0:
                print(f"  {col}: {info['count']} outliers ({info['percentage']:.1f}%)")
        
        return outliers_summary
    
    def generate_insights(self):
        \"\"\"Generate automated insights\"\"\"
        insights = []
        
        # Data quality insights
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct > 10:
            insights.append(f"⚠️  High missing data: {missing_pct:.1f}% of values are missing")
        
        # Skewness insights
        for col in self.numeric_cols:
            skew_val = stats.skew(self.df[col].dropna())
            if abs(skew_val) > 1:
                skew_type = "right" if skew_val > 0 else "left"
                insights.append(f"📊 {col} is highly {skew_type}-skewed (skew={skew_val:.2f})")
        
        # Correlation insights
        if len(self.numeric_cols) >= 2:
            corr_matrix = self.df[self.numeric_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            for var1, var2, corr in high_corr:
                insights.append(f"🔗 Strong correlation between {var1} and {var2} (r={corr:.3f})")
        
        print("\\n" + "=" * 50)
        print("AUTOMATED INSIGHTS")
        print("=" * 50)
        for insight in insights:
            print(insight)
        
        return insights

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    # analyzer = DataAnalyzer('your_dataset.csv')
    
    # Run comprehensive analysis
    # analyzer.data_overview()
    # analyzer.statistical_summary()
    # analyzer.correlation_analysis()
    # analyzer.outlier_detection()
    # analyzer.create_visualizations()
    # analyzer.generate_insights()
    
    print("Data analysis toolkit ready!")
```

This comprehensive data analysis toolkit provides statistical summaries, correlation analysis, outlier detection, visualizations, and automated insights generation."""
    
    def _generate_research_response(self, query: str) -> str:
        return f"""Research Analysis on: "{query[:100]}..."

**Executive Summary:**
Based on comprehensive research methodology, this analysis examines multiple dimensions of the query through systematic investigation.

**Key Findings:**
1. **Primary Context Analysis**: The request involves domain-specific research requiring interdisciplinary approach
2. **Methodology Applied**: Mixed-methods research combining quantitative data analysis and qualitative assessment
3. **Source Reliability**: Cross-referenced multiple authoritative sources and peer-reviewed materials

**Detailed Investigation:**
- **Literature Review**: Examined current academic consensus and emerging trends
- **Data Synthesis**: Integrated findings from primary and secondary sources
- **Gap Analysis**: Identified areas requiring further investigation
- **Expert Consultation**: Referenced domain specialist perspectives

**Recommendations:**
1. Implement phased approach for comprehensive understanding
2. Establish baseline metrics for measurable outcomes
3. Consider long-term implications and sustainability factors
4. Maintain iterative review process for emerging data

**Confidence Level**: High (85%) based on available evidence and source quality.

*This is a mock research response demonstrating comprehensive analytical approach.*"""
    
    def _generate_analysis_response(self, query: str) -> str:
        return f"""Analytical Report: {query[:80]}...

**Analysis Framework Applied:**
- Structured problem decomposition
- Multi-variable correlation assessment  
- Trend identification and pattern recognition
- Risk-benefit evaluation matrix

**Key Metrics Evaluated:**
- Performance indicators: Baseline vs. current state
- Efficiency ratios: Resource utilization optimization
- Quality benchmarks: Standards compliance assessment
- Timeline factors: Critical path dependencies

**Findings Summary:**
1. **Quantitative Analysis**: Statistical significance identified in key variables
2. **Qualitative Assessment**: Stakeholder impact evaluation completed
3. **Comparative Study**: Benchmarked against industry standards
4. **Predictive Modeling**: Future scenario projections developed

**Risk Assessment:**
- Low risk factors: Well-established procedures and proven methodologies
- Medium risk factors: External dependencies requiring monitoring
- High risk factors: Resource constraints and timeline pressures

**Strategic Recommendations:**
- Prioritize high-impact, low-risk initiatives
- Establish monitoring dashboard for key performance indicators
- Implement feedback loops for continuous improvement
- Develop contingency plans for identified risk scenarios

*This is a mock analysis demonstrating systematic analytical methodology.*"""
    
    def _generate_optimization_response(self, query: str) -> str:
        return f"""Optimization Strategy for: {query[:80]}...

**Current State Assessment:**
- Performance baseline established through comprehensive metrics analysis
- Resource utilization mapping completed across all operational dimensions
- Bottleneck identification using constraint theory principles

**Optimization Opportunities Identified:**
1. **Process Efficiency**: 23% improvement potential through workflow restructuring
2. **Resource Allocation**: 18% cost reduction via optimized distribution strategies
3. **Performance Enhancement**: 31% speed improvement through parallel processing
4. **Quality Assurance**: 15% defect reduction via enhanced validation protocols

**Implementation Framework:**
- Phase 1: Quick wins and low-hanging fruit (0-30 days)
- Phase 2: Structural improvements and system upgrades (30-90 days)  
- Phase 3: Advanced optimization and continuous improvement (90+ days)

**Technical Recommendations:**
- Implement automated monitoring and alerting systems
- Deploy machine learning algorithms for predictive optimization
- Establish feedback loops for real-time performance adjustment
- Create scalable architecture supporting future growth

**Expected Outcomes:**
- 25-35% overall performance improvement
- 20-30% reduction in operational costs
- 40-50% decrease in processing time
- 95%+ reliability and uptime achievement

**Monitoring & KPIs:**
- Real-time dashboard with key performance indicators
- Automated reporting and trend analysis
- Regular optimization review cycles
- Stakeholder feedback integration

*This represents a systematic optimization approach with measurable outcomes.*"""
    
    def _generate_coordination_response(self, query: str) -> str:
        return f"""Coordination Plan for: {query[:80]}...

**Project Structure & Governance:**
- Established clear hierarchy with defined roles and responsibilities
- Created communication protocols and escalation procedures
- Implemented project management framework with milestone tracking

**Stakeholder Management:**
- Primary stakeholders: Decision makers and resource owners
- Secondary stakeholders: End users and support teams
- External stakeholders: Vendors, partners, and regulatory bodies

**Resource Coordination:**
- Human resources: Team allocation and skill mapping completed
- Technical resources: Infrastructure and tool requirements identified
- Financial resources: Budget allocation and cost center assignment

**Timeline & Dependencies:**
- Critical path analysis identifies 4 major milestones
- Cross-functional dependencies mapped and managed
- Risk mitigation strategies for potential delays established

**Communication Framework:**
- Daily standup meetings for active workstreams
- Weekly stakeholder updates and progress reports
- Monthly steering committee reviews and strategic alignment

**Quality Assurance:**
- Defined acceptance criteria for all deliverables
- Regular checkpoint reviews and course correction protocols
- Continuous improvement feedback integration

**Success Metrics:**
- On-time delivery: Target 95% milestone adherence
- Budget compliance: Within 5% of approved budget
- Quality standards: Zero critical defects at completion
- Stakeholder satisfaction: 90%+ positive feedback scores

*This coordination approach ensures systematic project execution with clear accountability.*"""
    
    def _generate_assistant_response(self, query: str, system_context: str) -> str:
        return f"""I understand you're asking about: "{query[:100]}..."

Based on the context and requirements, here's my comprehensive response:

**Analysis of Your Request:**
Your query involves multiple considerations that I can help address through a structured approach. The key elements I've identified include the primary objective, potential challenges, and optimal solutions.

**Recommended Approach:**
1. **Initial Assessment**: Understanding the current situation and specific requirements
2. **Solution Design**: Developing a tailored approach that addresses your needs
3. **Implementation Strategy**: Creating actionable steps with clear timelines
4. **Monitoring & Adjustment**: Establishing feedback mechanisms for continuous improvement

**Key Considerations:**
- Resource requirements and constraints
- Timeline expectations and critical deadlines
- Quality standards and success criteria
- Risk factors and mitigation strategies

**Next Steps:**
I recommend we focus on the most critical aspects first, establish clear priorities, and create a structured plan that can be adapted as we learn more about specific requirements.

Would you like me to dive deeper into any particular aspect of this response, or would you prefer to explore alternative approaches to address your needs?

*This response demonstrates my general assistant capabilities while maintaining flexibility for various request types.*"""


# ============================================================================
# ENHANCED AGENT SYSTEM WITH INTELLIGENT ROUTING
# ============================================================================

class TaskRouter:
    """Intelligent task routing system"""
    
    def __init__(self, db: EnhancedDatabaseManager):
        self.db = db
        self.logger = logging.getLogger(__name__ + '.TaskRouter')
        self.routing_cache = {}
    
    async def route_task(self, task: Task) -> Optional[str]:
        """Route task to most suitable agent"""
        agents = await self.db.list_agents(active_only=True)
        
        if not agents:
            self.logger.warning("No active agents available")
            return None
        
        # Calculate suitability scores
        agent_scores = []
        for agent in agents:
            score = agent.can_handle_task(task.description, task.required_capabilities)
            if score > 0:
                agent_scores.append((agent, score))
        
        if not agent_scores:
            # Fallback to general assistant
            general_agents = [a for a in agents if a.role == AgentRole.ASSISTANT]
            if general_agents:
                return sorted(general_agents, key=lambda a: a.current_load)[0].id
            return None
        
        # Sort by score and availability
        agent_scores.sort(key=lambda x: (x[1], -x[0].current_load), reverse=True)
        selected_agent = agent_scores[0][0]
        
        self.logger.info(f"Routed task {task.id} to agent {selected_agent.name} (score: {agent_scores[0][1]:.3f})")
        return selected_agent.id


class SystemMonitor:
    """System performance and health monitor"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.SystemMonitor')
        self.metrics = {
            'tasks_processed': 0,
            'agents_active': 0,
            'average_response_time': 0.0,
            'error_count': 0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
        self.alerts = deque(maxlen=100)
        self._monitoring = False
        
    async def start_monitoring(self):
        """Start system monitoring"""
        self._monitoring = True
        asyncio.create_task(self._monitor_loop())
        self.logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self._monitoring = False
        self.logger.info("System monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self):
        """Collect system metrics"""
        try:
            # System resources
            self.metrics['cpu_usage'] = psutil.cpu_percent()
            self.metrics['memory_usage'] = psutil.virtual_memory().percent
            
            # Update timestamp
            self.metrics['last_updated'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        thresholds = self.config.monitoring.alert_thresholds
        
        # CPU usage alert
        if self.metrics['cpu_usage'] > thresholds['cpu_usage']:
            await self._trigger_alert(
                'HIGH_CPU_USAGE',
                f"CPU usage: {self.metrics['cpu_usage']:.1f}%"
            )
        
        # Memory usage alert
        if self.metrics['memory_usage'] > thresholds['memory_usage']:
            await self._trigger_alert(
                'HIGH_MEMORY_USAGE',
                f"Memory usage: {self.metrics['memory_usage']:.1f}%"
            )
    
    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger system alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'severity': 'WARNING'
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"ALERT: {alert_type} - {message}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        health_score = 100.0
        
        # Reduce score based on resource usage
        if self.metrics['cpu_usage'] > 80:
            health_score -= 20
        elif self.metrics['cpu_usage'] > 60:
            health_score -= 10
        
        if self.metrics['memory_usage'] > 85:
            health_score -= 25
        elif self.metrics['memory_usage'] > 70:
            health_score -= 15
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 70 else 'degraded' if health_score > 40 else 'critical',
            'metrics': self.metrics.copy(),
            'recent_alerts': list(self.alerts)[-10:]  # Last 10 alerts
        }


class EnhancedAgentSystem:
    """Enhanced agent system with advanced capabilities"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.db = EnhancedDatabaseManager(config)
        self.llm = EnhancedLLMProvider(config)
        self.router = TaskRouter(self.db)
        self.monitor = SystemMonitor(config)
        self.logger = logging.getLogger(__name__ + '.AgentSystem')
        
        # Task processing
        self.task_queue = asyncio.PriorityQueue()
        self.processing_tasks = {}
        self._shutdown_event = asyncio.Event()
        self._worker_tasks = []
        
        # Session management
        self.active_sessions = {}
        
    async def initialize(self):
        """Initialize all system components"""
        await self.db.initialize()
        await self.llm.initialize()
        await self.monitor.start_monitoring()
        
        # Create default agents
        await self._create_default_agents()
        
        # Start worker processes
        if self.config.processing.auto_process:
            await self._start_workers()
        
        self.logger.info("Enhanced agent system initialized")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        self.logger.info("Shutting down agent system...")
        
        # Stop workers
        self._shutdown_event.set()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Close components
        await self.monitor.stop_monitoring()
        await self.llm.close()
        await self.db.close()
        
        self.logger.info("Agent system shutdown complete")
    
    async def _create_default_agents(self):
        """Create enhanced default agents"""
        default_agents = [
            {
                'name': 'Advanced Assistant',
                'role': AgentRole.ASSISTANT,
                'description': 'Multi-purpose AI assistant with broad capabilities',
                'capabilities': [
                    AgentCapability('general_assistance', 9, 'General help and support', ['help', 'assist', 'support']),
                    AgentCapability('problem_solving', 8, 'Complex problem solving', ['solve', 'problem', 'issue']),
                    AgentCapability('communication', 9, 'Clear communication and explanation', ['explain', 'clarify', 'communicate'])
                ]
            },
            {
                'name': 'Research Specialist',
                'role': AgentRole.RESEARCHER,
                'description': 'Expert in research methodology and information analysis',
                'capabilities': [
                    AgentCapability('research', 10, 'Comprehensive research capabilities', ['research', 'investigate', 'study']),
                    AgentCapability('analysis', 9, 'Data and information analysis', ['analyze', 'examine', 'evaluate']),
                    AgentCapability('synthesis', 8, 'Information synthesis and summarization', ['synthesize', 'summarize', 'compile'])
                ]
            },
            {
                'name': 'Data Analyst',
                'role': AgentRole.ANALYST,
                'description': 'Specialized in statistical analysis and data interpretation',
                'capabilities': [
                    AgentCapability('statistics', 10, 'Statistical analysis and modeling', ['statistics', 'analysis', 'model']),
                    AgentCapability('visualization', 8, 'Data visualization and reporting', ['visualize', 'chart', 'graph']),
                    AgentCapability('interpretation', 9, 'Data interpretation and insights', ['interpret', 'insight', 'meaning'])
                ]
            },
            {
                'name': 'Code Specialist',
                'role': AgentRole.SPECIALIST,
                'description': 'Expert in software development and programming',
                'capabilities': [
                    AgentCapability('programming', 10, 'Software development and coding', ['code', 'program', 'develop']),
                    AgentCapability('debugging', 9, 'Code debugging and optimization', ['debug', 'fix', 'optimize']),
                    AgentCapability('architecture', 8, 'System architecture and design', ['architecture', 'design', 'structure'])
                ]
            },
            {
                'name': 'Quality Validator',
                'role': AgentRole.VALIDATOR,
                'description': 'Ensures quality and accuracy of outputs',
                'capabilities': [
                    AgentCapability('validation', 10, 'Quality assurance and validation', ['validate', 'verify', 'check']),
                    AgentCapability('testing', 9, 'Testing and quality control', ['test', 'quality', 'control']),
                    AgentCapability('review', 8, 'Content and process review', ['review', 'audit', 'assess'])
                ]
            },
            {
                'name': 'System Optimizer',
                'role': AgentRole.OPTIMIZER,
                'description': 'Focuses on performance optimization and improvement',
                'capabilities': [
                    AgentCapability('optimization', 10, 'Performance and process optimization', ['optimize', 'improve', 'enhance']),
                    AgentCapability('efficiency', 9, 'Efficiency analysis and improvement', ['efficiency', 'streamline', 'fast']),
                    AgentCapability('scalability', 8, 'Scalability and growth planning', ['scale', 'grow', 'expand'])
                ]
            }
        ]
        
        existing_agents = await self.db.list_agents()
        existing_names = {agent.name for agent in existing_agents}
        
        for agent_spec in default_agents:
            if agent_spec['name'] not in existing_names:
                agent = Agent(
                    id=str(uuid.uuid4()),
                    name=agent_spec['name'],
                    role=agent_spec['role'],
                    description=agent_spec['description'],
                    model=self.config.models.default,
                    capabilities=agent_spec['capabilities']
                )
                await self.db.save_agent(agent)
                await self.db.log_system_event(SystemEvent.AGENT_CREATED, agent_id=agent.id)
                self.logger.info(f"Created default agent: {agent.name}")
    
    async def _start_workers(self):
        """Start background worker tasks"""
        for i in range(self.config.processing.worker_count):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(worker_task)
        
        self.logger.info(f"Started {len(self._worker_tasks)} worker processes")
    
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing tasks"""
        self.logger.info(f"Worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get ready tasks from database
                ready_tasks = await self.db.get_ready_tasks(
                    limit=self.config.processing.batch_size
                )
                
                for task in ready_tasks:
                    if self._shutdown_event.is_set():
                        break
                    
                    # Check if task is already being processed
                    if task.id not in self.processing_tasks:
                        self.processing_tasks[task.id] = asyncio.create_task(
                            self._process_task(task, worker_id)
                        )
                
                # Clean up completed tasks
                completed_tasks = [
                    task_id for task_id, task_future in self.processing_tasks.items()
                    if task_future.done()
                ]
                
                for task_id in completed_tasks:
                    del self.processing_tasks[task_id]
                
                await asyncio.sleep(1)  # Brief pause before next iteration
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(5)
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: Task, worker_id: str):
        """Process a single task with comprehensive error handling"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Worker {worker_id} processing task {task.id}")
            
            # Update task status
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = start_time
            task.metrics.start_time = start_time
            await self.db.save_task(task)
            await self.db.log_system_event(SystemEvent.TASK_STARTED, task_id=task.id)
            
            # Route to appropriate agent
            agent_id = await self.router.route_task(task)
            if not agent_id:
                raise Exception("No suitable agent available")
            
            agent = await self.db.get_agent(agent_id)
            if not agent:
                raise Exception(f"Agent {agent_id} not found")
            
            # Update agent load
            agent.current_load += 1
            await self.db.save_agent(agent)
            
            task.assigned_agent_id = agent_id
            await self.db.save_task(task)
            
            # Generate response with timeout
            try:
                messages = [
                    {'role': 'system', 'content': f"You are {agent.name}, {agent.description}"},
                    {'role': 'user', 'content': task.description}
                ]
                
                response = await asyncio.wait_for(
                    self.llm.generate_response(messages, agent.model, agent.id),
                    timeout=task.timeout_seconds
                )
                
                # Validate response if validator agent exists
                if agent.role != AgentRole.VALIDATOR and self.config.processing.priority_queue_enabled:
                    response = await self._validate_response(response, task)
                
                # Update task with successful result
                task.status = TaskStatus.COMPLETED
                task.result = response['content']
                task.completed_at = datetime.now()
                task.metrics.end_time = task.completed_at
                task.metrics.tokens_used = response.get('usage', {}).get('total_tokens', 0)
                task.metrics.processing_time = (task.completed_at - start_time).total_seconds()
                
                # Update agent statistics
                agent.total_tasks += 1
                agent.successful_tasks += 1
                agent.last_activity = datetime.now()
                
                # Update average response time
                if agent.total_tasks > 1:
                    agent.average_response_time = (
                        (agent.average_response_time * (agent.total_tasks - 1) + 
                         task.metrics.processing_time) / agent.total_tasks
                    )
                else:
                    agent.average_response_time = task.metrics.processing_time
                
                await self.db.log_system_event(SystemEvent.TASK_COMPLETED, agent_id=agent.id, task_id=task.id)
                self.logger.info(f"Task {task.id} completed successfully by {agent.name}")
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.TIMEOUT
                task.error = "Task execution timeout"
                agent.total_tasks += 1
                self.logger.warning(f"Task {task.id} timed out")
            
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.retry_count += 1
                agent.total_tasks += 1
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.status = TaskStatus.PENDING
                    task.assigned_agent_id = None
                    task.error = None
                    self.logger.info(f"Retrying task {task.id} (attempt {task.retry_count + 1})")
                else:
                    await self.db.log_system_event(SystemEvent.TASK_FAILED, agent_id=agent.id, task_id=task.id)
                    self.logger.error(f"Task {task.id} failed permanently: {e}")
            
            finally:
                # Reduce agent load
                agent.current_load = max(0, agent.current_load - 1)
                
                # Update agent health score based on recent performance
                if agent.total_tasks > 0:
                    success_rate = agent.successful_tasks / agent.total_tasks
                    agent.health_score = min(1.0, success_rate + 0.1)  # Slight boost for participation
                
                await self.db.save_agent(agent)
                await self.db.save_task(task)
        
        except Exception as e:
            self.logger.error(f"Critical error processing task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = f"System error: {str(e)}"
            await self.db.save_task(task)
    
    async def _validate_response(self, response: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Validate response using validator agent"""
        try:
            validators = await self.db.list_agents(role=AgentRole.VALIDATOR)
            if not validators or not validators[0].is_available:
                return response  # Skip validation if no validator available
            
            validator = validators[0]
            validation_messages = [
                {
                    'role': 'system', 
                    'content': f"You are {validator.name}. Validate the following response for quality, accuracy, and completeness."
                },
                {
                    'role': 'user',
                    'content': f"Original task: {task.description}\n\nResponse to validate: {response['content']}\n\nProvide validation feedback or approve."
                }
            ]
            
            validation_response = await self.llm.generate_response(
                validation_messages, validator.model, validator.id
            )
            
            # Simple validation logic - in production, this would be more sophisticated
            if 'approved' in validation_response['content'].lower():
                return response
            else:
                # Could implement response improvement logic here
                response['validation_notes'] = validation_response['content']
                return response
            
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            return response
    
    async def submit_task(self, description: str, priority: TaskPriority = TaskPriority.NORMAL,
                         required_capabilities: List[str] = None,
                         scheduled_for: datetime = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Submit a task with enhanced options"""
        task = Task(
            id=str(uuid.uuid4()),
            description=description,
            status=TaskStatus.PENDING,
            priority=priority,
            required_capabilities=required_capabilities or [],
            scheduled_for=scheduled_for,
            metadata=metadata or {}
        )
        
        await self.db.save_task(task)
        await self.db.log_system_event(SystemEvent.TASK_SUBMITTED, task_id=task.id)
        
        self.logger.info(f"Submitted task: {task.id} with priority {priority.name}")
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive task status"""
        task_data = await self.db.get_task(task_id)
        
        if not task_data:
            return None
        
        result = task_data.to_dict()
        
        # Add agent information if assigned
        if task_data.assigned_agent_id:
            agent = await self.db.get_agent(task_data.assigned_agent_id)
            if agent:
                result['assigned_agent'] = {
                    'name': agent.name,
                    'role': agent.role.value,
                    'health_score': agent.health_score
                }
        
        return result
    
    async def create_agent(self, name: str, role: AgentRole, description: str,
                          capabilities: List[AgentCapability] = None,
                          model: str = None) -> Agent:
        """Create an enhanced agent"""
        agent = Agent(
            id=str(uuid.uuid4()),
            name=name,
            role=role,
            description=description,
            model=model or self.config.models.default,
            capabilities=capabilities or []
        )
        
        await self.db.save_agent(agent)
        await self.db.log_system_event(SystemEvent.AGENT_CREATED, agent_id=agent.id)
        
        self.logger.info(f"Created agent: {agent.name}")
        return agent
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        agents = await self.db.list_agents()
        health = self.monitor.get_system_health()
        
        # Task statistics from database
        async with self.db.pool.get_connection() as conn:
            task_stats = {}
            for status in TaskStatus:
                async with conn.execute("SELECT COUNT(*) FROM tasks WHERE status = ?", (status.value,)) as cursor:
                    count = (await cursor.fetchone())[0]
                    task_stats[status.value] = count
            
            # Performance metrics
            async with conn.execute("""
                SELECT AVG(processing_time) as avg_time, 
                       SUM(tokens_used) as total_tokens
                FROM task_metrics 
                WHERE start_time > datetime('now', '-24 hours')
            """) as cursor:
                perf_row = await cursor.fetchone()
                avg_processing_time = perf_row[0] if perf_row[0] else 0
                total_tokens_24h = perf_row[1] if perf_row[1] else 0
        
        return {
            'system_health': health,
            'agents': {
                'total': len(agents),
                'active': len([a for a in agents if a.is_active]),
                'by_role': {role.value: len([a for a in agents if a.role == role]) for role in AgentRole}
            },
            'tasks': task_stats,
            'performance': {
                'average_processing_time': avg_processing_time,
                'total_tokens_24h': total_tokens_24h,
                'active_processing_tasks': len(self.processing_tasks)
            },
            'configuration': {
                'auto_processing': self.config.processing.auto_process,
                'worker_count': self.config.processing.worker_count,
                'max_agents': self.config.agents.max_count
            }
        }


# ============================================================================
# ENHANCED CLI WITH RICH INTERFACE
# ============================================================================

class EnhancedCLI:
    """Enhanced CLI with better formatting and features"""
    
    def __init__(self, agent_system: EnhancedAgentSystem):
        self.agent_system = agent_system
        self.commands = {
            'help': self._show_help,
            'status': self._show_status,
            'health': self._show_health,
            'agents': self._list_agents,
            'create-agent': self._create_agent,
            'submit-task': self._submit_task,
            'task': self._task_status,
            'tasks': self._list_tasks,
            'metrics': self._show_metrics,
            'processing': self._toggle_processing,
            'clear': self._clear_screen,
            'exit': self._exit
        }
        self.session_start = datetime.now()
    
    def _print_header(self, title: str, width: int = 60):
        """Print formatted header"""
        print("\n" + "=" * width)
        print(f" {title.center(width-2)} ")
        print("=" * width)
    
    def _print_section(self, title: str, width: int = 40):
        """Print section header"""
        print(f"\n{title}")
        print("-" * len(title))
    
    def _show_help(self, args: List[str]):
        """Show comprehensive help"""
        self._print_header("ENHANCED MULTI-AGENT SYSTEM - HELP")
        
        help_sections = {
            "System Commands": [
                ("help", "Show this help message"),
                ("status", "Show system status overview"),
                ("health", "Show detailed system health"),
                ("metrics", "Show performance metrics"),
                ("clear", "Clear screen"),
                ("exit", "Exit the CLI")
            ],
            "Agent Management": [
                ("agents", "List all agents with details"),
                ("create-agent", "Create a new agent interactively")
            ],
            "Task Management": [
                ("submit-task", "Submit a new task interactively"),
                ("task <id>", "Get detailed task status"),
                ("tasks", "List recent tasks")
            ],
            "System Control": [
                ("processing [on|off]", "Toggle background processing")
            ]
        }
        
        for section, commands in help_sections.items():
            self._print_section(section)
            for cmd, desc in commands:
                print(f"  {cmd:<20} - {desc}")
        
        print(f"\nSession uptime: {datetime.now() - self.session_start}")
    
    async def _show_status(self, args: List[str]):
        """Show enhanced system status"""
        try:
            self._print_header("SYSTEM STATUS")
            
            stats = await self.agent_system.get_system_stats()
            
            # System health
            health = stats['system_health']
            health_icon = {
                'healthy': '🟢',
                'degraded': '🟡', 
                'critical': '🔴'
            }.get(health['status'], '⚪')
            
            print(f"System Health: {health_icon} {health['status'].upper()} ({health['health_score']:.1f}/100)")
            
            # Agents
            agents = stats['agents']
            print(f"Agents: {agents['active']}/{agents['total']} active")
            
            for role, count in agents['by_role'].items():
                if count > 0:
                    print(f"  {role.title()}: {count}")
            
            # Tasks
            tasks = stats['tasks']
            total_tasks = sum(tasks.values())
            print(f"Tasks: {total_tasks} total")
            
            for status, count in tasks.items():
                if count > 0:
                    status_icon = {
                        'pending': '⏳',
                        'queued': '📋',
                        'in_progress': '🔄',
                        'completed': '✅',
                        'failed': '❌',
                        'cancelled': '🚫',
                        'timeout': '⏰'
                    }.get(status, '❓')
                    print(f"  {status_icon} {status.replace('_', ' ').title()}: {count}")
            
            # Performance
            perf = stats['performance']
            print(f"Performance:")
            print(f"  Average processing time: {perf['average_processing_time']:.2f}s")
            print(f"  Active processing tasks: {perf['active_processing_tasks']}")
            print(f"  Tokens processed (24h): {perf['total_tokens_24h']:,}")
            
            # Configuration
            config = stats['configuration']
            processing_status = "ON" if config['auto_processing'] else "OFF"
            print(f"Configuration:")
            print(f"  Auto-processing: {processing_status}")
            print(f"  Worker processes: {config['worker_count']}")
            print(f"  Max agents: {config['max_agents']}")
        
        except Exception as e:
            print(f"Error getting status: {e}")
    
    async def _show_health(self, args: List[str]):
        """Show detailed system health"""
        try:
            self._print_header("SYSTEM HEALTH MONITORING")
            
            stats = await self.agent_system.get_system_stats()
            health = stats['system_health']
            
            print(f"Overall Health Score: {health['health_score']:.1f}/100")
            print(f"System Status: {health['status'].upper()}")
            
            # Detailed metrics
            metrics = health['metrics']
            print(f"\nResource Usage:")
            print(f"  CPU Usage: {metrics['cpu_usage']:.1f}%")
            print(f"  Memory Usage: {metrics['memory_usage']:.1f}%")
            print(f"  Last Updated: {metrics.get('last_updated', 'Unknown')}")
            
            # Recent alerts
            alerts = health['recent_alerts']
            if alerts:
                print(f"\nRecent Alerts ({len(alerts)}):")
                for alert in alerts[-5:]:  # Show last 5 alerts
                    timestamp = alert['timestamp'].strftime("%H:%M:%S")
                    print(f"  {timestamp} - {alert['type']}: {alert['message']}")
            else:
                print("\nNo recent alerts")
        
        except Exception as e:
            print(f"Error getting health status: {e}")
    
    async def _list_agents(self, args: List[str]):
        """List all agents with enhanced details"""
        try:
            agents = await self.agent_system.db.list_agents()
            
            if not agents:
                print("No agents found.")
                return
            
            self._print_header(f"AGENTS ({len(agents)})")
            
            # Group by role
            agents_by_role = {}
            for agent in agents:
                role = agent.role.value
                if role not in agents_by_role:
                    agents_by_role[role] = []
                agents_by_role[role].append(agent)
            
            for role, role_agents in agents_by_role.items():
                self._print_section(f"{role.upper()} ({len(role_agents)})")
                
                for agent in role_agents:
                    status_icon = "🟢" if agent.is_active else "🔴"
                    health_bar = "█" * int(agent.health_score * 10) + "░" * (10 - int(agent.health_score * 10))
                    
                    print(f"  {status_icon} {agent.name}")
                    print(f"    ID: {agent.id}")
                    print(f"    Model: {agent.model}")
                    print(f"    Health: {health_bar} {agent.health_score:.2f}")
                    print(f"    Load: {agent.current_load}/{agent.max_concurrent_tasks}")
                    print(f"    Performance: {agent.total_tasks} tasks, {agent.success_rate:.1%} success")
                    print(f"    Avg Response: {agent.average_response_time:.2f}s")
                    
                    if agent.capabilities:
                        caps = ", ".join([f"{cap.name}({cap.level})" for cap in agent.capabilities[:3]])
                        print(f"    Capabilities: {caps}")
                    
                    if agent.last_activity:
                        time_since = datetime.now() - agent.last_activity
                        print(f"    Last Active: {time_since.total_seconds()/3600:.1f}h ago")
                    
                    print()
        
        except Exception as e:
            print(f"Error listing agents: {e}")
    
    async def _create_agent(self, args: List[str]):
        """Create a new agent interactively"""
        try:
            self._print_header("CREATE NEW AGENT")
            
            name = input("Agent Name: ").strip()
            if not name:
                print("Name cannot be empty")
                return
            
            print("\nAvailable roles:")
            for role in AgentRole:
                print(f"  {role.value}")
            
            role_input = input("Role: ").strip().lower()
            try:
                role = AgentRole(role_input)
            except ValueError:
                print("Invalid role")
                return
            
            description = input("Description: ").strip()
            if not description:
                print("Description cannot be empty")
                return
            
            model = input("Model (press Enter for default): ").strip()
            if not model:
                model = None
            
            # Capabilities
            capabilities = []
            print("\nAdd capabilities (press Enter to skip):")
            while True:
                cap_name = input("Capability name (or Enter to finish): ").strip()
                if not cap_name:
                    break
                
                try:
                    cap_level = int(input(f"Level for {cap_name} (1-10): "))
                    if not (1 <= cap_level <= 10):
                        print("Level must be between 1 and 10")
                        continue
                except ValueError:
                    print("Invalid level")
                    continue
                
                cap_desc = input("Description: ").strip()
                keywords = input("Keywords (comma-separated): ").strip().split(",")
                keywords = [k.strip() for k in keywords if k.strip()]
                
                capabilities.append(AgentCapability(cap_name, cap_level, cap_desc, keywords))
            
            agent = await self.agent_system.create_agent(name, role, description, capabilities, model)
            print(f"\n✅ Created agent: {agent.name} ({agent.id})")
            
        except Exception as e:
            print(f"Error creating agent: {e}")
    
    async def _submit_task(self, args: List[str]):
        """Submit a task interactively"""
        try:
            self._print_header("SUBMIT NEW TASK")
            
            description = input("Task description: ").strip()
            if not description:
                print("Description cannot be empty")
                return
            
            # Priority
            print("\nPriority levels:")
            for priority in TaskPriority:
                print(f"  {priority.value}: {priority.name}")
            
            priority_input = input("Priority (press Enter for normal): ").strip()
            if priority_input:
                try:
                    priority = TaskPriority(int(priority_input))
                except ValueError:
                    priority = TaskPriority.NORMAL
            else:
                priority = TaskPriority.NORMAL
            
            # Required capabilities
            capabilities = input("Required capabilities (comma-separated, optional): ").strip()
            required_capabilities = []
            if capabilities:
                required_capabilities = [cap.strip() for cap in capabilities.split(",") if cap.strip()]
            
            # Schedule for later
            schedule_input = input("Schedule for later? (y/n): ").strip().lower()
            scheduled_for = None
            if schedule_input == 'y':
                try:
                    hours = float(input("Hours from now: "))
                    scheduled_for = datetime.now() + timedelta(hours=hours)
                except ValueError:
                    print("Invalid time, task will be processed immediately")
            
            task_id = await self.agent_system.submit_task(
                description=description,
                priority=priority,
                required_capabilities=required_capabilities,
                scheduled_for=scheduled_for
            )
            
            print(f"\n✅ Submitted task: {task_id}")
            print(f"Priority: {priority.name}")
            
            if scheduled_for:
                print(f"Scheduled for: {scheduled_for.strftime('%Y-%m-%d %H:%M:%S')}")
            elif self.agent_system.config.processing.auto_process:
                print("Task will be processed automatically in the background.")
            else:
                print("Background processing is OFF. Use 'processing on' to enable it.")
        
        except Exception as e:
            print(f"Error submitting task: {e}")
    
    async def _task_status(self, args: List[str]):
        """Get detailed task status"""
        if not args:
            print("Usage: task <task-id>")
            return
        
        task_id = args[0]
        try:
            task_data = await self.agent_system.get_task_status(task_id)
            
            if not task_data:
                print("Task not found")
                return
            
            self._print_header("TASK DETAILS")
            
            status_icons = {
                'pending': '⏳',
                'queued': '📋',
                'in_progress': '🔄',
                'validating': '🔍',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '🚫',
                'timeout': '⏰'
            }
            
            status_icon = status_icons.get(task_data['status'], '❓')
            
            print(f"ID: {task_data['id']}")
            print(f"Status: {status_icon} {task_data['status'].upper()}")
            print(f"Priority: {task_data['priority']}")
            print(f"Created: {task_data['created_at']}")
            
            if task_data.get('scheduled_for'):
                print(f"Scheduled: {task_data['scheduled_for']}")
            
            if task_data.get('assigned_agent'):
                agent = task_data['assigned_agent']
                print(f"Agent: {agent['name']} ({agent['role']}) - Health: {agent['health_score']:.2f}")
            
            if task_data.get('started_at'):
                print(f"Started: {task_data['started_at']}")
            
            if task_data.get('completed_at'):
                print(f"Completed: {task_data['completed_at']}")
                
                # Calculate duration
                if task_data.get('started_at'):
                    start = datetime.fromisoformat(task_data['started_at'])
                    end = datetime.fromisoformat(task_data['completed_at'])
                    duration = (end - start).total_seconds()
                    print(f"Duration: {duration:.2f}s")
            
            print(f"\nDescription:")
            print(f"  {task_data['description']}")
            
            if task_data.get('required_capabilities'):
                print(f"Required Capabilities: {', '.join(task_data['required_capabilities'])}")
            
            if task_data.get('retry_count', 0) > 0:
                print(f"Retries: {task_data['retry_count']}/{task_data.get('max_retries', 3)}")
            
            if task_data.get('result'):
                print(f"\nResult:")
                print(f"{'='*50}")
                print(f"{task_data['result']}")
                print(f"{'='*50}")
            
            if task_data.get('error'):
                print(f"\nError: {task_data['error']}")
        
        except Exception as e:
            print(f"Error getting task status: {e}")
    
    async def _list_tasks(self, args: List[str]):
        """List recent tasks with enhanced formatting"""
        try:
            limit = 15
            if args and args[0].isdigit():
                limit = int(args[0])
            
            # Get tasks from database
            async with self.agent_system.db.pool.get_connection() as conn:
                query = """
                    SELECT id, description, status, priority, created_at, completed_at, assigned_agent_id, error
                    FROM tasks 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                
                tasks = []
                async with conn.execute(query, (limit,)) as cursor:
                    async for row in cursor:
                        tasks.append(row)
            
            if not tasks:
                print("No tasks found.")
                return
            
            self._print_header(f"RECENT TASKS ({len(tasks)})")
            
            status_icons = {
                'pending': '⏳',
                'queued': '📋',
                'in_progress': '🔄',
                'validating': '🔍',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '🚫',
                'timeout': '⏰'
            }
            
            for row in tasks:
                task_id, desc, status, priority, created_at, completed_at, agent_id, error = row
                
                status_icon = status_icons.get(status, '❓')
                priority_indicator = "🔥" if priority >= 3 else "⭐" if priority == 2 else ""
                
                # Truncate description
                desc_short = desc[:60] + "..." if len(desc) > 60 else desc
                
                print(f"{status_icon} {priority_indicator} {desc_short}")
                print(f"   ID: {task_id}")
                print(f"   Status: {status.upper()}")
                print(f"   Created: {created_at}")
                
                if completed_at:
                    print(f"   Completed: {completed_at}")
                
                if agent_id:
                    # Get agent name
                    agent = await self.agent_system.db.get_agent(agent_id)
                    agent_name = agent.name if agent else "Unknown"
                    print(f"   Agent: {agent_name}")
                
                if error:
                    error_short = error[:50] + "..." if len(error) > 50 else error
                    print(f"   Error: {error_short}")
                
                print()
        
        except Exception as e:
            print(f"Error listing tasks: {e}")
    
    async def _show_metrics(self, args: List[str]):
        """Show detailed performance metrics"""
        try:
            self._print_header("PERFORMANCE METRICS")
            
            stats = await self.agent_system.get_system_stats()
            
            # System performance
            perf = stats['performance']
            print("Performance Overview:")
            print(f"  Average Processing Time: {perf['average_processing_time']:.2f}s")
            print(f"  Active Processing Tasks: {perf['active_processing_tasks']}")
            print(f"  Tokens Processed (24h): {perf['total_tokens_24h']:,}")
            
            # Resource usage
            health = stats['system_health']
            metrics = health['metrics']
            print(f"\nResource Usage:")
            print(f"  CPU: {metrics['cpu_usage']:.1f}%")
            print(f"  Memory: {metrics['memory_usage']:.1f}%")
            
            # Agent performance
            agents = await self.agent_system.db.list_agents()
            if agents:
                print(f"\nTop Performing Agents:")
                
                # Sort by success rate and task count
                top_agents = sorted(agents, 
                                  key=lambda a: (a.success_rate, a.total_tasks), 
                                  reverse=True)[:5]
                
                for agent in top_agents:
                    print(f"  {agent.name}: {agent.total_tasks} tasks, "
                          f"{agent.success_rate:.1%} success, "
                          f"{agent.average_response_time:.2f}s avg")
            
            # Task statistics
            tasks = stats['tasks']
            total_tasks = sum(tasks.values())
            if total_tasks > 0:
                print(f"\nTask Distribution:")
                for status, count in tasks.items():
                    if count > 0:
                        percentage = (count / total_tasks) * 100
                        print(f"  {status.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        except Exception as e:
            print(f"Error getting metrics: {e}")
    
    async def _toggle_processing(self, args: List[str]):
        """Toggle background processing"""
        try:
            current_state = self.agent_system.config.processing.auto_process
            
            if args and args[0].lower() in ['on', 'off']:
                new_state = args[0].lower() == 'on'
            else:
                new_state = not current_state
            
            self.agent_system.config.processing.auto_process = new_state
            
            if new_state and not self.agent_system._worker_tasks:
                await self.agent_system._start_workers()
                print("✅ Background processing enabled and started")
            elif not new_state:
                print("⚠️  Background processing disabled")
                print("   (Current tasks will complete, but no new tasks will be processed)")
            else:
                status = "enabled" if new_state else "disabled"
                print(f"✅ Background processing {status}")
        
        except Exception as e:
            print(f"Error toggling processing: {e}")
    
    def _clear_screen(self, args: List[str]):
        """Clear the screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _exit(self, args: List[str]):
        """Exit the CLI"""
        return False
    
    async def run(self):
        """Run the enhanced CLI"""
        self._print_header("ENHANCED MULTI-AGENT AI SYSTEM v2.0")
        print("Advanced production-ready multi-agent system with intelligent routing,")
        print("comprehensive monitoring, and enhanced task processing capabilities.")
        print("\nType 'help' for available commands")
        
        # Show initial status
        await self._show_status([])
        print()
        
        while True:
            try:
                line = input("🤖 agent-system> ").strip()
                if not line:
                    continue
                
                parts = line.split()
                command = parts[0]
                args = parts[1:]
                
                if command in self.commands:
                    if asyncio.iscoroutinefunction(self.commands[command]):
                        result = await self.commands[command](args)
                    else:
                        result = self.commands[command](args)
                    
                    if result is False:  # Exit command returns False
                        break
                else:
                    print(f"Unknown command: '{command}'. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


# ============================================================================
# ENHANCED APPLICATION CLASS
# ============================================================================

class EnhancedApplication:
    """Enhanced application with better lifecycle management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = SystemConfig.load_from_file(config_path)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.agent_system = EnhancedAgentSystem(self.config)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._shutdown_requested = False
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler with rotation
        file_handler = logging.FileHandler(log_dir / "system.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        console_handler.setFormatter(simple_formatter)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Reduce noise from third-party libraries
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        
        return logging.getLogger(__name__)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {sig}")
        self._shutdown_requested = True
        print(f"\nReceived shutdown signal {sig}. Initiating graceful shutdown...")
    
    async def run_async(self):
        """Run the application asynchronously"""
        try:
            self.logger.info("Starting Enhanced Multi-Agent AI System")
            
            # Initialize system
            await self.agent_system.initialize()
            
            # Run CLI
            cli = EnhancedCLI(self.agent_system)
            await cli.run()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise
        finally:
            # Graceful shutdown
            self.logger.info("Initiating system shutdown")
            await self.agent_system.shutdown()
            self.logger.info("System shutdown complete")
    
    def run(self):
        """Main entry point"""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"Fatal error: {e}")
            self.logger.error(f"Fatal error: {e}")


# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================

ENHANCED_CONFIG_YAML = """
# Enhanced Multi-Agent System Configuration

database:
  path: "data/system.db"
  max_connections: 20
  connection_timeout: 30
  pragma_settings:
    journal_mode: "WAL"
    synchronous: "NORMAL"
    cache_size: 10000
    foreign_keys: true
    temp_store: "memory"

models:
  default: "mock-model"  # Change to actual model if you have API keys
  timeout: 120
  max_tokens: 4000
  temperature: 0.7
  max_retries: 3
  rate_limit_per_minute: 60

agents:
  max_count: 100
  response_timeout: 300
  health_check_interval: 60
  max_concurrent_tasks: 10

security:
  api_key_length: 32
  session_timeout: 3600
  max_task_size: 1048576  # 1MB
  allowed_models:
    - "mock-model"
    - "gpt-3.5-turbo"
    - "gpt-4"
    - "claude-3"
  enable_audit_log: true

processing:
  auto_process: true
  worker_count: 4
  batch_size: 5
  priority_queue_enabled: true
  task_timeout: 600

monitoring:
  metrics_enabled: true
  performance_tracking: true
  alert_thresholds:
    cpu_usage: 80.0
    memory_usage: 85.0
    error_rate: 5.0
    response_time: 30.0
"""

def create_enhanced_config():
    """Create enhanced configuration file"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(ENHANCED_CONFIG_YAML)
        print(f"Created enhanced configuration: {config_path}")
    else:
        print("Configuration file already exists")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Enhanced main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Production Multi-Agent AI System v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_system.py                    # Run with default config
  python agent_system.py --config my.yaml  # Use custom config
  python agent_system.py --create-config   # Create default config file
        """
    )
    
    parser.add_argument("--config", default="config.yaml",
                       help="Configuration file path (default: config.yaml)")
    parser.add_argument("--create-config", action="store_true",
                       help="Create enhanced configuration file and exit")
    parser.add_argument("--version", action="version", version="Enhanced Multi-Agent System v2.0")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_enhanced_config()
        return
    
    # Ensure required directories exist
    for directory in ["data", "logs"]:
        Path(directory).mkdir(exist_ok=True)
    
    try:
        app = EnhancedApplication(args.config)
        app.run()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.error(f"Fatal application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
