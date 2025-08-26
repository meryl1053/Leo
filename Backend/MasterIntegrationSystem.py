#!/usr/bin/env python3
"""
🌟 MASTER INTEGRATION SYSTEM 🌟
Comprehensive integration hub that unifies ALL LEO AI components into a single,
cohesive, self-evolving artificial intelligence system.

This system brings together:
- UltimateAIOrchestrator (task coordination)
- AdvancedLearningEngine (continuous learning)
- IntelligentResourceOptimizer (resource management)
- MultimodalAIProcessor (cross-modal processing)
- All existing components (chatbot, search, voice, etc.)

Features:
- Unified API for all AI capabilities
- Real-time learning integration
- Intelligent resource management
- Cross-component communication
- Comprehensive monitoring and analytics
- Auto-scaling and optimization
"""

import asyncio
import logging
import os
import sys
import json
import time
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import traceback
import weakref
import hashlib
import psutil
import numpy as np
from collections import defaultdict, deque
import sqlite3

# Import our advanced components
try:
    from UltimateAIOrchestrator import UltimateAIOrchestrator, OrchestratorConfig, Task, TaskPriority, ProcessingMode, ModalityType
    from AdvancedLearningEngine import AdvancedLearningEngine, FeedbackType, UserInteraction
    from IntelligentResourceOptimizer import IntelligentResourceOptimizer, OptimizationStrategy
    from MultimodalAIProcessor import MultimodalAIProcessor, ModalityType as MultimodalModalityType
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Advanced components not available: {e}")
    ADVANCED_COMPONENTS_AVAILABLE = False

# Import existing LEO components
try:
    from UltraAdvancedAgentCreator import EnhancedAgentSystem
    from DataAnalyzer import UniversalAutonomousResearchAnalytics
    from RealtimeSearchEngine import RealtimeSearchEngine
    from Chatbot import ChatBot
    from Model import FirstLayerDMM
    from AutoUpdater import IntelligentAIUpdater
    from SpeechToText import SpeechRecognition
    from TextToSpeech import TextToSpeech
    from ImageGeneration import ImageGenerator
    from SoundTrigger import VoiceActivationSystem
    LEO_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LEO components not available: {e}")
    LEO_COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Integration status levels"""
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    OPTIMIZED = "optimized"
    LEARNING = "learning"
    ERROR = "error"


class SystemMode(Enum):
    """System operation modes"""
    VOICE_ASSISTANT = "voice_assistant"
    CHAT_INTERFACE = "chat_interface"
    DATA_ANALYSIS = "data_analysis"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    AUTONOMOUS_OPERATION = "autonomous_operation"


@dataclass
class ComponentStatus:
    """Status of individual components"""
    component_name: str
    status: str
    health_score: float
    last_used: datetime
    error_count: int
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    integration_level: str = "basic"


@dataclass
class SystemMetrics:
    """Overall system metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0
    user_satisfaction: float = 0.0
    learning_efficiency: float = 0.0
    resource_utilization: float = 0.0
    uptime_hours: float = 0.0
    active_users: int = 0


class MasterIntegrationSystem:
    """Master system that integrates all LEO AI components"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.system_id = str(uuid.uuid4())
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # System state
        self.status = IntegrationStatus.INITIALIZING
        self.current_mode = SystemMode.VOICE_ASSISTANT
        self.active_components = {}
        self.component_status = {}
        self.system_metrics = SystemMetrics()
        
        # Advanced components
        self.orchestrator = None
        self.learning_engine = None
        self.resource_optimizer = None
        self.multimodal_processor = None
        
        # LEO components
        self.agent_system = None
        self.data_analyzer = None
        self.search_engine = None
        self.chatbot = None
        self.voice_system = None
        self.image_generator = None
        self.updater = None
        
        # Integration layer
        self.task_queue = asyncio.Queue()
        self.response_cache = {}
        self.user_sessions = {}
        self.cross_component_memory = defaultdict(dict)
        
        # Monitoring and optimization
        self.performance_monitor = None
        self.integration_optimizer = None
        self.health_checker = None
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        self.logger.info(f"🌟 Master Integration System initialized - ID: {self.system_id}")
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            "system": {
                "max_concurrent_requests": 50,
                "default_timeout": 30.0,
                "enable_monitoring": True,
                "enable_learning": True,
                "enable_optimization": True
            },
            "components": {
                "orchestrator": {"enabled": True},
                "learning_engine": {"enabled": True},
                "resource_optimizer": {"enabled": True},
                "multimodal_processor": {"enabled": True}
            },
            "integration": {
                "cross_component_sharing": True,
                "unified_memory": True,
                "adaptive_routing": True,
                "real_time_optimization": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    async def initialize(self):
        """Initialize all components and establish integrations"""
        try:
            self.logger.info("🚀 Starting Master Integration System initialization...")
            
            # Step 1: Initialize advanced components
            await self._initialize_advanced_components()
            
            # Step 2: Initialize LEO components
            await self._initialize_leo_components()
            
            # Step 3: Establish cross-component integrations
            await self._establish_integrations()
            
            # Step 4: Start background monitoring and optimization
            await self._start_background_services()
            
            # Step 5: Verify system health
            await self._verify_system_health()
            
            self.status = IntegrationStatus.CONNECTED
            self.is_running = True
            
            self.logger.info("✅ Master Integration System fully initialized and operational!")
            
        except Exception as e:
            self.status = IntegrationStatus.ERROR
            self.logger.error(f"❌ Initialization failed: {e}")
            traceback.print_exc()
            raise
    
    async def _initialize_advanced_components(self):
        """Initialize advanced AI components"""
        try:
            if not ADVANCED_COMPONENTS_AVAILABLE:
                self.logger.warning("Advanced components not available")
                return
            
            # Initialize AI Orchestrator
            if self.config["components"]["orchestrator"]["enabled"]:
                orchestrator_config = OrchestratorConfig(
                    enable_real_time_learning=True,
                    enable_multimodal_fusion=True,
                    auto_scaling_enabled=True
                )
                self.orchestrator = UltimateAIOrchestrator(orchestrator_config)
                await self.orchestrator.initialize()
                self.active_components["orchestrator"] = self.orchestrator
                self.logger.info("✅ AI Orchestrator initialized")
            
            # Initialize Learning Engine
            if self.config["components"]["learning_engine"]["enabled"]:
                self.learning_engine = AdvancedLearningEngine("learning_data")
                learning_task = asyncio.create_task(self.learning_engine.start_learning_loop())
                self.background_tasks.append(learning_task)
                self.active_components["learning_engine"] = self.learning_engine
                self.logger.info("✅ Learning Engine initialized")
            
            # Initialize Resource Optimizer
            if self.config["components"]["resource_optimizer"]["enabled"]:
                self.resource_optimizer = IntelligentResourceOptimizer(OptimizationStrategy.BALANCED)
                optimizer_task = asyncio.create_task(self.resource_optimizer.start_optimization_loop())
                self.background_tasks.append(optimizer_task)
                self.active_components["resource_optimizer"] = self.resource_optimizer
                self.logger.info("✅ Resource Optimizer initialized")
            
            # Initialize Multimodal Processor
            if self.config["components"]["multimodal_processor"]["enabled"]:
                self.multimodal_processor = MultimodalAIProcessor()
                self.active_components["multimodal_processor"] = self.multimodal_processor
                self.logger.info("✅ Multimodal Processor initialized")
                
        except Exception as e:
            self.logger.error(f"Advanced component initialization failed: {e}")
            raise
    
    async def _initialize_leo_components(self):
        """Initialize existing LEO components"""
        try:
            if not LEO_COMPONENTS_AVAILABLE:
                self.logger.warning("LEO components not available")
                return
            
            # Initialize Agent System
            try:
                self.agent_system = EnhancedAgentSystem()
                self.active_components["agent_system"] = self.agent_system
                self.logger.info("✅ Agent System initialized")
            except Exception as e:
                self.logger.warning(f"Agent System initialization failed: {e}")
            
            # Initialize Data Analyzer
            try:
                self.data_analyzer = UniversalAutonomousResearchAnalytics()
                self.active_components["data_analyzer"] = self.data_analyzer
                self.logger.info("✅ Data Analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Data Analyzer initialization failed: {e}")
            
            # Initialize Search Engine
            try:
                self.search_engine = RealtimeSearchEngine()
                self.active_components["search_engine"] = self.search_engine
                self.logger.info("✅ Search Engine initialized")
            except Exception as e:
                self.logger.warning(f"Search Engine initialization failed: {e}")
            
            # Initialize Chatbot
            try:
                self.chatbot = ChatBot()
                self.active_components["chatbot"] = self.chatbot
                self.logger.info("✅ Chatbot initialized")
            except Exception as e:
                self.logger.warning(f"Chatbot initialization failed: {e}")
            
            # Initialize Voice System
            try:
                self.voice_system = {
                    "speech_recognition": SpeechRecognition(),
                    "text_to_speech": TextToSpeech(),
                    "voice_activation": VoiceActivationSystem()
                }
                self.active_components["voice_system"] = self.voice_system
                self.logger.info("✅ Voice System initialized")
            except Exception as e:
                self.logger.warning(f"Voice System initialization failed: {e}")
            
            # Initialize Image Generator
            try:
                self.image_generator = ImageGenerator()
                self.active_components["image_generator"] = self.image_generator
                self.logger.info("✅ Image Generator initialized")
            except Exception as e:
                self.logger.warning(f"Image Generator initialization failed: {e}")
            
            # Initialize Auto Updater
            try:
                self.updater = IntelligentAIUpdater()
                self.active_components["updater"] = self.updater
                self.logger.info("✅ Auto Updater initialized")
            except Exception as e:
                self.logger.warning(f"Auto Updater initialization failed: {e}")
                
        except Exception as e:
            self.logger.error(f"LEO component initialization failed: {e}")
            raise
    
    async def _establish_integrations(self):
        """Establish cross-component integrations"""
        try:
            self.logger.info("🔗 Establishing cross-component integrations...")
            
            # Connect Learning Engine to other components
            if self.learning_engine and self.orchestrator:
                # Set up learning feedback loop with orchestrator
                self.orchestrator.set_learning_callback(self._learning_feedback_callback)
            
            # Connect Resource Optimizer to components
            if self.resource_optimizer and self.orchestrator:
                # Set up resource optimization feedback
                self.orchestrator.set_resource_callback(self._resource_optimization_callback)
            
            # Connect Multimodal Processor to orchestrator
            if self.multimodal_processor and self.orchestrator:
                # Set up multimodal processing integration
                self.orchestrator.set_multimodal_processor(self.multimodal_processor)
            
            # Set up unified memory sharing
            if self.config["integration"]["unified_memory"]:
                await self._setup_unified_memory()
            
            # Set up adaptive routing
            if self.config["integration"]["adaptive_routing"]:
                await self._setup_adaptive_routing()
            
            self.logger.info("✅ Cross-component integrations established")
            
        except Exception as e:
            self.logger.error(f"Integration establishment failed: {e}")
            raise
    
    async def _setup_unified_memory(self):
        """Set up unified memory system across components"""
        try:
            # Create shared memory database
            self.memory_db = sqlite3.connect("shared_memory.db", check_same_thread=False)
            cursor = self.memory_db.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shared_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    component_source TEXT,
                    timestamp REAL,
                    expires_at REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS component_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    source_component TEXT,
                    target_component TEXT,
                    data TEXT,
                    timestamp REAL
                )
            ''')
            
            self.memory_db.commit()
            self.logger.info("✅ Unified memory system established")
            
        except Exception as e:
            self.logger.error(f"Unified memory setup failed: {e}")
    
    async def _setup_adaptive_routing(self):
        """Set up intelligent task routing between components"""
        try:
            # Initialize routing intelligence
            self.routing_intelligence = {
                "component_performance": defaultdict(lambda: {"success_rate": 1.0, "avg_time": 1.0}),
                "task_patterns": defaultdict(list),
                "optimal_routes": {}
            }
            
            self.logger.info("✅ Adaptive routing system established")
            
        except Exception as e:
            self.logger.error(f"Adaptive routing setup failed: {e}")
    
    async def _start_background_services(self):
        """Start background monitoring and optimization services"""
        try:
            # Start health monitoring
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self.background_tasks.append(health_task)
            
            # Start performance monitoring
            perf_task = asyncio.create_task(self._performance_monitoring_loop())
            self.background_tasks.append(perf_task)
            
            # Start integration optimization
            opt_task = asyncio.create_task(self._integration_optimization_loop())
            self.background_tasks.append(opt_task)
            
            # Start memory cleanup
            cleanup_task = asyncio.create_task(self._memory_cleanup_loop())
            self.background_tasks.append(cleanup_task)
            
            self.logger.info("✅ Background services started")
            
        except Exception as e:
            self.logger.error(f"Background services start failed: {e}")
    
    async def _verify_system_health(self):
        """Verify that all systems are healthy and integrated"""
        try:
            health_checks = []
            
            # Check each component
            for name, component in self.active_components.items():
                try:
                    if hasattr(component, 'get_status'):
                        status = component.get_status()
                        health_score = 1.0 if status.get('healthy', True) else 0.5
                    else:
                        health_score = 1.0  # Assume healthy if no status method
                    
                    self.component_status[name] = ComponentStatus(
                        component_name=name,
                        status="healthy",
                        health_score=health_score,
                        last_used=datetime.now(),
                        error_count=0
                    )
                    health_checks.append(True)
                    
                except Exception as e:
                    self.logger.warning(f"Health check failed for {name}: {e}")
                    health_checks.append(False)
            
            overall_health = sum(health_checks) / len(health_checks) if health_checks else 0
            
            if overall_health >= 0.8:
                self.logger.info(f"✅ System health verification passed: {overall_health:.2%}")
            else:
                self.logger.warning(f"⚠️  System health below optimal: {overall_health:.2%}")
                
        except Exception as e:
            self.logger.error(f"Health verification failed: {e}")
    
    # Core integration methods
    
    async def process_request(self, user_id: str, request: str, 
                            modality: str = "text", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Unified request processing through integrated system"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"🎯 Processing request {request_id[:8]} from user {user_id}")
            
            # Update metrics
            self.system_metrics.total_requests += 1
            
            # Initialize user session if new
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    "session_id": str(uuid.uuid4()),
                    "start_time": datetime.now(),
                    "interactions": [],
                    "preferences": {}
                }
            
            session = self.user_sessions[user_id]
            
            # Determine optimal processing route
            processing_route = await self._determine_processing_route(request, modality, context)
            
            # Process through integrated pipeline
            result = await self._execute_integrated_pipeline(
                request_id, user_id, request, modality, context, processing_route
            )
            
            # Learn from interaction
            if self.learning_engine:
                interaction_id = await self.learning_engine.process_interaction(
                    user_id=user_id,
                    input_text=request,
                    input_modality=modality,
                    response_generated=result.get("response", ""),
                    response_time=time.time() - start_time,
                    context=context
                )
                result["interaction_id"] = interaction_id
            
            # Update session
            session["interactions"].append({
                "request_id": request_id,
                "request": request,
                "response": result,
                "timestamp": datetime.now(),
                "processing_time": time.time() - start_time
            })
            
            # Update metrics
            self.system_metrics.successful_requests += 1
            self.system_metrics.average_response_time = (
                self.system_metrics.average_response_time * 0.9 + 
                (time.time() - start_time) * 0.1
            )
            
            result["request_id"] = request_id
            result["processing_time"] = time.time() - start_time
            result["route"] = processing_route
            
            self.logger.info(f"✅ Request {request_id[:8]} processed successfully in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Request {request_id[:8]} failed: {e}")
            return {
                "request_id": request_id,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    async def _determine_processing_route(self, request: str, modality: str, 
                                        context: Dict[str, Any]) -> List[str]:
        """Intelligently determine optimal processing route"""
        try:
            route = []
            
            # Analyze request type
            request_lower = request.lower()
            
            # Multimodal requests go to multimodal processor first
            if modality != "text" or "image" in request_lower or "audio" in request_lower:
                if self.multimodal_processor:
                    route.append("multimodal_processor")
            
            # Data analysis requests
            if any(word in request_lower for word in ["analyze", "data", "chart", "graph", "statistics"]):
                if self.data_analyzer:
                    route.append("data_analyzer")
            
            # Search requests
            if any(word in request_lower for word in ["search", "find", "lookup", "what is", "who is"]):
                if self.search_engine:
                    route.append("search_engine")
            
            # Image generation requests
            if any(word in request_lower for word in ["generate", "create", "draw", "image", "picture"]):
                if self.image_generator:
                    route.append("image_generator")
            
            # Agent system for complex tasks
            if any(word in request_lower for word in ["plan", "strategy", "complex", "multi-step"]):
                if self.agent_system:
                    route.append("agent_system")
            
            # Default to chatbot for conversation
            if not route and self.chatbot:
                route.append("chatbot")
            
            # Always route through orchestrator if available
            if self.orchestrator and "orchestrator" not in route:
                route.insert(0, "orchestrator")
            
            return route
            
        except Exception as e:
            self.logger.error(f"Route determination failed: {e}")
            return ["chatbot"]  # Fallback
    
    async def _execute_integrated_pipeline(self, request_id: str, user_id: str, 
                                         request: str, modality: str, context: Dict[str, Any],
                                         route: List[str]) -> Dict[str, Any]:
        """Execute request through integrated processing pipeline"""
        try:
            results = {}
            final_response = ""
            
            for component_name in route:
                if component_name not in self.active_components:
                    continue
                
                component = self.active_components[component_name]
                
                try:
                    # Process through component
                    if component_name == "orchestrator":
                        # Use orchestrator for complex routing
                        task = Task(
                            id=request_id,
                            description=request,
                            priority=TaskPriority.NORMAL,
                            modality=ModalityType.TEXT,
                            processing_mode=ProcessingMode.REAL_TIME,
                            input_data={"text": request, "modality": modality, "user_id": user_id},
                            metadata=context or {}
                        )
                        result = await component.process_task(task)
                        
                    elif component_name == "multimodal_processor":
                        # Process multimodal content
                        result = await component.process_multimodal_query(
                            query=request,
                            modalities=[modality],
                            context=context
                        )
                        
                    elif component_name == "chatbot":
                        # Simple chat processing
                        result = {"response": component.get_response(request)}
                        
                    elif component_name == "search_engine":
                        # Search processing
                        search_result = component.search(request)
                        result = {"response": search_result, "type": "search"}
                        
                    elif component_name == "data_analyzer":
                        # Data analysis
                        if hasattr(component, 'process_query'):
                            result = component.process_query(request)
                        else:
                            result = {"response": "Data analysis component ready", "type": "analysis"}
                        
                    else:
                        # Generic processing
                        if hasattr(component, 'process'):
                            result = component.process(request)
                        elif hasattr(component, 'get_response'):
                            result = {"response": component.get_response(request)}
                        else:
                            result = {"response": f"Processed by {component_name}"}
                    
                    results[component_name] = result
                    
                    # Extract response for chaining
                    if "response" in result:
                        final_response = result["response"]
                    
                    # Update component usage
                    if component_name in self.component_status:
                        self.component_status[component_name].last_used = datetime.now()
                    
                except Exception as e:
                    self.logger.warning(f"Component {component_name} processing failed: {e}")
                    results[component_name] = {"error": str(e)}
            
            return {
                "response": final_response,
                "component_results": results,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "error": str(e),
                "success": False
            }
    
    async def provide_feedback(self, interaction_id: str, feedback_type: str, 
                             satisfaction_score: float = None):
        """Provide feedback for learning"""
        try:
            if self.learning_engine and interaction_id:
                # Convert string feedback to enum
                feedback_enum = FeedbackType.EXPLICIT_POSITIVE
                if feedback_type.lower() in ["negative", "bad", "incorrect"]:
                    feedback_enum = FeedbackType.EXPLICIT_NEGATIVE
                elif feedback_type.lower() in ["success", "correct", "good"]:
                    feedback_enum = FeedbackType.TASK_SUCCESS
                
                await self.learning_engine.provide_feedback(
                    interaction_id, feedback_enum, satisfaction_score
                )
                
                self.logger.info(f"📝 Feedback provided for interaction {interaction_id}")
                
        except Exception as e:
            self.logger.error(f"Feedback processing failed: {e}")
    
    # Background monitoring loops
    
    async def _health_monitoring_loop(self):
        """Monitor component health continuously"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for name, component in self.active_components.items():
                    try:
                        # Basic health check
                        if hasattr(component, 'get_status'):
                            status = component.get_status()
                            health_score = 1.0 if status.get('healthy', True) else 0.5
                        else:
                            health_score = 1.0
                        
                        if name in self.component_status:
                            self.component_status[name].health_score = health_score
                        
                    except Exception as e:
                        self.logger.warning(f"Health check failed for {name}: {e}")
                        if name in self.component_status:
                            self.component_status[name].error_count += 1
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Update system metrics
                current_time = datetime.now()
                self.system_metrics.uptime_hours = (current_time - self.start_time).total_seconds() / 3600
                self.system_metrics.active_users = len(self.user_sessions)
                
                # Resource utilization
                self.system_metrics.resource_utilization = psutil.cpu_percent()
                
                # Log performance
                self.logger.info(f"📊 Performance - Requests: {self.system_metrics.total_requests}, "
                               f"Success: {self.system_metrics.successful_requests}, "
                               f"Avg Response: {self.system_metrics.average_response_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
    
    async def _integration_optimization_loop(self):
        """Optimize integrations based on performance"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Analyze routing performance
                await self._optimize_routing_intelligence()
                
                # Optimize resource allocation
                if self.resource_optimizer:
                    resource_status = self.resource_optimizer.get_resource_status()
                    self.logger.debug(f"🔧 Resource status: {resource_status}")
                
            except Exception as e:
                self.logger.error(f"Integration optimization error: {e}")
    
    async def _optimize_routing_intelligence(self):
        """Optimize intelligent routing based on performance data"""
        try:
            # Analyze component performance patterns
            for name, status in self.component_status.items():
                if hasattr(status, 'performance_metrics'):
                    # Update routing intelligence based on performance
                    self.routing_intelligence["component_performance"][name] = {
                        "health_score": status.health_score,
                        "error_rate": status.error_count / max(1, self.system_metrics.total_requests)
                    }
            
        except Exception as e:
            self.logger.error(f"Routing optimization failed: {e}")
    
    async def _memory_cleanup_loop(self):
        """Clean up memory and cache periodically"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Clean old cache entries
                current_time = time.time()
                expired_keys = []
                for key, (data, timestamp, ttl) in self.response_cache.items():
                    if current_time - timestamp > ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.response_cache[key]
                
                # Clean old user sessions
                cutoff_time = datetime.now() - timedelta(hours=24)
                expired_sessions = []
                for user_id, session in self.user_sessions.items():
                    if session["start_time"] < cutoff_time:
                        expired_sessions.append(user_id)
                
                for user_id in expired_sessions:
                    del self.user_sessions[user_id]
                
                self.logger.info(f"🧹 Memory cleanup: removed {len(expired_keys)} cache entries, "
                               f"{len(expired_sessions)} old sessions")
                
            except Exception as e:
                self.logger.error(f"Memory cleanup error: {e}")
    
    # Callback methods for component integration
    
    async def _learning_feedback_callback(self, task_result: Dict[str, Any]):
        """Callback for learning feedback from orchestrator"""
        try:
            if self.learning_engine and "interaction_id" in task_result:
                # Process learning feedback
                satisfaction = task_result.get("satisfaction_score", 0.7)
                feedback_type = FeedbackType.TASK_SUCCESS if task_result.get("success", True) else FeedbackType.TASK_FAILURE
                
                await self.learning_engine.provide_feedback(
                    task_result["interaction_id"],
                    feedback_type,
                    satisfaction
                )
                
        except Exception as e:
            self.logger.error(f"Learning feedback callback failed: {e}")
    
    async def _resource_optimization_callback(self, resource_metrics: Dict[str, Any]):
        """Callback for resource optimization from components"""
        try:
            if self.resource_optimizer:
                # Update resource optimization based on component metrics
                pass
                
        except Exception as e:
            self.logger.error(f"Resource optimization callback failed: {e}")
    
    # Public API methods
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                "system_id": self.system_id,
                "status": self.status.value,
                "mode": self.current_mode.value,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
                "active_components": list(self.active_components.keys()),
                "component_status": {
                    name: {
                        "health_score": status.health_score,
                        "status": status.status,
                        "last_used": status.last_used.isoformat(),
                        "error_count": status.error_count
                    } for name, status in self.component_status.items()
                },
                "metrics": {
                    "total_requests": self.system_metrics.total_requests,
                    "successful_requests": self.system_metrics.successful_requests,
                    "success_rate": self.system_metrics.successful_requests / max(1, self.system_metrics.total_requests),
                    "average_response_time": self.system_metrics.average_response_time,
                    "active_users": self.system_metrics.active_users,
                    "resource_utilization": self.system_metrics.resource_utilization
                }
            }
            
        except Exception as e:
            self.logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    def get_user_session(self, user_id: str) -> Dict[str, Any]:
        """Get user session information"""
        try:
            if user_id in self.user_sessions:
                session = self.user_sessions[user_id]
                return {
                    "session_id": session["session_id"],
                    "start_time": session["start_time"].isoformat(),
                    "interaction_count": len(session["interactions"]),
                    "preferences": session["preferences"]
                }
            return {"error": "Session not found"}
            
        except Exception as e:
            self.logger.error(f"Session retrieval failed: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the integration system"""
        try:
            self.logger.info("🛑 Shutting down Master Integration System...")
            
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Shutdown components
            for name, component in self.active_components.items():
                try:
                    if hasattr(component, 'shutdown'):
                        if asyncio.iscoroutinefunction(component.shutdown):
                            await component.shutdown()
                        else:
                            component.shutdown()
                except Exception as e:
                    self.logger.warning(f"Component {name} shutdown failed: {e}")
            
            # Close databases
            if hasattr(self, 'memory_db'):
                self.memory_db.close()
            
            self.logger.info("✅ Master Integration System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


# Example usage and testing
async def main():
    """Example of using the Master Integration System"""
    try:
        print("🌟 Initializing Master Integration System...")
        
        # Initialize the master system
        master_system = MasterIntegrationSystem()
        await master_system.initialize()
        
        print("✅ System initialized! Testing integrated processing...")
        
        # Test various request types
        test_requests = [
            ("user_1", "Hello, can you help me?", "text"),
            ("user_1", "Search for information about AI", "text"),
            ("user_1", "Analyze this data", "text"),
            ("user_2", "Generate an image of a sunset", "text"),
            ("user_2", "What is machine learning?", "text")
        ]
        
        for user_id, request, modality in test_requests:
            result = await master_system.process_request(user_id, request, modality)
            print(f"Request: '{request}' -> Response: '{result.get('response', 'No response')[:100]}...'")
            
            # Provide feedback
            if "interaction_id" in result:
                await master_system.provide_feedback(result["interaction_id"], "positive", 0.8)
        
        # Show system status
        status = master_system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"Components: {status['active_components']}")
        print(f"Total Requests: {status['metrics']['total_requests']}")
        print(f"Success Rate: {status['metrics']['success_rate']:.2%}")
        print(f"Avg Response Time: {status['metrics']['average_response_time']:.2f}s")
        
        # Keep running for a bit to see background services
        print("\n⏱️  Running for 30 seconds to demonstrate background services...")
        await asyncio.sleep(30)
        
        # Shutdown
        await master_system.shutdown()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
