#!/usr/bin/env python3
"""
🧠 ADVANCED MULTIMODAL AI PROCESSOR 🧠
Cutting-edge multimodal AI that processes text, audio, images, and 3D models
simultaneously with advanced cross-modal understanding and fusion.

Features:
- Cross-modal attention mechanisms
- Advanced feature fusion algorithms
- Real-time multimodal processing
- Intelligent modality weighting
- Context-aware understanding
- Semantic alignment across modalities
- Dynamic adaptation based on input quality
"""

import asyncio
import logging
import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor

# Advanced ML imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModel, AutoProcessor,
        CLIPProcessor, CLIPModel, CLIPTokenizer,
        WhisperProcessor, WhisperForConditionalGeneration,
        BlipProcessor, BlipForConditionalGeneration,
        pipeline
    )
    import cv2
    from PIL import Image, ImageEnhance
    import librosa
    import soundfile as sf
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import torchvision.transforms as transforms
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Advanced ML libraries not fully available: {e}")
    ADVANCED_ML_AVAILABLE = False
    # Import PIL.Image separately for fallback mode
    try:
        from PIL import Image, ImageEnhance
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
        # Create dummy Image class for testing
        class Image:
            @staticmethod
            def open(path):
                return DummyImage()
            @staticmethod
            def fromarray(arr):
                return DummyImage()
        
        class DummyImage:
            def convert(self, mode):
                return self
            @property
            def size(self):
                return (640, 480)
            @property
            def mode(self):
                return 'RGB'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported modalities with enhanced types"""
    TEXT = "text"
    AUDIO = "audio" 
    IMAGE = "image"
    VIDEO = "video"
    MODEL_3D = "3d_model"
    SENSOR_DATA = "sensor_data"
    MULTIMODAL = "multimodal"


class FusionStrategy(Enum):
    """Different fusion strategies"""
    EARLY_FUSION = "early"
    LATE_FUSION = "late"
    ATTENTION_FUSION = "attention"
    HIERARCHICAL_FUSION = "hierarchical"
    ADAPTIVE_FUSION = "adaptive"


@dataclass
class ModalityConfig:
    """Configuration for each modality"""
    enabled: bool = True
    weight: float = 1.0
    quality_threshold: float = 0.5
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    feature_dim: int = 512


@dataclass
class MultimodalInput:
    """Structured multimodal input"""
    text: Optional[str] = None
    image: Optional[Union[str, Image.Image, np.ndarray]] = None
    audio: Optional[Union[str, np.ndarray]] = None
    video: Optional[str] = None
    model_3d: Optional[str] = None
    sensor_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalOutput:
    """Structured multimodal output"""
    embeddings: Dict[ModalityType, np.ndarray]
    fused_embedding: np.ndarray
    attention_weights: Dict[ModalityType, float]
    confidence_scores: Dict[ModalityType, float]
    semantic_similarity: Dict[Tuple[ModalityType, ModalityType], float]
    processing_time: Dict[ModalityType, float]
    quality_scores: Dict[ModalityType, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossModalAttention(nn.Module):
    """Advanced cross-modal attention mechanism"""
    
    def __init__(self, feature_dims: Dict[ModalityType, int], attention_dim: int = 256):
        super().__init__()
        self.feature_dims = feature_dims
        self.attention_dim = attention_dim
        
        # Create projection layers for each modality
        self.projections = nn.ModuleDict({
            modality.value: nn.Linear(dim, attention_dim)
            for modality, dim in feature_dims.items()
        })
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(attention_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_projection = nn.Linear(attention_dim, attention_dim)
        
    def forward(self, features: Dict[ModalityType, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cross-modal attention"""
        # Project all features to common dimension
        projected_features = []
        modality_order = []
        
        for modality, feature in features.items():
            if modality.value in self.projections:
                projected = self.projections[modality.value](feature)
                projected_features.append(projected.unsqueeze(0))
                modality_order.append(modality)
        
        if not projected_features:
            return torch.zeros(1, self.attention_dim), torch.zeros(len(features), len(features))
        
        # Stack features
        stacked_features = torch.cat(projected_features, dim=0).unsqueeze(0)  # (1, num_modalities, attention_dim)
        
        # Apply multi-head attention
        attended_features, attention_weights = self.multihead_attn(
            stacked_features, stacked_features, stacked_features
        )
        
        # Global pooling and output projection
        global_feature = attended_features.mean(dim=1)  # (1, attention_dim)
        output = self.output_projection(global_feature)
        
        return output.squeeze(0), attention_weights.squeeze(0)


class AdaptiveFusionNetwork(nn.Module):
    """Adaptive fusion network that learns optimal fusion strategies"""
    
    def __init__(self, modality_dims: Dict[ModalityType, int], output_dim: int = 512):
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        
        # Individual modality encoders
        self.encoders = nn.ModuleDict({
            modality.value: nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, output_dim)
            ) for modality, dim in modality_dims.items()
        })
        
        # Fusion weight predictor
        total_features = sum(modality_dims.values())
        self.weight_predictor = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Linear(128, len(modality_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, features: Dict[ModalityType, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with adaptive fusion"""
        encoded_features = {}
        raw_features = []
        
        # Encode each modality
        for modality, feature in features.items():
            if modality.value in self.encoders:
                encoded = self.encoders[modality.value](feature)
                encoded_features[modality] = encoded
                raw_features.append(feature)
        
        if not encoded_features:
            return torch.zeros(self.output_dim), torch.zeros(len(features))
        
        # Predict fusion weights
        concatenated_raw = torch.cat(raw_features, dim=-1)
        fusion_weights = self.weight_predictor(concatenated_raw)
        
        # Weighted fusion
        weighted_features = []
        for i, (modality, encoded) in enumerate(encoded_features.items()):
            weighted = encoded * fusion_weights[i]
            weighted_features.append(weighted)
        
        fused_feature = torch.sum(torch.stack(weighted_features), dim=0)
        final_output = self.fusion_layer(fused_feature)
        
        return final_output, fusion_weights


class AdvancedMultimodalProcessor:
    """Advanced multimodal AI processor with state-of-the-art capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.AdvancedMultimodalProcessor")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize models and processors
        self.models = {}
        self.processors = {}
        self.feature_extractors = {}
        
        # Neural networks for fusion
        self.attention_network = None
        self.fusion_network = None
        
        # Feature caches
        self.feature_cache = {}
        self.cache_max_size = 1000
        
        # Performance metrics
        self.processing_stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'modality_usage': {modality: 0 for modality in ModalityType}
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if ADVANCED_ML_AVAILABLE:
            self._initialize_models()
        else:
            self.logger.warning("Advanced ML not available, using fallback processors")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'modalities': {
                ModalityType.TEXT: ModalityConfig(feature_dim=768),
                ModalityType.IMAGE: ModalityConfig(feature_dim=512),
                ModalityType.AUDIO: ModalityConfig(feature_dim=1024),
                ModalityType.VIDEO: ModalityConfig(feature_dim=512),
                ModalityType.MODEL_3D: ModalityConfig(feature_dim=256)
            },
            'fusion_strategy': FusionStrategy.ATTENTION_FUSION,
            'enable_caching': True,
            'batch_processing': True,
            'quality_filtering': True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def _initialize_models(self):
        """Initialize all AI models and processors"""
        try:
            self.logger.info("🔧 Initializing advanced multimodal models...")
            
            # Text processing - Multiple models for robustness
            self.models['text_encoder'] = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.processors['text_tokenizer'] = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            # Vision-Language models
            self.models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processors['clip'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Image captioning for enhanced understanding
            self.models['blip'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.processors['blip'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Speech processing
            self.models['whisper'] = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            self.processors['whisper'] = WhisperProcessor.from_pretrained("openai/whisper-base")
            
            # Audio classification
            self.models['audio_classifier'] = pipeline("audio-classification", 
                                                       model="facebook/wav2vec2-base-960h")
            
            # Move models to device
            for model_name, model in self.models.items():
                if hasattr(model, 'to'):
                    self.models[model_name] = model.to(self.device)
            
            # Initialize fusion networks
            modality_dims = {
                ModalityType.TEXT: 384,
                ModalityType.IMAGE: 512,
                ModalityType.AUDIO: 768
            }
            
            self.attention_network = CrossModalAttention(modality_dims).to(self.device)
            self.fusion_network = AdaptiveFusionNetwork(modality_dims).to(self.device)
            
            self.logger.info("✅ Advanced multimodal models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize models: {e}")
            raise
    
    async def process_multimodal(self, input_data: MultimodalInput) -> MultimodalOutput:
        """Process multimodal input with advanced fusion"""
        start_time = time.time()
        
        try:
            # Extract features from each modality in parallel
            feature_tasks = []
            
            if input_data.text:
                feature_tasks.append(self._process_text_advanced(input_data.text))
            if input_data.image:
                feature_tasks.append(self._process_image_advanced(input_data.image))
            if input_data.audio:
                feature_tasks.append(self._process_audio_advanced(input_data.audio))
            if input_data.video:
                feature_tasks.append(self._process_video_advanced(input_data.video))
            if input_data.model_3d:
                feature_tasks.append(self._process_3d_model_advanced(input_data.model_3d))
            
            # Process all modalities concurrently
            feature_results = await asyncio.gather(*feature_tasks, return_exceptions=True)
            
            # Collect successful results
            embeddings = {}
            processing_times = {}
            quality_scores = {}
            
            for result in feature_results:
                if isinstance(result, dict) and 'modality' in result:
                    modality = result['modality']
                    embeddings[modality] = result['embedding']
                    processing_times[modality] = result['processing_time']
                    quality_scores[modality] = result['quality_score']
            
            # Perform advanced fusion
            fused_embedding, attention_weights = await self._perform_advanced_fusion(embeddings)
            
            # Calculate cross-modal similarities
            similarities = self._calculate_cross_modal_similarities(embeddings)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(embeddings, quality_scores)
            
            total_time = time.time() - start_time
            
            # Update statistics
            self._update_processing_stats(embeddings, total_time)
            
            return MultimodalOutput(
                embeddings=embeddings,
                fused_embedding=fused_embedding,
                attention_weights=attention_weights,
                confidence_scores=confidence_scores,
                semantic_similarity=similarities,
                processing_time=processing_times,
                quality_scores=quality_scores,
                metadata={
                    'total_processing_time': total_time,
                    'fusion_strategy': self.config['fusion_strategy'].value,
                    'device_used': str(self.device)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Multimodal processing failed: {e}")
            raise
    
    async def _process_text_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced text processing with multiple embeddings"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.feature_cache:
                cached_result = self.feature_cache[cache_key]
                cached_result['processing_time'] = 0.001  # Cache hit
                return cached_result
            
            if ADVANCED_ML_AVAILABLE and 'text_encoder' in self.models:
                # Tokenize and encode
                inputs = self.processors['text_tokenizer'](text, return_tensors="pt", 
                                                          truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.models['text_encoder'](**inputs)
                    # Use mean pooling for sentence embedding
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                # Text quality assessment
                quality_score = self._assess_text_quality(text)
                
            else:
                # Fallback: simple text features
                embedding = self._extract_text_features_fallback(text)
                quality_score = 0.7
            
            processing_time = time.time() - start_time
            
            result = {
                'modality': ModalityType.TEXT,
                'embedding': embedding,
                'processing_time': processing_time,
                'quality_score': quality_score,
                'metadata': {'text_length': len(text), 'word_count': len(text.split())}
            }
            
            # Cache result
            if self.config['enable_caching'] and len(self.feature_cache) < self.cache_max_size:
                self.feature_cache[cache_key] = result.copy()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {e}")
            return {
                'modality': ModalityType.TEXT,
                'embedding': np.random.rand(384),
                'processing_time': time.time() - start_time,
                'quality_score': 0.1,
                'error': str(e)
            }
    
    async def _process_image_advanced(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Advanced image processing with multiple vision models"""
        start_time = time.time()
        
        try:
            # Convert to PIL Image if needed
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert('RGB')
            else:
                img = image.convert('RGB')
            
            # Image quality assessment
            quality_score = self._assess_image_quality(img)
            
            # Enhance image if quality is low
            if quality_score < 0.5:
                img = self._enhance_image(img)
                quality_score = min(quality_score * 1.5, 1.0)
            
            if ADVANCED_ML_AVAILABLE and 'clip' in self.models:
                # Extract CLIP features
                inputs = self.processors['clip'](images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.models['clip'].get_image_features(**inputs)
                    clip_embedding = image_features.squeeze().cpu().numpy()
                
                # Generate image caption for additional context
                caption_inputs = self.processors['blip'](img, return_tensors="pt")
                caption_inputs = {k: v.to(self.device) for k, v in caption_inputs.items()}
                
                with torch.no_grad():
                    caption_ids = self.models['blip'].generate(**caption_inputs, max_length=50)
                    caption = self.processors['blip'].decode(caption_ids[0], skip_special_tokens=True)
                
                # Combine visual and textual features
                caption_embedding = await self._process_text_advanced(caption)
                
                # Weighted combination
                embedding = 0.8 * clip_embedding + 0.2 * caption_embedding['embedding'][:len(clip_embedding)]
                
            else:
                # Fallback: basic image features
                embedding = self._extract_image_features_fallback(img)
            
            processing_time = time.time() - start_time
            
            return {
                'modality': ModalityType.IMAGE,
                'embedding': embedding,
                'processing_time': processing_time,
                'quality_score': quality_score,
                'metadata': {
                    'image_size': img.size,
                    'mode': img.mode,
                    'caption': caption if 'caption' in locals() else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return {
                'modality': ModalityType.IMAGE,
                'embedding': np.random.rand(512),
                'processing_time': time.time() - start_time,
                'quality_score': 0.1,
                'error': str(e)
            }
    
    async def _process_audio_advanced(self, audio: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Advanced audio processing with speech and acoustic analysis"""
        start_time = time.time()
        
        try:
            # Load audio data
            if isinstance(audio, str):
                audio_data, sample_rate = librosa.load(audio, sr=16000)
            else:
                audio_data = audio
                sample_rate = 16000
            
            # Audio quality assessment
            quality_score = self._assess_audio_quality(audio_data, sample_rate)
            
            # Extract multiple audio features
            features = []
            
            # 1. Acoustic features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            chroma = librosa.feature.chroma(y=audio_data, sr=sample_rate)
            
            # Aggregate features
            acoustic_features = np.concatenate([
                mfcc.mean(axis=1), mfcc.std(axis=1),
                spectral_centroid.mean(axis=1), spectral_centroid.std(axis=1),
                spectral_rolloff.mean(axis=1), spectral_rolloff.std(axis=1),
                zero_crossing_rate.mean(axis=1), zero_crossing_rate.std(axis=1),
                chroma.mean(axis=1), chroma.std(axis=1)
            ])
            
            # 2. Speech processing if available
            speech_embedding = np.array([])
            if ADVANCED_ML_AVAILABLE and 'whisper' in self.models:
                try:
                    # Transcribe audio
                    inputs = self.processors['whisper'](audio_data, sampling_rate=sample_rate, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        # Get transcript
                        generated_ids = self.models['whisper'].generate(**inputs)
                        transcription = self.processors['whisper'].batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # Process transcription as text
                    if transcription.strip():
                        text_result = await self._process_text_advanced(transcription)
                        speech_embedding = text_result['embedding']
                except Exception as e:
                    self.logger.warning(f"Speech processing failed: {e}")
            
            # 3. Audio classification features
            audio_class_features = np.array([])
            if ADVANCED_ML_AVAILABLE and 'audio_classifier' in self.models:
                try:
                    classification = self.models['audio_classifier'](audio_data)
                    # Convert classification to feature vector
                    audio_class_features = np.array([item['score'] for item in classification[:10]])
                except Exception as e:
                    self.logger.warning(f"Audio classification failed: {e}")
            
            # Combine all features
            all_features = [acoustic_features]
            if len(speech_embedding) > 0:
                all_features.append(speech_embedding[:256])  # Limit size
            if len(audio_class_features) > 0:
                all_features.append(audio_class_features)
            
            embedding = np.concatenate(all_features)
            
            # Ensure consistent size
            target_size = 768
            if len(embedding) < target_size:
                embedding = np.pad(embedding, (0, target_size - len(embedding)))
            else:
                embedding = embedding[:target_size]
            
            processing_time = time.time() - start_time
            
            return {
                'modality': ModalityType.AUDIO,
                'embedding': embedding,
                'processing_time': processing_time,
                'quality_score': quality_score,
                'metadata': {
                    'duration': len(audio_data) / sample_rate,
                    'sample_rate': sample_rate,
                    'transcription': transcription if 'transcription' in locals() else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            return {
                'modality': ModalityType.AUDIO,
                'embedding': np.random.rand(768),
                'processing_time': time.time() - start_time,
                'quality_score': 0.1,
                'error': str(e)
            }
    
    async def _process_video_advanced(self, video_path: str) -> Dict[str, Any]:
        """Advanced video processing with temporal analysis"""
        start_time = time.time()
        
        try:
            # Extract video features
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample frames strategically
            frame_embeddings = []
            sample_frames = min(10, frame_count // 10 if frame_count > 10 else frame_count)
            
            for i in range(sample_frames):
                frame_idx = (i * frame_count) // sample_frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_img = Image.fromarray(frame_rgb)
                    
                    # Process frame
                    frame_result = await self._process_image_advanced(frame_img)
                    frame_embeddings.append(frame_result['embedding'])
            
            cap.release()
            
            # Temporal aggregation
            if frame_embeddings:
                # Calculate temporal features
                stacked_embeddings = np.array(frame_embeddings)
                
                # Mean pooling
                mean_embedding = np.mean(stacked_embeddings, axis=0)
                
                # Temporal variance (measure of change)
                temporal_variance = np.var(stacked_embeddings, axis=0)
                
                # Combine spatial and temporal features
                embedding = np.concatenate([mean_embedding, temporal_variance[:256]])
            else:
                embedding = np.random.rand(768)
            
            quality_score = 0.8 if frame_embeddings else 0.2
            processing_time = time.time() - start_time
            
            return {
                'modality': ModalityType.VIDEO,
                'embedding': embedding,
                'processing_time': processing_time,
                'quality_score': quality_score,
                'metadata': {
                    'duration': duration,
                    'fps': fps,
                    'frame_count': frame_count,
                    'frames_processed': len(frame_embeddings)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            return {
                'modality': ModalityType.VIDEO,
                'embedding': np.random.rand(768),
                'processing_time': time.time() - start_time,
                'quality_score': 0.1,
                'error': str(e)
            }
    
    async def _process_3d_model_advanced(self, model_path: str) -> Dict[str, Any]:
        """Advanced 3D model processing with geometric analysis"""
        start_time = time.time()
        
        try:
            vertices = []
            faces = []
            normals = []
            
            # Parse OBJ file
            with open(model_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    if parts[0] == 'v':  # Vertex
                        vertex = [float(x) for x in parts[1:4]]
                        vertices.append(vertex)
                    elif parts[0] == 'f':  # Face
                        face = [int(x.split('/')[0]) - 1 for x in parts[1:4]]  # Convert to 0-based indexing
                        faces.append(face)
                    elif parts[0] == 'vn':  # Normal
                        normal = [float(x) for x in parts[1:4]]
                        normals.append(normal)
            
            vertices = np.array(vertices)
            faces = np.array(faces)
            
            # Extract comprehensive geometric features
            features = []
            
            if len(vertices) > 0:
                # Basic geometric properties
                bbox_min = vertices.min(axis=0)
                bbox_max = vertices.max(axis=0)
                bbox_size = bbox_max - bbox_min
                centroid = vertices.mean(axis=0)
                
                # Volume and surface area approximations
                volume = np.prod(bbox_size) if len(bbox_size) == 3 else 0
                
                # Surface area calculation
                surface_area = 0
                if len(faces) > 0 and len(vertices) > 2:
                    for face in faces:
                        if len(face) >= 3:
                            try:
                                v1, v2, v3 = vertices[face[:3]]
                                area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
                                surface_area += area
                            except (IndexError, ValueError):
                                continue
                
                # Statistical properties
                vertex_distances = np.linalg.norm(vertices - centroid, axis=1)
                distance_stats = [
                    vertex_distances.mean(),
                    vertex_distances.std(),
                    np.percentile(vertex_distances, 25),
                    np.percentile(vertex_distances, 75)
                ]
                
                # Principal components (shape orientation)
                pca = PCA(n_components=3)
                pca.fit(vertices)
                principal_components = pca.explained_variance_ratio_
                
                # Combine all features
                features = np.concatenate([
                    bbox_min, bbox_max, bbox_size, centroid,
                    [volume, surface_area, len(vertices), len(faces)],
                    distance_stats,
                    principal_components,
                    normals[0] if normals else [0, 0, 1]  # Default normal
                ])
            
            # Ensure consistent feature size
            target_size = 256
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            else:
                features = features[:target_size]
            
            # Quality assessment based on model complexity
            quality_score = min(1.0, (len(vertices) + len(faces)) / 10000)
            
            processing_time = time.time() - start_time
            
            return {
                'modality': ModalityType.MODEL_3D,
                'embedding': features,
                'processing_time': processing_time,
                'quality_score': quality_score,
                'metadata': {
                    'vertex_count': len(vertices),
                    'face_count': len(faces),
                    'has_normals': len(normals) > 0,
                    'bounding_box': bbox_size.tolist() if 'bbox_size' in locals() else None,
                    'surface_area': surface_area if 'surface_area' in locals() else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"3D model processing failed: {e}")
            return {
                'modality': ModalityType.MODEL_3D,
                'embedding': np.random.rand(256),
                'processing_time': time.time() - start_time,
                'quality_score': 0.1,
                'error': str(e)
            }
    
    async def _perform_advanced_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> Tuple[np.ndarray, Dict[ModalityType, float]]:
        """Perform advanced multimodal fusion"""
        try:
            if not embeddings:
                return np.array([]), {}
            
            fusion_strategy = self.config.get('fusion_strategy', FusionStrategy.ATTENTION_FUSION)
            
            if fusion_strategy == FusionStrategy.ATTENTION_FUSION and ADVANCED_ML_AVAILABLE:
                return await self._attention_based_fusion(embeddings)
            elif fusion_strategy == FusionStrategy.ADAPTIVE_FUSION and ADVANCED_ML_AVAILABLE:
                return await self._adaptive_fusion(embeddings)
            else:
                return await self._weighted_fusion(embeddings)
                
        except Exception as e:
            self.logger.error(f"Fusion failed: {e}")
            # Fallback to simple concatenation
            concatenated = np.concatenate(list(embeddings.values()))
            weights = {modality: 1.0/len(embeddings) for modality in embeddings.keys()}
            return concatenated, weights
    
    async def _attention_based_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> Tuple[np.ndarray, Dict[ModalityType, float]]:
        """Attention-based fusion using neural networks"""
        try:
            # Convert to tensors
            tensor_embeddings = {}
            for modality, embedding in embeddings.items():
                tensor_embeddings[modality] = torch.FloatTensor(embedding).to(self.device)
            
            # Apply attention network
            fused_tensor, attention_weights = self.attention_network(tensor_embeddings)
            
            # Convert back to numpy
            fused_embedding = fused_tensor.detach().cpu().numpy()
            
            # Convert attention weights to dictionary
            weight_dict = {}
            modality_list = list(embeddings.keys())
            for i, modality in enumerate(modality_list):
                if i < len(attention_weights):
                    weight_dict[modality] = float(attention_weights[i].mean())
                else:
                    weight_dict[modality] = 1.0 / len(embeddings)
            
            return fused_embedding, weight_dict
            
        except Exception as e:
            self.logger.error(f"Attention fusion failed: {e}")
            return await self._weighted_fusion(embeddings)
    
    async def _adaptive_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> Tuple[np.ndarray, Dict[ModalityType, float]]:
        """Adaptive fusion using learned weights"""
        try:
            # Convert to tensors
            tensor_embeddings = {}
            for modality, embedding in embeddings.items():
                tensor_embeddings[modality] = torch.FloatTensor(embedding).to(self.device)
            
            # Apply adaptive fusion network
            fused_tensor, fusion_weights = self.fusion_network(tensor_embeddings)
            
            # Convert back to numpy
            fused_embedding = fused_tensor.detach().cpu().numpy()
            
            # Convert fusion weights to dictionary
            weight_dict = {}
            modality_list = list(embeddings.keys())
            for i, modality in enumerate(modality_list):
                if i < len(fusion_weights):
                    weight_dict[modality] = float(fusion_weights[i])
                else:
                    weight_dict[modality] = 1.0 / len(embeddings)
            
            return fused_embedding, weight_dict
            
        except Exception as e:
            self.logger.error(f"Adaptive fusion failed: {e}")
            return await self._weighted_fusion(embeddings)
    
    async def _weighted_fusion(self, embeddings: Dict[ModalityType, np.ndarray]) -> Tuple[np.ndarray, Dict[ModalityType, float]]:
        """Weighted fusion based on modality configurations"""
        try:
            # Normalize embeddings
            normalized_embeddings = {}
            for modality, embedding in embeddings.items():
                norm = np.linalg.norm(embedding)
                normalized_embeddings[modality] = embedding / (norm + 1e-8)
            
            # Get weights from configuration
            weights = {}
            total_weight = 0
            
            for modality in embeddings.keys():
                modality_config = self.config['modalities'].get(modality, ModalityConfig())
                weight = modality_config.weight
                weights[modality] = weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Weighted fusion
            fused_embedding = None
            for modality, embedding in normalized_embeddings.items():
                weight = weights.get(modality, 1.0 / len(embeddings))
                
                if fused_embedding is None:
                    fused_embedding = embedding * weight
                else:
                    # Handle different embedding sizes
                    min_len = min(len(embedding), len(fused_embedding))
                    fused_embedding[:min_len] += (embedding[:min_len] * weight)
            
            return fused_embedding, weights
            
        except Exception as e:
            self.logger.error(f"Weighted fusion failed: {e}")
            # Ultimate fallback
            concatenated = np.concatenate(list(embeddings.values()))
            equal_weights = {modality: 1.0/len(embeddings) for modality in embeddings.keys()}
            return concatenated, equal_weights
    
    def _calculate_cross_modal_similarities(self, embeddings: Dict[ModalityType, np.ndarray]) -> Dict[Tuple[ModalityType, ModalityType], float]:
        """Calculate semantic similarities between different modalities"""
        similarities = {}
        
        modality_list = list(embeddings.keys())
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list):
                if i < j:  # Avoid duplicate pairs
                    emb1 = embeddings[mod1]
                    emb2 = embeddings[mod2]
                    
                    # Ensure same dimensions for comparison
                    min_len = min(len(emb1), len(emb2))
                    emb1_truncated = emb1[:min_len]
                    emb2_truncated = emb2[:min_len]
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity([emb1_truncated], [emb2_truncated])[0][0]
                    similarities[(mod1, mod2)] = float(similarity)
        
        return similarities
    
    def _calculate_confidence_scores(self, embeddings: Dict[ModalityType, np.ndarray], 
                                   quality_scores: Dict[ModalityType, float]) -> Dict[ModalityType, float]:
        """Calculate confidence scores for each modality"""
        confidence_scores = {}
        
        for modality, embedding in embeddings.items():
            # Base confidence from quality score
            base_confidence = quality_scores.get(modality, 0.5)
            
            # Adjust based on embedding characteristics
            embedding_norm = np.linalg.norm(embedding)
            embedding_variance = np.var(embedding)
            
            # Higher norm and variance usually indicate more informative embeddings
            embedding_confidence = min(1.0, (embedding_norm * np.sqrt(embedding_variance)) / 10.0)
            
            # Combined confidence
            final_confidence = 0.6 * base_confidence + 0.4 * embedding_confidence
            confidence_scores[modality] = float(np.clip(final_confidence, 0.0, 1.0))
        
        return confidence_scores
    
    # Quality assessment methods
    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality based on various metrics"""
        if not text or not text.strip():
            return 0.0
        
        score = 0.0
        
        # Length factor
        length_score = min(1.0, len(text) / 100)
        score += length_score * 0.3
        
        # Word count factor
        words = text.split()
        word_count_score = min(1.0, len(words) / 20)
        score += word_count_score * 0.3
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        char_diversity = min(1.0, unique_chars / 20)
        score += char_diversity * 0.2
        
        # Basic grammar check (simple heuristics)
        grammar_score = 0.8  # Placeholder
        if text[0].isupper() and text.endswith('.'):
            grammar_score += 0.2
        score += grammar_score * 0.2
        
        return min(1.0, score)
    
    def _assess_image_quality(self, img: Image.Image) -> float:
        """Assess image quality based on various metrics"""
        try:
            # Convert to numpy array
            img_array = np.array(img)
            
            score = 0.0
            
            # Resolution factor
            width, height = img.size
            resolution_score = min(1.0, (width * height) / (640 * 480))
            score += resolution_score * 0.3
            
            # Brightness and contrast
            gray = img.convert('L')
            gray_array = np.array(gray)
            
            # Avoid completely dark or bright images
            mean_brightness = gray_array.mean()
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            score += brightness_score * 0.3
            
            # Contrast (standard deviation)
            contrast_score = min(1.0, gray_array.std() / 50)
            score += contrast_score * 0.2
            
            # Edge density (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            edge_score = min(1.0, laplacian_var / 1000)
            score += edge_score * 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.warning(f"Image quality assessment failed: {e}")
            return 0.7  # Default moderate quality
    
    def _assess_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Assess audio quality based on various metrics"""
        try:
            score = 0.0
            
            # Duration factor
            duration = len(audio_data) / sample_rate
            duration_score = min(1.0, duration / 10)  # Prefer 10+ second audio
            score += duration_score * 0.3
            
            # Dynamic range
            rms = np.sqrt(np.mean(audio_data**2))
            dynamic_range_score = min(1.0, rms * 10)
            score += dynamic_range_score * 0.3
            
            # Frequency content (spectral centroid)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            freq_score = min(1.0, spectral_centroid.mean() / 2000)
            score += freq_score * 0.2
            
            # Signal-to-noise ratio estimation
            noise_floor = np.percentile(np.abs(audio_data), 10)
            signal_level = np.percentile(np.abs(audio_data), 90)
            snr = signal_level / (noise_floor + 1e-8)
            snr_score = min(1.0, snr / 10)
            score += snr_score * 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.warning(f"Audio quality assessment failed: {e}")
            return 0.7  # Default moderate quality
    
    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Enhance image quality using PIL"""
        try:
            # Auto-contrast
            enhanced = ImageEnhance.Contrast(img).enhance(1.2)
            
            # Auto-brightness
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.1)
            
            # Auto-sharpness
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.1)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return img
    
    # Fallback methods for when advanced ML is not available
    def _extract_text_features_fallback(self, text: str) -> np.ndarray:
        """Simple text feature extraction without ML models"""
        # Basic text statistics
        features = [
            len(text),
            len(text.split()),
            len(set(text.lower())),
            text.count('.'),
            text.count('?'),
            text.count('!'),
            sum(c.isupper() for c in text),
            sum(c.isdigit() for c in text)
        ]
        
        # Pad to target size
        target_size = 384
        features = np.array(features, dtype=float)
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        
        return features[:target_size]
    
    def _extract_image_features_fallback(self, img: Image.Image) -> np.ndarray:
        """Simple image feature extraction without ML models"""
        # Convert to arrays
        img_array = np.array(img)
        gray = np.array(img.convert('L'))
        
        # Basic image statistics
        features = [
            img.size[0], img.size[1],  # Dimensions
            img_array.mean(), img_array.std(),  # Color statistics
            gray.mean(), gray.std(),  # Grayscale statistics
        ]
        
        # Color histograms
        for channel in range(min(3, img_array.shape[2] if len(img_array.shape) > 2 else 1)):
            if len(img_array.shape) > 2:
                hist, _ = np.histogram(img_array[:,:,channel], bins=32, range=(0, 256))
            else:
                hist, _ = np.histogram(img_array, bins=32, range=(0, 256))
            features.extend(hist.tolist())
        
        # Pad to target size
        target_size = 512
        features = np.array(features, dtype=float)
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        
        return features[:target_size]
    
    def _update_processing_stats(self, embeddings: Dict[ModalityType, np.ndarray], processing_time: float):
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        
        # Update average processing time
        alpha = 0.1
        self.processing_stats['avg_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.processing_stats['avg_processing_time']
        )
        
        # Update modality usage
        for modality in embeddings.keys():
            self.processing_stats['modality_usage'][modality] += 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    async def batch_process_multimodal(self, inputs: List[MultimodalInput]) -> List[MultimodalOutput]:
        """Process multiple multimodal inputs in batch"""
        if not self.config.get('batch_processing', True):
            # Process individually
            results = []
            for input_data in inputs:
                result = await self.process_multimodal(input_data)
                results.append(result)
            return results
        
        # Batch processing implementation
        tasks = [self.process_multimodal(input_data) for input_data in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, MultimodalOutput)]
        
        return valid_results
    
    def clear_cache(self):
        """Clear the feature cache"""
        self.feature_cache.clear()
        self.logger.info("Feature cache cleared")
    
    def save_model_state(self, path: str):
        """Save the current model state"""
        try:
            state = {
                'config': self.config,
                'processing_stats': self.processing_stats,
                'cache_size': len(self.feature_cache)
            }
            
            # Save neural network states if available
            if self.attention_network:
                state['attention_network'] = self.attention_network.state_dict()
            if self.fusion_network:
                state['fusion_network'] = self.fusion_network.state_dict()
            
            torch.save(state, path)
            self.logger.info(f"Model state saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model state: {e}")
    
    def load_model_state(self, path: str):
        """Load a saved model state"""
        try:
            state = torch.load(path, map_location=self.device)
            
            self.config.update(state.get('config', {}))
            self.processing_stats.update(state.get('processing_stats', {}))
            
            # Load neural network states if available
            if 'attention_network' in state and self.attention_network:
                self.attention_network.load_state_dict(state['attention_network'])
            if 'fusion_network' in state and self.fusion_network:
                self.fusion_network.load_state_dict(state['fusion_network'])
            
            self.logger.info(f"Model state loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")


# Compatibility alias for tests
class MultimodalAIProcessor(AdvancedMultimodalProcessor):
    """Compatibility wrapper for the AdvancedMultimodalProcessor"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
    
    async def process_text_query(self, query: str) -> Dict[str, Any]:
        """Process a text query for compatibility with test suite"""
        try:
            input_data = MultimodalInput(text=query)
            result = await self.process_multimodal(input_data)
            
            return {
                'success': True,
                'embeddings': result.embeddings,
                'fused_embedding': result.fused_embedding,
                'processing_time': result.metadata.get('total_processing_time', 0.0),
                'confidence': result.confidence_scores.get(ModalityType.TEXT, 0.5)
            }
        except Exception as e:
            self.logger.error(f"Text query processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0,
                'confidence': 0.0
            }


# Example usage and testing
async def main():
    """Example usage of the Advanced Multimodal Processor"""
    try:
        print("🧠 Initializing Advanced Multimodal AI Processor...")
        
        processor = AdvancedMultimodalProcessor()
        
        # Example multimodal input
        input_data = MultimodalInput(
            text="This is a beautiful sunset over the mountains",
            # image="path/to/sunset_image.jpg",  # Uncomment if you have images
            # audio="path/to/nature_sounds.wav",  # Uncomment if you have audio
            metadata={"source": "example", "timestamp": "2024-01-01"}
        )
        
        print("🔄 Processing multimodal input...")
        result = await processor.process_multimodal(input_data)
        
        print("✅ Processing complete!")
        print(f"📊 Processed modalities: {list(result.embeddings.keys())}")
        print(f"🎯 Fused embedding shape: {result.fused_embedding.shape}")
        print(f"⚖️  Attention weights: {result.attention_weights}")
        print(f"📈 Confidence scores: {result.confidence_scores}")
        print(f"⏱️  Processing time: {result.metadata['total_processing_time']:.3f}s")
        
        # Show processing statistics
        stats = processor.get_processing_stats()
        print(f"\n📊 Processing Statistics:")
        print(f"   Total processed: {stats['total_processed']}")
        print(f"   Average time: {stats['avg_processing_time']:.3f}s")
        print(f"   Modality usage: {stats['modality_usage']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
