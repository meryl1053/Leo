#!/usr/bin/env python3
"""
🎤 Enhanced Speech Recognition System 🎤
Advanced speech recognition with multiple engines, noise reduction, and adaptive listening
"""

import speech_recognition as sr
import numpy as np
import threading
import time
import queue
import os
from typing import Optional, Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSpeechRecognizer:
    """Advanced speech recognition with multiple engines and adaptive settings"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Enhanced recognition settings
        self.setup_recognizer()
        
        # Recognition engines (in order of preference)
        self.recognition_engines = [
            ("Google", self.recognize_google),
            ("Sphinx", self.recognize_sphinx),
            ("Google Cloud", self.recognize_google_cloud),
            ("Azure", self.recognize_azure),
            ("IBM", self.recognize_ibm)
        ]
        
        # Adaptive settings
        self.energy_threshold_history = []
        self.ambient_noise_adjustments = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        
        # Performance tracking
        self.recognition_stats = {
            "total_attempts": 0,
            "successful": 0,
            "engine_performance": {}
        }
        
        logger.info("🎤 Enhanced Speech Recognition System initialized")
    
    def setup_recognizer(self):
        """Configure recognizer with optimal settings"""
        # Dynamic energy threshold (will be adjusted based on environment)
        self.recognizer.energy_threshold = 300  # Start with moderate sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        
        # Timeout settings
        self.timeout = 2.0  # Time to wait for speech to start
        self.phrase_time_limit = 10.0  # Maximum time for a phrase
        
        # Pause threshold (time of silence to end phrase)
        self.recognizer.pause_threshold = 0.8
        
        # Non-speaking duration (silence before timeout)
        self.recognizer.non_speaking_duration = 0.5
        
        logger.info(f"🔧 Recognizer configured with energy_threshold={self.recognizer.energy_threshold}")
    
    def calibrate_microphone(self, duration: float = 1.5) -> bool:
        """Calibrate microphone for current environment"""
        try:
            logger.info("🎚️ Calibrating microphone for ambient noise...")
            
            with self.microphone as source:
                # More thorough ambient noise adjustment
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                
            self.ambient_noise_adjustments += 1
            current_threshold = self.recognizer.energy_threshold
            self.energy_threshold_history.append(current_threshold)
            
            logger.info(f"✅ Microphone calibrated. Energy threshold: {current_threshold}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Microphone calibration failed: {e}")
            return False
    
    def recognize_google(self, audio_data) -> Optional[str]:
        """Google Speech Recognition"""
        try:
            result = self.recognizer.recognize_google(audio_data, language='en-US', show_all=False)
            return result.lower().strip() if result else None
        except Exception as e:
            logger.debug(f"Google recognition failed: {e}")
            return None
    
    def recognize_sphinx(self, audio_data) -> Optional[str]:
        """Sphinx (offline) recognition"""
        try:
            result = self.recognizer.recognize_sphinx(audio_data)
            return result.lower().strip() if result else None
        except Exception as e:
            logger.debug(f"Sphinx recognition failed: {e}")
            return None
    
    def recognize_google_cloud(self, audio_data) -> Optional[str]:
        """Google Cloud Speech-to-Text"""
        try:
            # This requires Google Cloud credentials
            result = self.recognizer.recognize_google_cloud(audio_data, language='en-US')
            return result.lower().strip() if result else None
        except Exception as e:
            logger.debug(f"Google Cloud recognition failed: {e}")
            return None
    
    def recognize_azure(self, audio_data) -> Optional[str]:
        """Azure Speech Services"""
        try:
            # This requires Azure Speech Service key
            result = self.recognizer.recognize_azure(audio_data, language='en-US')
            return result.lower().strip() if result else None
        except Exception as e:
            logger.debug(f"Azure recognition failed: {e}")
            return None
    
    def recognize_ibm(self, audio_data) -> Optional[str]:
        """IBM Watson Speech to Text"""
        try:
            # This requires IBM Watson credentials
            result = self.recognizer.recognize_ibm(audio_data, language='en-US')
            return result.lower().strip() if result else None
        except Exception as e:
            logger.debug(f"IBM recognition failed: {e}")
            return None
    
    def preprocess_audio(self, audio_data) -> sr.AudioData:
        """Apply noise reduction and audio enhancement"""
        try:
            # Convert audio to numpy array for processing
            audio_array = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            
            # Simple noise reduction (you can enhance this)
            # Remove very quiet sounds (likely noise)
            threshold = np.max(np.abs(audio_array)) * 0.01
            audio_array = np.where(np.abs(audio_array) < threshold, 0, audio_array)
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 32767
            
            # Convert back to AudioData
            processed_audio = sr.AudioData(
                audio_array.astype(np.int16).tobytes(),
                audio_data.sample_rate,
                audio_data.sample_width
            )
            
            return processed_audio
            
        except Exception as e:
            logger.debug(f"Audio preprocessing failed: {e}")
            return audio_data  # Return original if preprocessing fails
    
    def listen_with_multiple_engines(self) -> Optional[str]:
        """Listen and try multiple recognition engines"""
        self.recognition_stats["total_attempts"] += 1
        
        try:
            logger.info("👂 Listening for voice input...")
            
            with self.microphone as source:
                # Quick ambient noise adjustment if needed
                if self.ambient_noise_adjustments < 3:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    self.ambient_noise_adjustments += 1
                
                try:
                    # Listen for audio with improved timeouts
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.phrase_time_limit
                    )
                    
                    logger.info("🎵 Audio captured, processing...")
                    
                    # Preprocess audio
                    processed_audio = self.preprocess_audio(audio)
                    
                    # Try each recognition engine
                    for engine_name, engine_func in self.recognition_engines:
                        try:
                            result = engine_func(processed_audio)
                            if result and len(result.strip()) > 0:
                                # Update statistics
                                self.successful_recognitions += 1
                                self.recognition_stats["successful"] += 1
                                
                                if engine_name not in self.recognition_stats["engine_performance"]:
                                    self.recognition_stats["engine_performance"][engine_name] = {"success": 0, "attempts": 0}
                                
                                self.recognition_stats["engine_performance"][engine_name]["success"] += 1
                                self.recognition_stats["engine_performance"][engine_name]["attempts"] += 1
                                
                                logger.info(f"✅ Recognition successful using {engine_name}: '{result}'")
                                return result
                            else:
                                # Track failed attempt for this engine
                                if engine_name not in self.recognition_stats["engine_performance"]:
                                    self.recognition_stats["engine_performance"][engine_name] = {"success": 0, "attempts": 0}
                                self.recognition_stats["engine_performance"][engine_name]["attempts"] += 1
                                
                        except Exception as e:
                            logger.debug(f"{engine_name} recognition failed: {e}")
                            continue
                    
                    # If we reach here, all engines failed
                    self.failed_recognitions += 1
                    logger.warning("⚠️ All recognition engines failed to understand the audio")
                    return None
                    
                except sr.WaitTimeoutError:
                    logger.info("⏱️ Listening timed out - no speech detected")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Listening error: {e}")
            self.failed_recognitions += 1
            return None
    
    def adaptive_adjust_settings(self):
        """Dynamically adjust recognition settings based on performance"""
        total_attempts = self.successful_recognitions + self.failed_recognitions
        
        if total_attempts < 5:
            return  # Need more data
        
        success_rate = self.successful_recognitions / total_attempts
        
        if success_rate < 0.3:  # Less than 30% success rate
            # Increase sensitivity and timeout
            self.recognizer.energy_threshold = max(50, self.recognizer.energy_threshold * 0.8)
            self.timeout = min(5.0, self.timeout + 0.5)
            self.phrase_time_limit = min(15.0, self.phrase_time_limit + 1.0)
            logger.info(f"🔧 Adjusted settings for better recognition: threshold={self.recognizer.energy_threshold}")
            
        elif success_rate > 0.8:  # More than 80% success rate
            # Fine-tune for efficiency
            self.timeout = max(1.0, self.timeout - 0.1)
            logger.info("🎯 Settings optimized for efficient recognition")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_attempts = self.successful_recognitions + self.failed_recognitions
        success_rate = (self.successful_recognitions / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            "total_attempts": total_attempts,
            "successful": self.successful_recognitions,
            "failed": self.failed_recognitions,
            "success_rate": round(success_rate, 1),
            "current_energy_threshold": self.recognizer.energy_threshold,
            "ambient_adjustments": self.ambient_noise_adjustments,
            "engine_stats": self.recognition_stats["engine_performance"]
        }

# Global enhanced recognizer instance
_enhanced_recognizer = None

def get_enhanced_recognizer() -> EnhancedSpeechRecognizer:
    """Get the global enhanced recognizer instance"""
    global _enhanced_recognizer
    if _enhanced_recognizer is None:
        _enhanced_recognizer = EnhancedSpeechRecognizer()
        # Initial calibration
        _enhanced_recognizer.calibrate_microphone()
    return _enhanced_recognizer

def EnhancedListen() -> Optional[str]:
    """Enhanced Listen function with improved recognition"""
    recognizer = get_enhanced_recognizer()
    
    # Adaptive adjustment every few attempts
    if recognizer.recognition_stats["total_attempts"] % 10 == 0:
        recognizer.adaptive_adjust_settings()
    
    # Listen with multiple engines
    result = recognizer.listen_with_multiple_engines()
    
    # Recalibrate periodically if having issues
    if recognizer.failed_recognitions > 0 and recognizer.failed_recognitions % 5 == 0:
        logger.info("🔄 Recalibrating microphone due to recognition issues...")
        recognizer.calibrate_microphone(duration=1.0)
    
    return result

def get_speech_stats() -> Dict[str, Any]:
    """Get current speech recognition statistics"""
    if _enhanced_recognizer:
        return _enhanced_recognizer.get_performance_stats()
    return {}

if __name__ == "__main__":
    # Test the enhanced speech recognition
    print("🎤 Testing Enhanced Speech Recognition")
    print("Say something...")
    
    result = EnhancedListen()
    if result:
        print(f"✅ Recognized: {result}")
        print(f"📊 Stats: {get_speech_stats()}")
    else:
        print("❌ No speech recognized")
