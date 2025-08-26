import os
import sys
import time
import threading
import logging
from typing import Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import queue

import sounddevice as sd
import numpy as np
import speech_recognition as sr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for voice activation system."""
    sample_rate: int = 44100
    clap_threshold: float = 0.8  # Increased threshold for more accurate detection
    clap_window: float = 1.5  # Max time between claps
    cooldown: float = 10.0    # Increased cooldown to prevent rapid reactivation
    speech_timeout: float = 5.0
    ambient_noise_duration: float = 0.3  # Reduced ambient noise adjustment time
    wake_phrases: list = None
    debug: bool = False
    
    # Advanced clap detection parameters
    min_clap_duration: float = 0.05  # Minimum duration for a clap (50ms)
    max_clap_duration: float = 0.3   # Maximum duration for a clap (300ms)
    clap_cooldown: float = 0.2       # Minimum time between individual claps
    volume_history_size: int = 10    # Number of recent volume samples to track
    
    def __post_init__(self):
        if self.wake_phrases is None:
            self.wake_phrases = ["wake up daddy's home", "wake up daddys home"]


class VoiceActivationSystem:
    """Enhanced voice activation system with improved error handling and structure."""
    
    def __init__(self, config: Config, assistant_callback: Optional[Callable] = None):
        self.config = config
        self.assistant_callback = assistant_callback
        
        # State tracking
        self._last_clap_time = 0
        self._clap_count = 0
        self._last_activation = 0
        self._is_listening = False
        self._shutdown_requested = False
        self._speech_active = False  # Track if speech recognition is active
        self._assistant_active = False  # Track if main assistant is running
        
        # Advanced clap detection state
        self._volume_history = []
        self._last_individual_clap = 0
        self._clap_start_time = None
        self._in_potential_clap = False
        self._baseline_volume = 0.1  # Dynamic baseline volume
        
        # Audio components
        self._recognizer = sr.Recognizer()
        self._microphone = None
        self._audio_stream = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._speech_queue = queue.Queue()
        
        self._initialize_audio_components()
    
    def _initialize_audio_components(self):
        """Initialize audio components with error handling."""
        try:
            self._microphone = sr.Microphone()
            logger.info("Audio components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio components: {e}")
            raise
    
    def _load_assistant_module(self, project_root: str) -> Optional[Callable]:
        """Dynamically load the assistant module."""
        try:
            import importlib.util
            main_path = os.path.join(project_root, "Main.py")
            
            if not os.path.exists(main_path):
                logger.error(f"Main.py not found at {main_path}")
                return None
                
            spec = importlib.util.spec_from_file_location("Main", main_path)
            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)
            
            if hasattr(main_module, 'run_assistant'):
                logger.info("Assistant module loaded successfully")
                return main_module.run_assistant
            else:
                logger.error("run_assistant function not found in Main.py")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load assistant module: {e}")
            return None
    
    @contextmanager
    def _audio_context(self):
        """Context manager for audio stream."""
        stream = None
        try:
            stream = sd.InputStream(
                callback=self._clap_detection_callback,
                channels=1,
                samplerate=self.config.sample_rate,
                blocksize=1024
            )
            stream.start()
            yield stream
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
            raise
        finally:
            if stream:
                stream.stop()
                stream.close()
    
    def _clap_detection_callback(self, indata, frames, time_info, status):
        """Advanced audio callback for clap detection."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            volume = np.linalg.norm(indata) * 10
            current_time = time.time()
            
            # Update volume history for dynamic baseline
            self._volume_history.append(volume)
            if len(self._volume_history) > self.config.volume_history_size:
                self._volume_history.pop(0)
            
            # Calculate dynamic baseline (average of recent low volumes)
            if len(self._volume_history) >= 5:
                sorted_volumes = sorted(self._volume_history)
                self._baseline_volume = np.mean(sorted_volumes[:len(sorted_volumes)//2])
            
            if self.config.debug:
                logger.debug(f"Volume: {volume:.3f}, Baseline: {self._baseline_volume:.3f}, Threshold: {self.config.clap_threshold}")
            
            with self._lock:
                self._process_advanced_clap_detection(volume, current_time)
                    
        except Exception as e:
            logger.error(f"Error in clap detection: {e}")
    
    def _process_advanced_clap_detection(self, volume: float, current_time: float):
        """Advanced clap detection with improved accuracy."""
        # Calculate dynamic threshold based on baseline
        dynamic_threshold = max(self.config.clap_threshold, self._baseline_volume * 3)
        
        # Check if we're in a potential clap
        if volume > dynamic_threshold and not self._in_potential_clap:
            # Start of potential clap
            self._clap_start_time = current_time
            self._in_potential_clap = True
            if self.config.debug:
                logger.debug(f"Potential clap start - Volume: {volume:.3f}")
                
        elif self._in_potential_clap and volume <= dynamic_threshold:
            # End of potential clap
            if self._clap_start_time:
                clap_duration = current_time - self._clap_start_time
                
                # Check if it's a valid clap based on duration
                if (self.config.min_clap_duration <= clap_duration <= self.config.max_clap_duration and
                    current_time - self._last_individual_clap > self.config.clap_cooldown):
                    
                    self._register_valid_clap(current_time)
                    
                elif self.config.debug:
                    logger.debug(f"Invalid clap - Duration: {clap_duration:.3f}s")
            
            self._in_potential_clap = False
            self._clap_start_time = None
        
        # Handle clap sequence timing
        if current_time - self._last_clap_time > self.config.clap_window:
            self._clap_count = 0  # Reset if too much time has passed
    
    def _register_valid_clap(self, current_time: float):
        """Register a valid clap and check for sequence."""
        self._last_individual_clap = current_time
        
        if current_time - self._last_clap_time <= self.config.clap_window:
            self._clap_count += 1
        else:
            self._clap_count = 1
        
        self._last_clap_time = current_time
        
        # Only log claps if not in assistant mode (reduces spam)
        if not self._assistant_active:
            logger.info(f"👏 Valid clap detected! Count: {self._clap_count}")
        elif self.config.debug:
            logger.debug(f"👏 Clap detected but assistant is active! Count: {self._clap_count}")
        
        # Don't trigger if assistant is already active or speech recognition is active
        if (self._clap_count >= 2 and 
            (current_time - self._last_activation) > self.config.cooldown and
            not self._speech_active and not self._assistant_active):
            
            logger.info("👏👏 Two claps detected!")
            self._clap_count = 0
            self._last_activation = current_time
            self._speech_active = True
            
            # Start speech recognition in separate thread
            threading.Thread(
                target=self._handle_speech_recognition,
                daemon=True
            ).start()
        elif self._clap_count >= 2 and (self._assistant_active or self._speech_active):
            # Reset clap count if assistant is active to prevent buildup
            self._clap_count = 0
            if self.config.debug:
                logger.debug("🚫 Claps ignored - assistant or speech recognition active")
    
    def _handle_speech_recognition(self):
        """Handle speech recognition after clap detection."""
        try:
            # Create a new microphone instance for speech recognition
            speech_mic = sr.Microphone()
            
            with speech_mic as source:
                # Adjust for ambient noise
                logger.info("🔧 Adjusting for ambient noise...")
                self._recognizer.adjust_for_ambient_noise(
                    source, 
                    duration=self.config.ambient_noise_duration
                )
                
                logger.info("🎤 Listening for wake phrase...")
                audio = self._recognizer.listen(
                    source, 
                    timeout=self.config.speech_timeout
                )
                
                # Recognize speech
                phrase = self._recognizer.recognize_google(audio).lower()
                logger.info(f"Speech recognized: '{phrase}'")
                
                if self._is_wake_phrase_matched(phrase):
                    logger.info("✅ Wake phrase matched!")
                    self._activate_assistant()
                else:
                    logger.info("❌ Wake phrase did not match")
                    
        except sr.WaitTimeoutError:
            logger.info("⏰ Speech recognition timeout")
        except sr.UnknownValueError:
            logger.info("❓ Could not understand audio")
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in speech recognition: {e}")
        finally:
            # Always reset speech_active flag
            with self._lock:
                self._speech_active = False
    
    def _is_wake_phrase_matched(self, phrase: str) -> bool:
        """Check if the recognized phrase matches any wake phrase."""
        phrase = phrase.lower().strip()
        return any(wake_phrase in phrase for wake_phrase in self.config.wake_phrases)
    
    def _activate_assistant(self):
        """Activate the assistant with error handling."""
        try:
            if self.assistant_callback:
                logger.info("🤖 Activating assistant...")
                with self._lock:
                    self._assistant_active = True  # Set before calling
                time.sleep(0.5)  # Brief pause before activation
                
                # Run assistant in a separate thread to monitor it
                assistant_thread = threading.Thread(
                    target=self._run_assistant_monitored,
                    daemon=True
                )
                assistant_thread.start()
                
                # Wait for assistant to complete (with timeout)
                assistant_thread.join(timeout=120)  # 2 minute timeout
                
            else:
                logger.warning("No assistant callback configured")
        except Exception as e:
            logger.error(f"Error activating assistant: {e}")
        finally:
            # Always reset assistant state when done
            with self._lock:
                self._assistant_active = False
                logger.info("🏁 Assistant session ended")
    
    def _run_assistant_monitored(self):
        """Run the assistant in a monitored thread."""
        try:
            self.assistant_callback()
        except Exception as e:
            logger.error(f"Assistant callback error: {e}")
        finally:
            # This will run when assistant completes
            logger.info("🤖 Assistant callback completed")
    
    def start_listening(self):
        """Start the voice activation system."""
        if self._is_listening:
            logger.warning("System is already listening")
            return
        
        self._is_listening = True
        self._shutdown_requested = False
        
        logger.info("👂 Starting voice activation system...")
        logger.info(f"Wake phrases: {', '.join(self.config.wake_phrases)}")
        
        try:
            with self._audio_context():
                while not self._shutdown_requested:
                    time.sleep(0.1)  # Reduced CPU usage
                    
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main listening loop: {e}")
        finally:
            self._is_listening = False
            logger.info("👋 Voice activation system stopped")
    
    def stop_listening(self):
        """Stop the voice activation system."""
        logger.info("Stopping voice activation system...")
        self._shutdown_requested = True
    
    @property
    def is_listening(self) -> bool:
        """Check if the system is currently listening."""
        return self._is_listening
    
    def set_assistant_active(self, active: bool):
        """Manually set assistant active state (for external control)."""
        with self._lock:
            self._assistant_active = active
            if active:
                logger.info("🤖 Assistant marked as active (external)")
            else:
                logger.info("🏁 Assistant marked as inactive (external)")
    
    def get_status(self) -> dict:
        """Get current system status."""
        with self._lock:
            return {
                "listening": self._is_listening,
                "speech_active": self._speech_active,
                "assistant_active": self._assistant_active,
                "clap_count": self._clap_count,
                "last_activation": self._last_activation
            }


def setup_project_path():
    """Setup project path for module imports."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def main():
    """Main function to run the voice activation system."""
    try:
        # Setup project path
        project_root = setup_project_path()
        
        # Create conservative configuration to reduce false positives
        config = Config(
            debug=False,
            clap_threshold=1.5,    # Very high threshold
            cooldown=20.0,         # Long cooldown
            clap_cooldown=0.4,     # Longer gap between claps
            max_clap_duration=0.2, # Stricter clap duration
            volume_history_size=20  # More baseline samples
        )
        
        # Initialize system
        system = VoiceActivationSystem(config)
        
        # Load assistant callback
        assistant_callback = system._load_assistant_module(project_root)
        if assistant_callback:
            system.assistant_callback = assistant_callback
        else:
            logger.warning("Assistant module not loaded - system will run without callback")
        
        # Start listening
        system.start_listening()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
