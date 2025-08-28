# Filename: ModelMakerPro.py
# Purpose: Enhanced 3D model generator with advanced features for better quality models

import argparse
import os
import sys
import time
import logging
import json
import random
import subprocess
import platform
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import torch
import torch.nn.functional as F

try:
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import decode_latent_mesh
    from shap_e.util.notebooks import create_pan_cameras
    import trimesh
    from PIL import Image
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install: pip install git+https://github.com/openai/shap-e.git trimesh pillow")
    sys.exit(1)


class PreviewLauncher:
    """Handle automatic preview of generated 3D models."""
    
    @staticmethod
    def open_file(file_path: str) -> bool:
        """Open a file with the system's default application."""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                subprocess.run(["open", file_path], check=True)
            elif system == "Windows":
                os.startfile(file_path)
            elif system == "Linux":
                subprocess.run(["xdg-open", file_path], check=True)
            else:
                logging.warning(f"Unsupported system: {system}")
                return False
            
            return True
            
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to open file with system viewer: {e}")
            return False
        except Exception as e:
            logging.warning(f"Error opening file: {e}")
            return False
    
    @staticmethod
    def find_3d_viewer() -> Optional[str]:
        """Find available 3D viewers on the system."""
        viewers = {
            "Darwin": [  # macOS
                "Preview",
                "/Applications/Blender.app/Contents/MacOS/Blender",
                "/Applications/MeshLab.app/Contents/MacOS/meshlab",
                "/Applications/Cinema 4D.app/Contents/MacOS/Cinema 4D"
            ],
            "Windows": [
                "blender.exe",
                "meshlab.exe",
                "3D Viewer",  # Windows 10/11 built-in
                "Paint 3D"
            ],
            "Linux": [
                "blender",
                "meshlab",
                "g3dviewer",
                "view3dscene"
            ]
        }
        
        system = platform.system()
        if system not in viewers:
            return None
        
        for viewer in viewers[system]:
            try:
                if system == "Windows":
                    result = subprocess.run(["where", viewer], 
                                          capture_output=True, text=True)
                else:
                    result = subprocess.run(["which", viewer], 
                                          capture_output=True, text=True)
                
                if result.returncode == 0:
                    return viewer.strip()
            except:
                continue
        
        return None
    
    @staticmethod
    def launch_preview(file_path: str, auto_preview: bool = True) -> bool:
        """Launch preview of the 3D model file."""
        if not auto_preview:
            return False
        
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False
        
        logging.info(f"Launching preview for: {file_path}")
        
        # Try system default first
        if PreviewLauncher.open_file(file_path):
            logging.info("✅ Opened with system default viewer")
            return True
        
        # Try to find and use a specific 3D viewer
        viewer = PreviewLauncher.find_3d_viewer()
        if viewer:
            try:
                subprocess.Popen([viewer, file_path])
                logging.info(f"✅ Opened with {viewer}")
                return True
            except Exception as e:
                logging.warning(f"Failed to open with {viewer}: {e}")
        
        logging.warning("❌ Could not find suitable 3D viewer")
        logging.info("💡 Try installing Blender, MeshLab, or another 3D viewer")
        return False


@dataclass
class GenerationConfig:
    """Configuration for enhanced model generation."""
    steps: int = 128  # Better quality steps
    resolution: int = 128  # Good balance
    guidance_scale: float = 20.0  # Stronger guidance for better quality
    batch_size: int = 1  # Single generation by default
    use_ensemble: bool = False  # Disabled by default for speed
    post_process: bool = True  # Apply post-processing
    optimize_mesh: bool = True  # Mesh optimization
    generate_textures: bool = False  # Experimental texture generation
    seed: Optional[int] = None  # For reproducible results
    quality_preset: str = "balanced"  # fast, balanced, high, ultra
    auto_preview: bool = True  # Automatically open in preview


class PromptEnhancer:
    """Enhance text prompts for better 3D generation."""
    
    STYLE_MODIFIERS = [
        "highly detailed", "professional quality", "clean geometry", 
        "smooth surfaces", "well-proportioned", "realistic proportions"
    ]
    
    QUALITY_TERMS = [
        "high quality", "detailed", "precise", "well-crafted", 
        "professional", "clean", "polished"
    ]
    
    @staticmethod
    def enhance_prompt(text: str, style: str = "realistic") -> str:
        """Enhance the input prompt for better results."""
        enhanced = text.strip()
        
        # Add quality modifiers if not present
        if not any(term in enhanced.lower() for term in PromptEnhancer.QUALITY_TERMS):
            enhanced = f"high quality, detailed {enhanced}"
        
        # Add style-specific enhancements
        if style == "realistic":
            if "realistic" not in enhanced.lower():
                enhanced = f"realistic {enhanced}"
        elif style == "stylized":
            enhanced = f"stylized, clean geometry {enhanced}"
        elif style == "lowpoly":
            enhanced = f"low poly, geometric {enhanced}"
        
        # Add technical quality terms
        enhanced += ", clean mesh, good topology"
        
        return enhanced
    
    @staticmethod
    def generate_variations(text: str, num_variations: int = 3) -> List[str]:
        """Generate prompt variations for ensemble generation."""
        base = text.strip()
        variations = [base]
        
        modifiers = [
            "highly detailed",
            "clean and precise",
            "professional quality",
            "well-proportioned",
            "smooth and polished"
        ]
        
        for i in range(min(num_variations - 1, len(modifiers))):
            variation = f"{modifiers[i]} {base}"
            variations.append(variation)
        
        return variations


class MeshOptimizer:
    """Advanced mesh optimization and post-processing."""
    
    @staticmethod
    def optimize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply various mesh optimization techniques."""
        try:
            # Remove duplicate vertices
            mesh.merge_vertices()
            
            # Remove degenerate faces
            mesh.remove_degenerate_faces()
            
            # Remove unreferenced vertices
            mesh.remove_unreferenced_vertices()
            
            # Fix face winding
            mesh.fix_normals()
            
            # Smooth mesh if it has too many faces
            if len(mesh.faces) > 50000:
                mesh = mesh.simplify_quadric_decimation(face_count=25000)
            
            # Apply light smoothing
            mesh = mesh.smoothed()
            
            return mesh
            
        except Exception as e:
            logging.warning(f"Mesh optimization failed: {e}")
            return mesh
    
    @staticmethod
    def validate_mesh(mesh: trimesh.Trimesh) -> Tuple[bool, str]:
        """Validate mesh quality."""
        issues = []
        
        if not mesh.is_watertight:
            issues.append("not watertight")
        
        if mesh.body_count > 1:
            issues.append(f"multiple bodies ({mesh.body_count})")
        
        if len(mesh.vertices) < 10:
            issues.append("too few vertices")
        
        if len(mesh.faces) < 10:
            issues.append("too few faces")
        
        # Check for extreme aspect ratios
        bounds = mesh.bounds
        sizes = bounds[1] - bounds[0]
        if np.max(sizes) / np.min(sizes) > 100:
            issues.append("extreme aspect ratio")
        
        is_valid = len(issues) == 0
        message = "Valid mesh" if is_valid else f"Issues: {', '.join(issues)}"
        
        return is_valid, message


class EnhancedModelGenerator:
    """Enhanced 3D model generator with advanced features."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = self._get_device(device)
        self.model = None
        self.diffusion = None
        self.renderer = None
        self.prompt_enhancer = PromptEnhancer()
        self.mesh_optimizer = MeshOptimizer()
        self.preview_launcher = PreviewLauncher()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Configure for different devices
        self._configure_device()
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Auto-detect or validate the specified device."""
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # For Mac users, we'll start with MPS but fallback to CPU if needed
                self.logger.info("MPS detected - will attempt MPS with CPU fallback")
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _configure_device(self):
        """Configure device-specific settings."""
        if self.device.type == 'mps':
            torch.set_default_dtype(torch.float32)
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # Force all numpy operations to use float32
            import numpy as np
            np.seterr(all='ignore')  # Suppress numpy warnings
        elif self.device.type == 'cuda':
            # Enable optimizations for CUDA
            torch.backends.cudnn.benchmark = True
    
    def load_models(self) -> None:
        """Load all required models."""
        try:
            self.logger.info(f"Loading enhanced Shap-E models on {self.device}...")
            
            # Load models with error handling
            self.model = load_model('text300M', device=self.device)
            self.diffusion = diffusion_from_config(load_config('diffusion'))
            self.renderer = load_model('transmitter', device=self.device)
            
            self.logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def _sample_with_advanced_settings(
        self, 
        prompt: str, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """Sample latents with advanced settings."""
        
        # Set seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)
        
        # Advanced sampling parameters based on quality preset
        if config.quality_preset == "ultra":
            karras_steps = min(config.steps, 512)
            sigma_min = 1e-4
            sigma_max = 200.0
            s_churn = 0.1
        elif config.quality_preset == "high":
            karras_steps = config.steps
            sigma_min = 1e-3
            sigma_max = 160.0
            s_churn = 0.05
        elif config.quality_preset == "balanced":
            karras_steps = max(config.steps // 2, 32)  # Good balance
            sigma_min = 1e-3
            sigma_max = 140.0
            s_churn = 0.02
        elif config.quality_preset == "medium":
            karras_steps = max(config.steps // 2, 32)
            sigma_min = 1e-2
            sigma_max = 120.0
            s_churn = 0
        else:  # fast
            karras_steps = max(config.steps // 3, 16)
            sigma_min = 1e-2
            sigma_max = 80.0
            s_churn = 0
        
        # Try MPS first, fallback to CPU if it fails
        if self.device.type == 'mps':
            try:
                self.logger.info("Attempting MPS generation...")
                with torch.no_grad():
                    latents = sample_latents(
                        batch_size=config.batch_size,
                        model=self.model,
                        diffusion=self.diffusion,
                        guidance_scale=float(config.guidance_scale),
                        model_kwargs=dict(texts=[prompt] * config.batch_size),
                        progress=True,
                        clip_denoised=True,
                        use_fp16=False,
                        use_karras=True,
                        karras_steps=karras_steps,
                        sigma_min=float(sigma_min),
                        sigma_max=float(sigma_max),
                        s_churn=float(s_churn),
                        device=self.device,
                    )
                return latents
            except Exception as mps_error:
                self.logger.warning(f"MPS failed: {mps_error}")
                self.logger.info("Falling back to CPU...")
                
                # Move models to CPU
                self.model = self.model.to('cpu')
                self.diffusion = self.diffusion
                self.renderer = self.renderer.to('cpu')
                self.device = torch.device('cpu')
                
                # Retry with CPU
                with torch.no_grad():
                    latents = sample_latents(
                        batch_size=config.batch_size,
                        model=self.model,
                        diffusion=self.diffusion,
                        guidance_scale=config.guidance_scale,
                        model_kwargs=dict(texts=[prompt] * config.batch_size),
                        progress=True,
                        clip_denoised=True,
                        use_fp16=False,
                        use_karras=True,
                        karras_steps=karras_steps,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        s_churn=s_churn,
                        device=self.device,
                    )
                return latents
        else:
            with torch.no_grad():
                latents = sample_latents(
                    batch_size=config.batch_size,
                    model=self.model,
                    diffusion=self.diffusion,
                    guidance_scale=config.guidance_scale,
                    model_kwargs=dict(texts=[prompt] * config.batch_size),
                    progress=True,
                    clip_denoised=True,
                    use_fp16=False,
                    use_karras=True,
                    karras_steps=karras_steps,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    s_churn=s_churn,
                    device=self.device,
                )
            return latents
    
    def _convert_to_trimesh(self, mesh) -> trimesh.Trimesh:
        """Convert any mesh format to trimesh.Trimesh."""
        try:
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
            
            # Handle TorchMesh or other Shap-E mesh formats
            vertices = None
            faces = None
            
            # Try different attribute names
            if hasattr(mesh, 'vertices'):
                vertices = mesh.vertices
            elif hasattr(mesh, 'verts'):
                vertices = mesh.verts
            elif hasattr(mesh, 'vertex_coordinates'):
                vertices = mesh.vertex_coordinates
            
            if hasattr(mesh, 'faces'):
                faces = mesh.faces
            elif hasattr(mesh, 'triangles'):
                faces = mesh.triangles
            elif hasattr(mesh, 'face_indices'):
                faces = mesh.face_indices
            
            if vertices is None or faces is None:
                self.logger.error(f"Could not extract vertices/faces from mesh type: {type(mesh)}")
                self.logger.error(f"Available attributes: {[attr for attr in dir(mesh) if not attr.startswith('_')]}")
                return None
            
            # Convert tensors to numpy
            if hasattr(vertices, 'detach'):
                vertices = vertices.detach().cpu().numpy()
            elif hasattr(vertices, 'numpy'):
                vertices = vertices.numpy()
            vertices = np.array(vertices, dtype=np.float32)
            
            if hasattr(faces, 'detach'):
                faces = faces.detach().cpu().numpy()
            elif hasattr(faces, 'numpy'):
                faces = faces.numpy()
            faces = np.array(faces)
            
            # Create trimesh object
            trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
            return trimesh_obj
            
        except Exception as e:
            self.logger.error(f"Failed to convert mesh to trimesh: {e}")
            return None
    
    def _evaluate_mesh_quality(self, mesh: trimesh.Trimesh) -> float:
        """Evaluate mesh quality with multiple metrics."""
        try:
            score = 0.0
            
            # Vertex count (more is generally better, up to a point)
            vertex_score = min(len(mesh.vertices) / 10000.0, 1.0)
            score += vertex_score * 0.2
            
            # Face count
            face_score = min(len(mesh.faces) / 20000.0, 1.0)
            score += face_score * 0.2
            
            # Watertight bonus
            if mesh.is_watertight:
                score += 0.3
            
            # Volume (non-zero volume is good)
            try:
                volume = float(mesh.volume)
                if abs(volume) > 1e-6:
                    score += 0.2
            except:
                pass
            
            # Surface area
            try:
                area = float(mesh.area)
                if area > 0:
                    score += 0.1
            except:
                pass
            
            return score
            
        except Exception:
            return 0.0
    
    def _select_best_mesh(self, meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
        """Select the best mesh from multiple candidates."""
        if not meshes:
            return None
        
        if len(meshes) == 1:
            return meshes[0]
        
        # Evaluate each mesh
        scored_meshes = []
        for mesh in meshes:
            if mesh is not None:
                score = self._evaluate_mesh_quality(mesh)
                scored_meshes.append((score, mesh))
        
        if not scored_meshes:
            return None
        
        # Sort by score and return the best
        scored_meshes.sort(key=lambda x: x[0], reverse=True)
        best_score, best_mesh = scored_meshes[0]
        
        self.logger.info(f"Selected best mesh with quality score: {best_score:.3f}")
        return best_mesh
    
    def generate_enhanced_model(
        self,
        text_prompt: str,
        output_path: str = None,
        config: Optional[GenerationConfig] = None
    ) -> Tuple[bool, str]:
        """Generate an enhanced 3D model with advanced techniques."""
        
        if config is None:
            config = GenerationConfig()
        
        # Create output path if not provided
        if output_path is None:
            # Clean the prompt for filename
            clean_name = "".join(c for c in text_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_name = clean_name[:50]  # Limit length
            clean_name = clean_name.replace(' ', '_')
            output_path = f"Backend/Models3D/{clean_name}.obj"
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load models if needed
            if self.model is None:
                self.load_models()
            
            start_time = time.time()
            
            # Enhance the prompt
            enhanced_prompt = self.prompt_enhancer.enhance_prompt(text_prompt)
            self.logger.info(f"Enhanced prompt: '{enhanced_prompt}'")
            
            # Generate multiple candidates if using ensemble
            if config.use_ensemble and config.batch_size > 1:
                prompt_variations = self.prompt_enhancer.generate_variations(
                    enhanced_prompt, 
                    min(3, config.batch_size)
                )
                
                all_meshes = []
                
                for i, prompt_var in enumerate(prompt_variations):
                    self.logger.info(f"Generating with variation {i+1}/{len(prompt_variations)}")
                    
                    # Sample latents
                    latents = self._sample_with_advanced_settings(prompt_var, config)
                    
                    # Decode each latent
                    for j, latent in enumerate(latents):
                        try:
                            # Special handling for MPS
                            if self.device.type == 'mps':
                                # Ensure latent is float32
                                latent = latent.to(dtype=torch.float32)
                            
                            mesh = decode_latent_mesh(self.renderer, latent)
                            if mesh is not None:
                                # Convert to trimesh
                                mesh = self._convert_to_trimesh(mesh)
                                if mesh is not None:
                                    all_meshes.append(mesh)
                                    
                        except Exception as e:
                            self.logger.warning(f"Failed to decode latent {j}: {e}")
                            continue
                
                # Select the best mesh
                best_mesh = self._select_best_mesh(all_meshes)
                
            else:
                # Single generation
                latents = self._sample_with_advanced_settings(enhanced_prompt, config)
                latent = latents[0]
                
                # Special handling for MPS
                if self.device.type == 'mps':
                    latent = latent.to(dtype=torch.float32)
                
                best_mesh = decode_latent_mesh(self.renderer, latent)
                
                # Convert to trimesh
                best_mesh = self._convert_to_trimesh(best_mesh)
            
            if best_mesh is None:
                return False, "Failed to generate any valid meshes"
            
            # Validate mesh
            is_valid, validation_msg = self.mesh_optimizer.validate_mesh(best_mesh)
            self.logger.info(f"Mesh validation: {validation_msg}")
            
            # Save the OBJ file
            self.logger.info(f"Saving model to: {output_path}")
            best_mesh.export(output_path)
            
            generation_time = time.time() - start_time
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            
            # Launch preview if enabled
            preview_launched = False
            if config.auto_preview:
                preview_launched = self.preview_launcher.launch_preview(output_path, config.auto_preview)
            
            success_msg = (
                f"✅ Model generated successfully!\n"
                f"   Time: {generation_time:.2f}s\n"
                f"   Vertices: {len(best_mesh.vertices)}\n"
                f"   Faces: {len(best_mesh.faces)}\n"
                f"   File: {output_path} ({file_size:.2f} MB)"
            )
            
            if config.auto_preview:
                if preview_launched:
                    success_msg += "\n   🎨 Model opened in preview!"
                else:
                    success_msg += "\n   ⚠️  Could not auto-launch preview"
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Enhanced generation failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced 3D model generator with advanced features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument("prompt", type=str, help="Text prompt for 3D model")
    parser.add_argument("--output", default=None, help="Output file path (auto-generated if not specified)")
    
    # Enhanced generation parameters
    parser.add_argument("--steps", type=int, default=128, help="Diffusion steps")
    parser.add_argument("--resolution", type=int, default=128, help="Mesh resolution")
    parser.add_argument("--guidance", type=float, default=20.0, help="Guidance scale")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of candidates")
    parser.add_argument("--quality", choices=["fast", "balanced", "high", "ultra"], 
                       default="balanced", help="Quality preset")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    # Feature toggles
    parser.add_argument("--ensemble", action="store_true", help="Enable ensemble generation (better quality)")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable post-processing")
    parser.add_argument("--no-optimization", action="store_true", help="Disable mesh optimization")
    parser.add_argument("--no-preview", action="store_true", help="Disable automatic preview")
    
    # System options
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], 
                       default="auto", help="Device to use")
    parser.add_argument("--force-cpu", action="store_true", 
                       help="Force CPU usage (recommended for Mac if MPS issues)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = GenerationConfig(
        steps=args.steps,
        resolution=args.resolution,
        guidance_scale=args.guidance,
        batch_size=args.batch_size,
        use_ensemble=args.ensemble,
        post_process=not args.no_postprocess,
        optimize_mesh=not args.no_optimization,
        seed=args.seed,
        quality_preset=args.quality,
        auto_preview=not args.no_preview  # Enable by default, disable with --no-preview
    )
    
    # Initialize generator
    device = None if args.device == "auto" else args.device
    if args.force_cpu:
        device = "cpu"
    generator = EnhancedModelGenerator(device=device)
    
    # Print configuration
    print(f"🚀 ModelMaker Pro - Optimized 3D Generation")
    print(f"Device: {generator.device}")
    print(f"Quality: {config.quality_preset}")
    print(f"Ensemble: {config.use_ensemble}")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps: {config.steps}")
    print(f"Auto Preview: {config.auto_preview}")
    print(f"Prompt: '{args.prompt}'")
    print("-" * 60)
    
    # Generate enhanced model with auto-generated path
    success, message = generator.generate_enhanced_model(
        text_prompt=args.prompt,
        output_path=args.output,
        config=config
    )
    
    if success:
        print(message)
        sys.exit(0)
    else:
        print(f"❌ {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
