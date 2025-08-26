import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import json
import time
from datetime import datetime
import logging
from typing import List, Optional, Tuple
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check device: use CUDA if available, else MPS on Apple Silicon, else CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
    print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float32  # MPS works better with float32
    print("Using Apple Silicon MPS")
else:
    device = torch.device("cpu")
    dtype = torch.float32
    print("Using CPU (this will be slow)")

print(f"Device: {device} | Data type: {dtype}")

class EnhancedTextToImage:
    def __init__(self, prompt: str, workspace: str, num_images: int = 8, 
                 guidance_scale: float = 7.5, num_inference_steps: int = 50,
                 use_xl: bool = False, enhance_images: bool = True):
        self.prompt = prompt
        self.workspace = workspace
        self.num_images = num_images
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.use_xl = use_xl
        self.enhance_images = enhance_images
        self.start_time = time.time()
        
        # Create output folder structure
        self.setup_workspace()
        
        # Initialize pipeline
        self.pipe = None
        self.load_pipeline()
        
        # Quality enhancement prompts
        self.quality_keywords = [
            "masterpiece", "best quality", "ultra detailed", "8k", "highly detailed",
            "professional photography", "sharp focus", "vivid colors", "perfect lighting",
            "cinematic", "photorealistic", "ultra high resolution"
        ]
        
        # Style variations
        self.style_variations = [
            "photorealistic, detailed",
            "digital art, concept art",
            "oil painting, classical",
            "watercolor, artistic",
            "3D render, modern",
            "vintage photography",
            "cinematic lighting",
            "studio lighting"
        ]
        
    def setup_workspace(self):
        """Create organized output directory structure"""
        folders = ['images', 'enhanced', 'composites', 'logs', 'configs']
        for folder in folders:
            os.makedirs(os.path.join(self.workspace, folder), exist_ok=True)
        
        # Save configuration
        config = {
            'prompt': self.prompt,
            'num_images': self.num_images,
            'guidance_scale': self.guidance_scale,
            'num_inference_steps': self.num_inference_steps,
            'use_xl': self.use_xl,
            'enhance_images': self.enhance_images,
            'device': str(device),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.workspace, 'configs', 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Workspace created at: {self.workspace}")
    
    def load_pipeline(self):
        """Load Stable Diffusion pipeline with error handling"""
        try:
            # Try multiple model options for better compatibility
            models_to_try = [
                ("runwayml/stable-diffusion-v1-5", StableDiffusionPipeline, False),
                ("CompVis/stable-diffusion-v1-4", StableDiffusionPipeline, False),
            ]
            
            if self.use_xl:
                models_to_try.insert(0, ("stabilityai/stable-diffusion-xl-base-1.0", StableDiffusionXLPipeline, True))
            
            for model_id, pipeline_class, is_xl in models_to_try:
                try:
                    logger.info(f"Attempting to load {model_id}...")
                    
                    # Different loading strategies based on device
                    load_kwargs = {
                        "safety_checker": None,
                        "requires_safety_checker": False,
                    }
                    
                    # Adjust dtype based on device
                    if device.type == "mps":
                        load_kwargs["torch_dtype"] = torch.float32
                        load_kwargs["variant"] = None
                    elif device.type == "cuda":
                        load_kwargs["torch_dtype"] = torch.float16
                        load_kwargs["variant"] = "fp16"
                    else:
                        load_kwargs["torch_dtype"] = torch.float32
                        load_kwargs["variant"] = None
                    
                    # Try with safetensors first, fallback without
                    try:
                        load_kwargs["use_safetensors"] = True
                        self.pipe = pipeline_class.from_pretrained(model_id, **load_kwargs)
                    except:
                        logger.warning("Safetensors failed, trying without...")
                        load_kwargs.pop("use_safetensors", None)
                        load_kwargs.pop("variant", None)
                        self.pipe = pipeline_class.from_pretrained(model_id, **load_kwargs)
                    
                    # Move to device carefully
                    if device.type == "mps":
                        # MPS has specific requirements
                        self.pipe = self.pipe.to(device)
                    else:
                        self.pipe = self.pipe.to(device)
                    
                    # Enable optimizations carefully
                    try:
                        if hasattr(self.pipe, 'enable_attention_slicing'):
                            self.pipe.enable_attention_slicing()
                            logger.info("Attention slicing enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable attention slicing: {e}")
                    
                    # Skip CPU offload for MPS
                    if device.type != 'mps':
                        try:
                            if hasattr(self.pipe, 'enable_model_cpu_offload'):
                                self.pipe.enable_model_cpu_offload()
                                logger.info("CPU offload enabled")
                        except Exception as e:
                            logger.warning(f"Could not enable CPU offload: {e}")
                    
                    logger.info(f"Successfully loaded {model_id}!")
                    self.use_xl = is_xl  # Update XL flag based on what actually loaded
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_id}: {e}")
                    continue
            else:
                raise Exception("Failed to load any Stable Diffusion model")
                
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def enhance_prompt(self, base_prompt: str, style: str = None, seed: int = None) -> str:
        """Enhance the base prompt with quality keywords and style"""
        enhanced = base_prompt
        
        # Add random quality keywords
        if seed is not None:
            random.seed(seed)
        
        quality_boost = random.sample(self.quality_keywords, k=min(3, len(self.quality_keywords)))
        
        # Add style if specified
        if style:
            enhanced = f"{enhanced}, {style}"
        
        # Add quality keywords
        enhanced = f"{enhanced}, {', '.join(quality_boost)}"
        
        return enhanced
    
    def _generate_image_attempt(self, prompt: str, generator, attempt_type: str):
        """Helper method to try different generation parameters"""
        try:
            # Adjust parameters based on attempt type
            if attempt_type == "normal":
                guidance = self.guidance_scale
                steps = self.num_inference_steps
            elif attempt_type == "lower_guidance":
                guidance = max(3.0, self.guidance_scale - 2.0)
                steps = self.num_inference_steps
            elif attempt_type == "simple":
                guidance = 7.5
                steps = 30
            else:  # basic
                guidance = 5.0
                steps = 20
            
            logger.info(f"Trying {attempt_type}: guidance={guidance}, steps={steps}")
            
            # Use proper autocast context
            autocast_context = torch.autocast(device.type) if device.type != 'mps' else torch.autocast('cpu')
            
            with autocast_context:
                if self.use_xl:
                    result = self.pipe(
                        prompt,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                        height=1024,
                        width=1024,
                        generator=generator,
                        negative_prompt="blurry, bad quality, distorted, black image, dark, poorly drawn"
                    )
                else:
                    result = self.pipe(
                        prompt,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                        height=512,
                        width=512,
                        generator=generator,
                        negative_prompt="blurry, bad quality, distorted, black image, dark, poorly drawn"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Generation attempt failed: {e}")
            return None
    
    def generate_image_variations(self) -> List[Tuple[Image.Image, str, dict]]:
        """Generate multiple high-quality image variations"""
        logger.info(f"Generating {self.num_images} high-quality images for: '{self.prompt}'")
        
        images_data = []
        
        for i in range(self.num_images):
            try:
                # Use different styles and seeds for variety
                style = self.style_variations[i % len(self.style_variations)] if i < len(self.style_variations) else None
                seed = 42 + i * 1000  # Spread out seeds for variety
                enhanced_prompt = self.enhance_prompt(self.prompt, style, seed)
                
                logger.info(f"Generating image {i+1}/{self.num_images}")
                logger.info(f"Style: {style if style else 'Default'}")
                
                # Generate image with enhanced parameters and better error handling
                generator = torch.Generator(device=device).manual_seed(seed)
                
                # Try generation with different approaches
                result = None
                attempts = [
                    # Attempt 1: Normal generation
                    lambda: self._generate_image_attempt(enhanced_prompt, generator, "normal"),
                    # Attempt 2: Lower guidance scale
                    lambda: self._generate_image_attempt(enhanced_prompt, generator, "lower_guidance"),
                    # Attempt 3: Simplified prompt
                    lambda: self._generate_image_attempt(self.prompt, generator, "simple"),
                    # Attempt 4: Very basic prompt
                    lambda: self._generate_image_attempt("a beautiful image", generator, "basic")
                ]
                
                for attempt_num, attempt_func in enumerate(attempts, 1):
                    try:
                        logger.info(f"Generation attempt {attempt_num}/4...")
                        result = attempt_func()
                        if result and len(result.images) > 0:
                            img = result.images[0]
                            # Check if image is not just black/empty
                            img_array = np.array(img)
                            if img_array.max() > 10:  # Not completely black
                                logger.info(f"Successful generation on attempt {attempt_num}")
                                break
                            else:
                                logger.warning(f"Attempt {attempt_num} produced black image")
                                result = None
                    except Exception as e:
                        logger.warning(f"Attempt {attempt_num} failed: {e}")
                        result = None
                        continue
                
                if not result:
                    logger.error(f"All generation attempts failed for image {i+1}")
                    continue
                
                img = result.images[0]
                
                # Store metadata
                metadata = {
                    'original_prompt': self.prompt,
                    'enhanced_prompt': enhanced_prompt,
                    'style': style,
                    'seed': seed,
                    'guidance_scale': self.guidance_scale,
                    'num_inference_steps': self.num_inference_steps
                }
                
                # Save original image
                style_name = style.split(',')[0].replace(' ', '_') if style else 'default'
                img_filename = f'image_{i+1:02d}_{style_name}_seed_{seed}.png'
                img_path = os.path.join(self.workspace, 'images', img_filename)
                
                # Add metadata to image
                img_with_metadata = self.add_metadata_to_image(img, metadata)
                img_with_metadata.save(img_path, pnginfo=self.create_png_info(metadata))
                
                images_data.append((img, enhanced_prompt, metadata))
                logger.info(f"Saved: {img_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                continue
        
        return images_data
    
    def add_metadata_to_image(self, img: Image.Image, metadata: dict) -> Image.Image:
        """Add subtle metadata overlay to image"""
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        try:
            # Try to use a small font, fallback to default if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Add small text overlay with key info
            text = f"Seed: {metadata['seed']} | Steps: {metadata['num_inference_steps']} | Scale: {metadata['guidance_scale']}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position at bottom right
            x = img.width - text_width - 10
            y = img.height - text_height - 10
            
            # Add semi-transparent background
            draw.rectangle([x-5, y-2, x+text_width+5, y+text_height+2], fill=(0, 0, 0, 128))
            draw.text((x, y), text, fill=(255, 255, 255, 200), font=font)
            
        except Exception as e:
            logger.warning(f"Could not add metadata overlay: {e}")
            return img
        
        return img_copy
    
    def create_png_info(self, metadata: dict):
        """Create PNG info for saving metadata"""
        from PIL.PngImagePlugin import PngInfo
        pnginfo = PngInfo()
        for key, value in metadata.items():
            pnginfo.add_text(key, str(value))
        return pnginfo
    
    def enhance_images_post_processing(self, images_data: List[Tuple[Image.Image, str, dict]]) -> List[Image.Image]:
        """Apply post-processing enhancements to all images"""
        if not self.enhance_images:
            return [img for img, _, _ in images_data]
        
        logger.info("Applying post-processing enhancements...")
        enhanced_images = []
        
        for i, (img, prompt, metadata) in enumerate(images_data):
            try:
                enhanced = self.apply_enhancements(img)
                
                # Save enhanced version
                style_name = metadata.get('style', 'default').split(',')[0].replace(' ', '_')
                enhanced_filename = f'enhanced_{i+1:02d}_{style_name}_seed_{metadata["seed"]}.png'
                enhanced_path = os.path.join(self.workspace, 'enhanced', enhanced_filename)
                enhanced.save(enhanced_path, pnginfo=self.create_png_info(metadata))
                
                enhanced_images.append(enhanced)
                logger.info(f"Enhanced and saved: {enhanced_filename}")
                
            except Exception as e:
                logger.error(f"Failed to enhance image {i+1}: {e}")
                enhanced_images.append(img)  # Use original if enhancement fails
        
        return enhanced_images
    
    def apply_enhancements(self, img: Image.Image) -> Image.Image:
        """Apply various post-processing enhancements"""
        enhanced = img.copy()
        
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.15)
            
            # Enhance color saturation
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            # Apply subtle unsharp mask for better detail
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            # Optional: slight brightness adjustment
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.02)
            
        except Exception as e:
            logger.warning(f"Enhancement step failed: {e}")
        
        return enhanced
    
    def create_composite_grid(self, images: List[Image.Image], enhanced_images: List[Image.Image] = None):
        """Create composite grids showing all generated images"""
        if not images:
            return
        
        try:
            # Calculate grid dimensions
            num_images = len(images)
            if num_images <= 4:
                rows, cols = 2, 2
            elif num_images <= 6:
                rows, cols = 2, 3
            elif num_images <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 4, 4
            
            # Get image dimensions
            img_width, img_height = images[0].size
            
            # Create original images grid
            composite_width = cols * img_width
            composite_height = rows * img_height
            composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
            
            for i, img in enumerate(images[:rows * cols]):
                row = i // cols
                col = i % cols
                x = col * img_width
                y = row * img_height
                composite.paste(img, (x, y))
            
            composite_path = os.path.join(self.workspace, 'composites', 'original_grid.png')
            composite.save(composite_path, quality=95)
            logger.info(f"Original composite saved: {composite_path}")
            
            # Create enhanced images grid if available
            if enhanced_images and self.enhance_images:
                enhanced_composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
                
                for i, img in enumerate(enhanced_images[:rows * cols]):
                    row = i // cols
                    col = i % cols
                    x = col * img_width
                    y = row * img_height
                    enhanced_composite.paste(img, (x, y))
                
                enhanced_composite_path = os.path.join(self.workspace, 'composites', 'enhanced_grid.png')
                enhanced_composite.save(enhanced_composite_path, quality=95)
                logger.info(f"Enhanced composite saved: {enhanced_composite_path}")
                
                # Create comparison grid (original vs enhanced)
                comparison_composite = Image.new('RGB', (composite_width * 2, composite_height), (255, 255, 255))
                comparison_composite.paste(composite, (0, 0))
                comparison_composite.paste(enhanced_composite, (composite_width, 0))
                
                comparison_path = os.path.join(self.workspace, 'composites', 'comparison_grid.png')
                comparison_composite.save(comparison_path, quality=95)
                logger.info(f"Comparison grid saved: {comparison_path}")
            
        except Exception as e:
            logger.error(f"Failed to create composite grids: {e}")
    
    def generate_html_gallery(self, images_data: List[Tuple[Image.Image, str, dict]]):
        """Generate an HTML gallery for easy viewing"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Images - {self.prompt}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .prompt {{ font-size: 24px; font-weight: bold; color: #333; margin-bottom: 10px; }}
        .info {{ color: #666; }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .image-card {{ background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .image-card img {{ width: 100%; height: auto; border-radius: 8px; }}
        .image-meta {{ margin-top: 10px; font-size: 12px; color: #666; }}
        .enhanced-label {{ color: #007bff; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="prompt">"{self.prompt}"</div>
        <div class="info">Generated {len(images_data)} images | Device: {device} | Model: {"SDXL" if self.use_xl else "SD 1.5"}</div>
    </div>
    
    <div class="gallery">
"""
            
            for i, (_, prompt, metadata) in enumerate(images_data):
                style_name = metadata.get('style', 'default').split(',')[0].replace(' ', '_')
                img_filename = f'image_{i+1:02d}_{style_name}_seed_{metadata["seed"]}.png'
                enhanced_filename = f'enhanced_{i+1:02d}_{style_name}_seed_{metadata["seed"]}.png'
                
                html_content += f"""
        <div class="image-card">
            <img src="images/{img_filename}" alt="Generated image {i+1}">
            <div class="image-meta">
                <strong>Style:</strong> {metadata.get('style', 'Default')}<br>
                <strong>Seed:</strong> {metadata['seed']}<br>
                <strong>Steps:</strong> {metadata['num_inference_steps']}<br>
                <strong>Guidance:</strong> {metadata['guidance_scale']}
            </div>
"""
                
                if self.enhance_images:
                    html_content += f"""
            <div style="margin-top: 10px;">
                <a href="enhanced/{enhanced_filename}" class="enhanced-label">View Enhanced Version</a>
            </div>
"""
                
                html_content += "        </div>\n"
            
            html_content += """
    </div>
</body>
</html>
"""
            
            html_path = os.path.join(self.workspace, 'gallery.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"HTML gallery created: {html_path}")
            
        except Exception as e:
            logger.error(f"Failed to create HTML gallery: {e}")
    
    def generate_progress_report(self, images_data: List[Tuple[Image.Image, str, dict]]):
        """Generate detailed progress report"""
        elapsed_time = time.time() - self.start_time
        
        report = {
            'prompt': self.prompt,
            'workspace': self.workspace,
            'num_images_requested': self.num_images,
            'num_images_generated': len(images_data),
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_formatted': time.strftime('%H:%M:%S', time.gmtime(elapsed_time)),
            'device_used': str(device),
            'model_used': 'SDXL' if self.use_xl else 'SD 1.5',
            'settings': {
                'guidance_scale': self.guidance_scale,
                'num_inference_steps': self.num_inference_steps,
                'enhance_images': self.enhance_images
            },
            'files_generated': [],
            'completion_time': datetime.now().isoformat()
        }
        
        # Count generated files
        for root, dirs, files in os.walk(self.workspace):
            for file in files:
                if file.endswith(('.png', '.jpg', '.html', '.json')):
                    report['files_generated'].append(os.path.relpath(os.path.join(root, file), self.workspace))
        
        # Save report
        report_path = os.path.join(self.workspace, 'logs', 'generation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generation report saved: {report_path}")
        return report
    
    def run(self):
        """Main execution pipeline"""
        try:
            logger.info("="*60)
            logger.info(f"ENHANCED TEXT-TO-IMAGE GENERATION")
            logger.info(f"Prompt: '{self.prompt}'")
            logger.info(f"Images to generate: {self.num_images}")
            logger.info(f"Model: {'Stable Diffusion XL' if self.use_xl else 'Stable Diffusion 1.5'}")
            logger.info("="*60)
            
            # Generate image variations
            images_data = self.generate_image_variations()
            
            if not images_data:
                logger.error("No images were generated successfully!")
                return
            
            # Apply post-processing enhancements
            enhanced_images = self.enhance_images_post_processing(images_data)
            
            # Create composite grids
            original_images = [img for img, _, _ in images_data]
            self.create_composite_grid(original_images, enhanced_images if self.enhance_images else None)
            
            # Generate HTML gallery
            self.generate_html_gallery(images_data)
            
            # Generate final report
            report = self.generate_progress_report(images_data)
            
            logger.info("="*60)
            logger.info("GENERATION COMPLETE!")
            logger.info(f"Successfully generated: {len(images_data)}/{self.num_images} images")
            logger.info(f"Total time: {report['elapsed_time_formatted']}")
            logger.info(f"Files created: {len(report['files_generated'])}")
            logger.info(f"Output location: {self.workspace}")
            logger.info(f"View gallery: {os.path.join(self.workspace, 'gallery.html')}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Text to High-Quality Image Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_image_gen.py --text "a majestic dragon" --workspace ./dragon_images --num_images 6
  python enhanced_image_gen.py --text "futuristic cityscape" --workspace ./city --xl --enhance --guidance_scale 8.0
  python enhanced_image_gen.py --text "portrait of a wise wizard" --workspace ./wizard --num_images 4 --steps 75
        """
    )
    
    parser.add_argument("--text", type=str, required=True,
                       help="Text prompt for image generation")
    parser.add_argument("--workspace", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--num_images", type=int, default=8,
                       help="Number of images to generate (default: 8)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale (1.0-20.0, higher = more prompt adherence)")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps (higher = better quality)")
    parser.add_argument("--xl", action="store_true",
                       help="Use Stable Diffusion XL (higher quality, slower)")
    parser.add_argument("--enhance", action="store_true",
                       help="Apply post-processing enhancements")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.guidance_scale < 1.0 or args.guidance_scale > 20.0:
        logger.warning("Guidance scale should be between 1.0 and 20.0")
        args.guidance_scale = max(1.0, min(20.0, args.guidance_scale))
    
    if args.steps < 10 or args.steps > 150:
        logger.warning("Steps should be between 10 and 150")
        args.steps = max(10, min(150, args.steps))
    
    if args.num_images < 1 or args.num_images > 20:
        logger.warning("Number of images should be between 1 and 20")
        args.num_images = max(1, min(20, args.num_images))
    
    # Create and run generator
    generator = EnhancedTextToImage(
        prompt=args.text,
        workspace=args.workspace,
        num_images=args.num_images,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        use_xl=args.xl,
        enhance_images=args.enhance
    )
    
    generator.run()

if __name__ == "__main__":
    main()
