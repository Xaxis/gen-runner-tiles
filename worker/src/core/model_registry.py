"""
Model registry for lazy loading of FLUX models, ControlNets, and theme configurations.
Each stage loads only the models it needs when it needs them.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import torch
import structlog

logger = structlog.get_logger()

# Initialize accelerate environment for CPU offloading on MacBook
try:
    import accelerate
    from accelerate import Accelerator
    # Initialize accelerator globally
    accelerator = Accelerator()
    ACCELERATE_AVAILABLE = True
    logger.info("Accelerate initialized successfully for CPU offloading")
except Exception as e:
    ACCELERATE_AVAILABLE = False
    logger.warning("Failed to initialize accelerate", error=str(e))

@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_id: str
    model_type: str  # "base", "controlnet", "lora"
    precision: str = "bfloat16"
    device_map: str = "auto"
    enable_cpu_offload: bool = True
    cache_dir: Optional[str] = None
    requires_auth: bool = False

@dataclass
class ThemeConfig:
    """Configuration for a theme with associated models and prompts."""
    name: str
    description: str
    base_model: str
    controlnet_models: List[str]
    lora_models: List[str]
    style_prompts: Dict[str, str]
    negative_prompts: List[str]
    recommended_steps: int
    guidance_scale: float

@dataclass
class PaletteConfig:
    """Configuration for color palettes."""
    name: str
    description: str
    colors: List[str]  # Hex color codes
    quantization_method: str = "kmeans"
    dithering_enabled: bool = False
    color_space: str = "rgb"

class ModelRegistry:
    """Registry for lazy loading of models, themes, and palettes."""
    
    def __init__(self):

        # Model configurations (lightweight - no actual models loaded)
        self.model_configs: Dict[str, ModelConfig] = {}
        self.theme_configs: Dict[str, ThemeConfig] = {}
        self.palette_configs: Dict[str, PaletteConfig] = {}
        
        # Loaded models cache (only populated when models are actually loaded)
        self.loaded_models: Dict[str, Any] = {}
        
        self._initialize_default_configs()

    # TODO: Move to external config file
    def _initialize_default_configs(self):
        """Initialize default model, theme, and palette configurations."""

        # Base diffusion models
        self.model_configs.update({
            "flux-dev": ModelConfig(
                name="FLUX.1-dev",
                model_id="black-forest-labs/FLUX.1-dev",
                model_type="base",
                precision="bfloat16",
                enable_cpu_offload=True,
                requires_auth=True
            ),
            "flux-schnell": ModelConfig(
                name="FLUX.1-schnell", 
                model_id="black-forest-labs/FLUX.1-schnell",
                model_type="base",
                precision="bfloat16",
                enable_cpu_offload=True,
                requires_auth=False
            )
        })
        
        # ControlNet models
        self.model_configs.update({
            "flux-controlnet-union": ModelConfig(
                name="FLUX ControlNet Union Pro",
                model_id="Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
                model_type="controlnet",
                precision="bfloat16",
                enable_cpu_offload=True
            ),
            "flux-controlnet-depth": ModelConfig(
                name="FLUX ControlNet Depth",
                model_id="Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
                model_type="controlnet",
                precision="bfloat16",
                enable_cpu_offload=True
            ),
            "flux-controlnet-canny": ModelConfig(
                name="FLUX ControlNet Canny",
                model_id="Shakker-Labs/FLUX.1-dev-ControlNet-Canny",
                model_type="controlnet",
                precision="bfloat16",
                enable_cpu_offload=True
            )
        })
        
        # Theme configurations
        self.theme_configs.update({
            "fantasy": ThemeConfig(
                name="Fantasy",
                description="Medieval fantasy style with castles, magic, and mystical elements",
                base_model="flux-dev",
                controlnet_models=["flux-controlnet-union"],
                lora_models=[],
                style_prompts={
                    "base": "medieval fantasy style, detailed pixel art, game tiles",
                    "environment": "stone textures, moss, ancient architecture",
                    "lighting": "warm torch light, magical glow"
                },
                negative_prompts=[
                    "modern elements", "technology", "cars", "phones",
                    "blurry", "low quality", "distorted"
                ],
                recommended_steps=50,
                guidance_scale=3.5
            ),
            "sci-fi": ThemeConfig(
                name="Sci-Fi",
                description="Futuristic sci-fi style with technology and space elements",
                base_model="flux-dev",
                controlnet_models=["flux-controlnet-union"],
                lora_models=[],
                style_prompts={
                    "base": "futuristic sci-fi style, detailed pixel art, game tiles",
                    "environment": "metal textures, neon lights, high-tech surfaces",
                    "lighting": "cool blue lighting, neon glow, holographic effects"
                },
                negative_prompts=[
                    "medieval", "fantasy", "magic", "organic textures",
                    "blurry", "low quality", "distorted"
                ],
                recommended_steps=50,
                guidance_scale=3.5
            ),
            "pixel": ThemeConfig(
                name="Pixel Art",
                description="Classic pixel art style with retro gaming aesthetics",
                base_model="flux-schnell",  # Faster for pixel art
                controlnet_models=["flux-controlnet-canny"],
                lora_models=[],
                style_prompts={
                    "base": "pixel art style, 16-bit graphics, retro game tiles",
                    "environment": "clean pixel textures, limited color palette",
                    "lighting": "flat lighting, no gradients, sharp edges"
                },
                negative_prompts=[
                    "realistic", "photographic", "3d", "smooth gradients",
                    "blurry", "anti-aliased", "high resolution details"
                ],
                recommended_steps=20,  # Faster for pixel art
                guidance_scale=2.0
            ),
            "nature": ThemeConfig(
                name="Nature",
                description="Natural environments with organic textures and earth tones",
                base_model="flux-dev",
                controlnet_models=["flux-controlnet-union"],
                lora_models=[],
                style_prompts={
                    "base": "natural environment style, detailed pixel art, game tiles",
                    "environment": "organic textures, wood, stone, grass, water",
                    "lighting": "natural sunlight, soft shadows, ambient lighting"
                },
                negative_prompts=[
                    "artificial", "synthetic", "metal", "technology",
                    "blurry", "low quality", "distorted"
                ],
                recommended_steps=50,
                guidance_scale=3.5
            )
        })
        
        # Palette configurations
        self.palette_configs.update({
            "retro": PaletteConfig(
                name="Retro 16-Color",
                description="Classic 16-color palette inspired by retro gaming",
                colors=[
                    "#000000", "#1D2B53", "#7E2553", "#008751",
                    "#AB5236", "#5F574F", "#C2C3C7", "#FFF1E8",
                    "#FF004D", "#FFA300", "#FFEC27", "#00E436",
                    "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"
                ],
                quantization_method="kmeans",
                dithering_enabled=True
            ),
            "medieval": PaletteConfig(
                name="Medieval Fantasy",
                description="Classic medieval fantasy palette with warm earth tones and metallic accents",
                colors=[
                    "#2D1C10", "#5B4037", "#8B5A2B", "#B48367",
                    "#D7B19D", "#E4CDAE", "#F0EAD6", "#FFFFFF",
                    "#A57C50", "#7B5833", "#4C3328", "#1C1C1C",
                    "#A5C6E0", "#7393B3", "#4C607D", "#2D3447"
                ],
                quantization_method="kmeans",
                dithering_enabled=False
            ),
            "earth": PaletteConfig(
                name="Earth Tones",
                description="Natural earth tones and organic colors",
                colors=[
                    "#2F1B14", "#8B4513", "#A0522D", "#CD853F",
                    "#DEB887", "#F5DEB3", "#228B22", "#32CD32",
                    "#9ACD32", "#6B8E23", "#556B2F", "#8FBC8F",
                    "#4682B4", "#87CEEB", "#B0C4DE", "#F0F8FF"
                ],
                quantization_method="median_cut",
                dithering_enabled=False
            ),
            "neon": PaletteConfig(
                name="Neon Colors",
                description="Bright neon colors for cyberpunk aesthetics",
                colors=[
                    "#000000", "#FF0080", "#00FF80", "#8000FF",
                    "#FF8000", "#0080FF", "#80FF00", "#FF0040",
                    "#40FF00", "#0040FF", "#FF4000", "#00FF40",
                    "#4000FF", "#FFFF00", "#FF00FF", "#00FFFF"
                ],
                quantization_method="kmeans",
                dithering_enabled=True
            ),
            "monochrome": PaletteConfig(
                name="Monochrome",
                description="Black and white with grayscale",
                colors=[
                    "#000000", "#111111", "#222222", "#333333",
                    "#444444", "#555555", "#666666", "#777777",
                    "#888888", "#999999", "#AAAAAA", "#BBBBBB",
                    "#CCCCCC", "#DDDDDD", "#EEEEEE", "#FFFFFF"
                ],
                quantization_method="uniform",
                dithering_enabled=True
            )
        })
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name (lightweight - no model loading)."""
        return self.model_configs.get(model_name)
    
    def get_theme_config(self, theme_name: str) -> Optional[ThemeConfig]:
        """Get theme configuration by name."""
        return self.theme_configs.get(theme_name)
    
    def get_palette_config(self, palette_name: str) -> Optional[PaletteConfig]:
        """Get palette configuration by name."""
        return self.palette_configs.get(palette_name)
    
    def load_base_model(self, model_name: str):
        """Lazy load a base diffusion model only when needed."""
        if model_name in self.loaded_models:
            logger.info("Using cached base model", model_name=model_name)
            return self.loaded_models[model_name]
        
        config = self.get_model_config(model_name)
        if not config:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if config.model_type != "base":
            raise ValueError(f"Model {model_name} is not a base model")
        
        logger.info("Loading base model", model_name=model_name, model_id=config.model_id)
        
        try:
            # Check for required dependencies first
            try:
                import sentencepiece
            except ImportError:
                raise ImportError(
                    "sentencepiece is required for FLUX models. "
                    "Install it with: pip install sentencepiece"
                )

            from diffusers import FluxPipeline

            # Determine torch dtype
            torch_dtype = getattr(torch, config.precision)
            
            # Load the pipeline with HuggingFace token
            import os
            from huggingface_hub import login

            hf_token = os.getenv('HUGGINGFACE_TOKEN')

            if hf_token:
                logger.info("Logging in to HuggingFace with token")
                login(token=hf_token)
            else:
                logger.warning("No HuggingFace token found in environment")

            if config.enable_cpu_offload and ACCELERATE_AVAILABLE:
                # Create pipeline with device mapping for automatic CPU offloading
                try:
                    pipeline = FluxPipeline.from_pretrained(
                        config.model_id,
                        torch_dtype=torch_dtype,
                        device_map="balanced",  # Balanced device mapping for CPU offloading
                        low_cpu_mem_usage=True  # Reduce CPU memory usage
                    )
                    logger.info("Pipeline created with CPU offloading", model_name=model_name)
                except Exception as e:
                    logger.warning("Failed to create pipeline with CPU offloading, using standard",
                                 model_name=model_name, error=str(e))
                    pipeline = FluxPipeline.from_pretrained(
                        config.model_id,
                        torch_dtype=torch_dtype
                    )
            else:
                pipeline = FluxPipeline.from_pretrained(
                    config.model_id,
                    torch_dtype=torch_dtype
                )
            
            self.loaded_models[model_name] = pipeline
            # Apply retro pixel LoRA for overall aesthetic
            self._apply_retro_pixel_lora(pipeline, model_name)

            logger.info("Base model loaded successfully", model_name=model_name)

            return pipeline
            
        except Exception as e:
            logger.error("Failed to load base model", model_name=model_name, error=str(e))
            raise

    def _apply_retro_pixel_lora(self, pipeline, model_name: str):
        """Apply retro pixel LoRA for overall aesthetic regardless of theme."""
        try:
            # Path to retro pixel LoRA
            retro_pixel_lora_path = "models/loras/retro_pixel_tileset_v1.safetensors"

            # Check if LoRA exists
            import os
            if not os.path.exists(retro_pixel_lora_path):
                logger.warning("Retro pixel LoRA not found, using base model only",
                             path=retro_pixel_lora_path)
                return

            # Load and apply LoRA
            pipeline.load_lora_weights(retro_pixel_lora_path, adapter_name="retro_pixel")
            pipeline.set_adapters(["retro_pixel"], adapter_weights=[0.75])  # 75% strength

            logger.info("Applied retro pixel LoRA",
                       model_name=model_name,
                       lora_path=retro_pixel_lora_path,
                       weight=0.75)

        except Exception as e:
            logger.warning("Failed to apply retro pixel LoRA, continuing with base model",
                         error=str(e), model_name=model_name)
    
    def load_controlnet_model(self, model_name: str):
        """Lazy load a ControlNet model only when needed."""
        if model_name in self.loaded_models:
            logger.info("Using cached ControlNet model", model_name=model_name)
            return self.loaded_models[model_name]
        
        config = self.get_model_config(model_name)
        if not config:
            raise ValueError(f"ControlNet {model_name} not found in registry")
        
        if config.model_type != "controlnet":
            raise ValueError(f"Model {model_name} is not a ControlNet model")
        
        logger.info("Loading ControlNet model", model_name=model_name, model_id=config.model_id)
        
        try:
            from diffusers.models import FluxControlNetModel
            
            # Determine torch dtype
            torch_dtype = getattr(torch, config.precision)
            
            # Load the ControlNet (token already set via login above)
            controlnet = FluxControlNetModel.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype
                # Uses global HuggingFace cache automatically
            )
            
            self.loaded_models[model_name] = controlnet
            logger.info("ControlNet loaded successfully", model_name=model_name)
            
            return controlnet
            
        except Exception as e:
            logger.error("Failed to load ControlNet model", model_name=model_name, error=str(e))
            raise
    
    def validate_job_spec_models(self, job_spec: Dict[str, Any]) -> bool:
        """Validate that all models required by job spec are available."""
        theme_name = job_spec.get("theme")
        
        if not theme_name:
            return False
        
        theme_config = self.get_theme_config(theme_name)
        if not theme_config:
            logger.error("Theme not found", theme=theme_name)
            return False
        
        # Check base model
        if not self.get_model_config(theme_config.base_model):
            logger.error("Base model not found", model=theme_config.base_model)
            return False
        
        # Check ControlNet models
        for controlnet_name in theme_config.controlnet_models:
            if not self.get_model_config(controlnet_name):
                logger.error("ControlNet not found", controlnet=controlnet_name)
                return False
        
        # Check palette
        palette_name = job_spec.get("palette")
        if palette_name and not self.get_palette_config(palette_name):
            logger.error("Palette not found", palette=palette_name)
            return False
        
        return True
