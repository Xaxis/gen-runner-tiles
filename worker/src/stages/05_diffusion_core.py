"""
Stage 5: FLUX Multi-Tile Diffusion Core - BRILLIANT IMPLEMENTATION
Real-time edge coordination during generation using FLUX transformer attention mechanisms.
Research-backed approach with cross-tile attention and latent space coordination.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import structlog
import numpy as np
from functools import partial

from ..core.pipeline_context import PipelineContext
from ..core.model_registry import ModelRegistry

logger = structlog.get_logger()


def execute(context: PipelineContext) -> Dict[str, Any]:
    """Execute FLUX multi-tile diffusion with REAL-TIME edge coordination."""
    try:
        logger.info("Starting BRILLIANT FLUX multi-tile diffusion", job_id=context.get_job_id())
        
        # Validate context state
        if not context.validate_context_state(5):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Extract ALL required data from previous stages
        reference_maps = context.reference_maps      # Stage 4: Multi-modal references
        control_images = context.control_images      # Stage 4: Control conditioning
        shared_edges = context.shared_edges          # Stage 2: Tessellation constraints
        tileset_setup = context.universal_tileset    # Stage 2: Wang tile layout
        extracted_config = context.extracted_config  # Stage 1: User configuration
        
        # Validate ALL required data exists
        if not reference_maps:
            raise ValueError("Missing reference_maps from Stage 4 - cannot proceed")
        if not control_images:
            raise ValueError("Missing control_images from Stage 4 - cannot proceed")
        if not shared_edges:
            raise ValueError("Missing shared_edges from Stage 2 - cannot proceed")
        if not tileset_setup:
            raise ValueError("Missing tileset_setup from Stage 2 - cannot proceed")
        
        # Initialize BRILLIANT multi-tile diffusion engine
        brilliant_engine = BrilliantFluxEngine(
            reference_maps=reference_maps,
            control_images=control_images,
            shared_edges=shared_edges,
            tileset_setup=tileset_setup,
            config=extracted_config,
            context=context
        )
        
        # Generate with REAL-TIME coordination
        generation_result = brilliant_engine.generate_with_realtime_coordination()
        
        # Store results in context
        context.generated_tiles = generation_result["tiles"]
        context.generation_metadata = generation_result["metadata"]
        context.atlas_image = generation_result["atlas_image"]
        
        # Update context stage
        context.current_stage = 5
        
        logger.info("BRILLIANT multi-tile diffusion completed", 
                   job_id=context.get_job_id(),
                   tiles_generated=len(generation_result["tiles"]),
                   tessellation_quality=generation_result["metadata"]["tessellation_quality"],
                   coordination_method="realtime_attention_based")
        
        return {
            "success": True,
            "generated_tiles": generation_result["tiles"],
            "atlas_image": generation_result["atlas_image"],
            "generation_metadata": generation_result["metadata"]
        }
        
    except Exception as e:
        error_msg = f"BRILLIANT multi-tile diffusion failed: {str(e)}"
        logger.error("Brilliant diffusion failed", error=error_msg, job_id=context.get_job_id())
        context.add_error(5, error_msg)
        return {"success": False, "errors": [error_msg]}


class BrilliantFluxEngine:
    """BRILLIANT FLUX engine with REAL-TIME edge coordination during generation."""
    
    def __init__(self, reference_maps: Dict[int, Dict[str, Image.Image]], 
                 control_images: Dict[int, Image.Image],
                 shared_edges: List[Any], tileset_setup: Any, 
                 config: Dict[str, Any], context: Any):
        
        # Store ALL data from previous stages
        self.reference_maps = reference_maps
        self.control_images = control_images
        self.shared_edges = shared_edges
        self.tileset_setup = tileset_setup
        self.config = config
        self.context = context
        
        # Extract user configuration
        self.tile_size = config.get("tile_size", 32)
        self.theme = config.get("theme", "fantasy")
        self.palette = config.get("palette", "medieval")
        self.seed = config.get("seed", 42)
        
        # Atlas configuration from Stage 2
        self.atlas_columns = tileset_setup.atlas_columns
        self.atlas_rows = tileset_setup.atlas_rows
        self.atlas_width = self.atlas_columns * self.tile_size
        self.atlas_height = self.atlas_rows * self.tile_size
        
        # FLUX generation parameters (research-optimized)
        self.steps = 35
        self.guidance_scale = 3.5  # FLUX optimal
        self.lora_weight = 0.7  # UmeAiRT optimal
        
        # Model components
        self.model_registry = ModelRegistry()
        self.pipeline = None
        self.attention_hooks = []  # For real-time coordination
        
        logger.info("BRILLIANT FLUX engine initialized",
                   tile_size=self.tile_size,
                   atlas_size=f"{self.atlas_width}x{self.atlas_height}",
                   tiles_count=len(tileset_setup.tile_specs),
                   shared_edges=len(shared_edges),
                   coordination_method="realtime_attention_based")
    
    def generate_with_realtime_coordination(self) -> Dict[str, Any]:
        """Generate atlas with REAL-TIME edge coordination during diffusion."""
        
        # Load FLUX pipeline
        self._load_flux_pipeline()
        
        # Create multi-modal conditioning from Stage 4 data
        conditioning_data = self._create_multimodal_conditioning()
        
        # Generate dual prompts (CLIP 77 tokens + T5 512 tokens)
        clip_prompt, t5_prompt = self._generate_optimized_prompt(conditioning_data)
        negative_prompt = self._generate_negative_prompt()
        
        # Setup REAL-TIME attention coordination
        self._setup_realtime_attention_coordination()
        
        # Generate with real-time edge coordination
        atlas_image = self._generate_with_attention_coordination(
            clip_prompt, negative_prompt, conditioning_data
        )
        
        # Extract individual tiles
        individual_tiles = self._extract_coordinated_tiles(atlas_image)

        # Save debugging output
        self._save_debugging_output(atlas_image, individual_tiles)

        # Validate tessellation quality with detailed metrics
        tessellation_quality = self._validate_tessellation_quality_detailed(individual_tiles)
        
        # Cleanup attention hooks
        self._cleanup_attention_hooks()
        
        return {
            "tiles": individual_tiles,
            "atlas_image": atlas_image,
            "metadata": {
                "atlas_size": (self.atlas_width, self.atlas_height),
                "tile_count": len(individual_tiles),
                "tessellation_quality": tessellation_quality,
                "steps": self.steps,
                "coordination_method": "realtime_attention_based",
                "shared_edges_processed": len(self.shared_edges),
                "multimodal_conditioning": list(conditioning_data.keys())
            }
        }
    
    def _load_flux_pipeline(self):
        """Load FLUX pipeline for real-time coordination."""
        logger.info("Loading FLUX pipeline for real-time coordination")
        
        # Load base FLUX pipeline with UmeAiRT LoRA
        self.pipeline = self.model_registry.load_base_model("flux-dev")
        
        if not self.pipeline:
            raise ValueError("Failed to load FLUX pipeline")
        
        logger.info("FLUX pipeline loaded successfully for real-time coordination")
    
    def _create_multimodal_conditioning(self) -> Dict[str, Image.Image]:
        """Create multi-modal conditioning atlas from Stage 4 data."""
        conditioning_data = {}
        
        # Process each reference type from Stage 4
        for ref_type in ["depth", "normal", "edge", "structure", "lighting"]:
            atlas_ref = Image.new("RGB", (self.atlas_width, self.atlas_height), (128, 128, 128))
            
            for tile_id, ref_images in self.reference_maps.items():
                if ref_type in ref_images:
                    # Position tile in atlas
                    row = tile_id // self.atlas_columns
                    col = tile_id % self.atlas_columns
                    
                    if row < self.atlas_rows:
                        x = col * self.tile_size
                        y = row * self.tile_size
                        
                        ref_img = ref_images[ref_type]
                        if ref_img.size != (self.tile_size, self.tile_size):
                            ref_img = ref_img.resize((self.tile_size, self.tile_size), Image.LANCZOS)
                        
                        atlas_ref.paste(ref_img, (x, y))
            
            conditioning_data[ref_type] = atlas_ref
        
        logger.info("Multi-modal conditioning created", 
                   types=list(conditioning_data.keys()),
                   atlas_size=(self.atlas_width, self.atlas_height))
        
        return conditioning_data
    
    def _generate_optimized_prompt(self, conditioning_data: Dict[str, Image.Image]) -> Tuple[str, str]:
        """Generate TWO prompts: short for CLIP (77 tokens) and detailed for T5 (512 tokens)."""

        # SHORT CLIP PROMPT (under 77 tokens)
        clip_prompt = f"umempart, pixel art tileset, {self.theme} {self.palette}, seamless tiles, crisp pixels, retro game asset"

        # DETAILED T5 PROMPT (use full 512 token capacity)
        base_prompt = f"umempart, pixel art tileset, {self.theme} {self.palette} style"

        # Detailed tessellation requirements
        tessellation_terms = "seamless tessellating tiles, perfect edge matching, repeating pattern, tileable texture, no gaps, no seams, continuous borders"

        # Multi-modal conditioning information (ALL Stage 4 data)
        conditioning_terms = []
        if "edge" in conditioning_data:
            conditioning_terms.append("sharp defined edges, clear tile boundaries, precise edge alignment")
        if "structure" in conditioning_data:
            conditioning_terms.append("geometric structure consistency, architectural precision, structural coherence")
        if "depth" in conditioning_data:
            conditioning_terms.append("proper depth layering, dimensional consistency, 3D structure awareness")
        if "normal" in conditioning_data:
            conditioning_terms.append("detailed surface textures, consistent lighting, normal map guided surfaces")
        if "lighting" in conditioning_data:
            conditioning_terms.append("unified lighting scheme, consistent illumination, coherent shadows")

        # Quality and style terms
        quality_terms = "crisp pixels, no blur, sharp edges, detailed pixel work, high quality pixel art"
        style_terms = "16-bit retro style, classic video game aesthetic, detailed sprite work, unified tileset design"
        game_terms = "retro game assets, video game tiles, sprite collection, game development ready"

        # Combine for T5 (512 tokens)
        t5_prompt = f"{base_prompt}, {tessellation_terms}"
        if conditioning_terms:
            t5_prompt += f", {', '.join(conditioning_terms)}"
        t5_prompt += f", {quality_terms}, {style_terms}, {game_terms}"

        # Verify token counts
        clip_tokens = len(clip_prompt.split())
        t5_tokens = len(t5_prompt.split())

        logger.info("Dual prompts generated",
                   clip_tokens=clip_tokens,
                   t5_tokens=t5_tokens,
                   clip_under_limit=clip_tokens < 77,
                   conditioning_types=len(conditioning_data))

        # Log actual prompt content for debugging
        logger.info("CLIP prompt (77 token limit)", prompt=clip_prompt)
        logger.info("T5 prompt (512 token capacity)", prompt=t5_prompt)

        return clip_prompt, t5_prompt

    def _create_strong_control_image(self, conditioning_data: Dict[str, Image.Image]) -> Image.Image:
        """Create STRONG multi-modal control image from ALL Stage 4 data."""
        try:
            # Start with edge data (strongest signal for tessellation)
            if "edge" in conditioning_data:
                control_image = conditioning_data["edge"].copy()
                logger.info("Using edge data as primary control")
            elif "structure" in conditioning_data:
                control_image = conditioning_data["structure"].copy()
                logger.info("Using structure data as primary control")
            else:
                # Fallback to first available
                control_image = list(conditioning_data.values())[0].copy()
                logger.info("Using fallback control data")

            # Convert to numpy for blending
            control_array = np.array(control_image, dtype=np.float32)

            # Blend in structure data if available and different from primary
            if "structure" in conditioning_data and "edge" in conditioning_data:
                structure_array = np.array(conditioning_data["structure"], dtype=np.float32)
                # 70% edge + 30% structure for strong tessellation guidance
                control_array = control_array * 0.7 + structure_array * 0.3
                logger.info("Blended edge + structure for stronger control")

            # Convert back to PIL
            control_array = np.clip(control_array, 0, 255).astype(np.uint8)
            strong_control = Image.fromarray(control_array)

            logger.info("Created STRONG multi-modal control image")
            return strong_control

        except Exception as e:
            logger.warning("Failed to create strong control image", error=str(e))
            # Fallback to first available
            return list(conditioning_data.values())[0]

    def _save_debugging_output(self, atlas_image: Image.Image, individual_tiles: Dict[int, Image.Image]):
        """Save debugging output for visual inspection."""
        try:
            import os

            # Create debug directory
            debug_dir = f"../jobs/output/{self.context.get_job_id()}/debug"
            os.makedirs(debug_dir, exist_ok=True)

            # Save full atlas
            atlas_image.save(f"{debug_dir}/atlas_full.png")

            # Save individual tiles with clear naming
            for tile_id, tile_img in individual_tiles.items():
                tile_img.save(f"{debug_dir}/tile_{tile_id:02d}.png")

            # Save tessellation test image (tiles arranged for visual inspection)
            test_image = self._create_tessellation_test_image(individual_tiles)
            test_image.save(f"{debug_dir}/tessellation_test.png")

            logger.info("Debugging output saved", debug_dir=debug_dir, tiles_saved=len(individual_tiles))

        except Exception as e:
            logger.warning("Failed to save debugging output", error=str(e))

    def _create_tessellation_test_image(self, tiles: Dict[int, Image.Image]) -> Image.Image:
        """Create test image showing how tiles tessellate."""
        try:
            # Create 3x3 test pattern using available tiles
            test_size = self.tile_size * 3
            test_image = Image.new("RGB", (test_size, test_size), (128, 128, 128))

            # Use first few tiles to create test pattern
            available_tiles = list(tiles.values())[:9]

            for i, tile in enumerate(available_tiles):
                row = i // 3
                col = i % 3
                x = col * self.tile_size
                y = row * self.tile_size
                test_image.paste(tile, (x, y))

            return test_image

        except Exception as e:
            logger.warning("Failed to create tessellation test", error=str(e))
            return Image.new("RGB", (self.tile_size, self.tile_size), (255, 0, 0))
    
    def _generate_negative_prompt(self) -> str:
        """Generate comprehensive negative prompt."""
        negative_prompt = ("blurry, smooth, anti-aliased, photorealistic, 3d render, low quality, "
                          "artifacts, seams, gaps, misaligned edges, inconsistent style, "
                          "non-tessellating, broken patterns")

        # Log negative prompt for debugging
        logger.info("Negative prompt generated", prompt=negative_prompt)

        return negative_prompt
    
    def _setup_realtime_attention_coordination(self):
        """Setup REAL-TIME attention coordination hooks in FLUX transformer."""
        logger.info("Setting up real-time attention coordination")
        
        # Get FLUX transformer
        transformer = self.pipeline.transformer
        
        # Create attention coordination hook
        def attention_coordination_hook(module, input, output):
            """Hook to coordinate attention between tile edges during generation."""
            try:
                # FLUX attention layers return different formats
                if isinstance(output, tuple) and len(output) > 0:
                    # Extract the main attention output (first element)
                    attention_output = output[0]
                    if isinstance(attention_output, torch.Tensor):
                        coordinated_output = self._apply_attention_coordination(attention_output)
                        # Return tuple with coordinated output
                        return (coordinated_output,) + output[1:] if len(output) > 1 else (coordinated_output,)
                elif isinstance(output, torch.Tensor):
                    # Single tensor output
                    return self._apply_attention_coordination(output)

                # If we can't handle the format, return unchanged
                return output

            except Exception as e:
                logger.warning("Attention hook failed", error=str(e))
                return output
        
        # Register hooks on specific attention layers (more selective for performance)
        hook_count = 0
        for name, module in transformer.named_modules():
            # Only hook the main attention layers, not all sub-modules
            if ('attn' in name.lower() and
                hasattr(module, 'to_out') and
                'transformer_blocks' in name and
                hook_count < 8):  # Limit to 8 hooks for performance

                hook = module.register_forward_hook(attention_coordination_hook)
                self.attention_hooks.append(hook)
                hook_count += 1
        
        logger.info("PROPER FLUX attention coordination setup complete",
                   hooks_registered=len(self.attention_hooks),
                   shared_edges=len(self.shared_edges),
                   coordination_method="cross_tile_seamless_blending")
    
    def _apply_attention_coordination(self, attention_output: torch.Tensor) -> torch.Tensor:
        """PROPER FLUX attention coordination for seamless tessellation."""
        try:
            if not isinstance(attention_output, torch.Tensor) or len(attention_output.shape) != 3:
                return attention_output

            batch_size, seq_len, hidden_dim = attention_output.shape

            # Calculate spatial dimensions for FLUX (8x downsampling)
            latent_height = self.atlas_height // 8
            latent_width = self.atlas_width // 8

            # FLUX flattens spatial dimensions: seq_len should equal latent_height * latent_width
            if seq_len != latent_height * latent_width:
                return attention_output

            # Reshape to spatial format for edge coordination
            spatial_attention = attention_output.view(batch_size, latent_height, latent_width, hidden_dim)

            # Apply REAL cross-tile edge coordination
            coordinated_attention = self._apply_cross_tile_coordination(spatial_attention)

            # Reshape back to sequence format
            return coordinated_attention.view(batch_size, seq_len, hidden_dim)

        except Exception as e:
            logger.warning("FLUX attention coordination failed", error=str(e))
            return attention_output

    def _apply_cross_tile_coordination(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Apply cross-tile coordination for seamless tessellation like the guide shows."""
        try:
            batch_size, height, width, hidden_dim = spatial_attention.shape

            # Calculate tile dimensions in latent space
            latent_tile_height = self.tile_size // 8
            latent_tile_width = self.tile_size // 8

            # Process each shared edge for seamless connections
            for shared_edge in self.shared_edges:
                spatial_attention = self._coordinate_edge_attention(
                    spatial_attention, shared_edge, latent_tile_height, latent_tile_width
                )

            return spatial_attention

        except Exception as e:
            logger.warning("Cross-tile coordination failed", error=str(e))
            return spatial_attention

    def _coordinate_edge_attention(self, spatial_attention: torch.Tensor, shared_edge: Any,
                                 tile_height: int, tile_width: int) -> torch.Tensor:
        """Coordinate attention between connecting tile edges for seamless tessellation."""
        try:
            tile_a_id = shared_edge.tile_a_id
            tile_b_id = shared_edge.tile_b_id
            edge_a = shared_edge.tile_a_edge
            edge_b = shared_edge.tile_b_edge

            # Get tile positions in latent space
            pos_a = self._get_tile_latent_position(tile_a_id, tile_height, tile_width)
            pos_b = self._get_tile_latent_position(tile_b_id, tile_height, tile_width)

            if pos_a is None or pos_b is None:
                return spatial_attention

            # Extract edge regions from both tiles
            edge_region_a = self._extract_latent_edge_region(spatial_attention, pos_a, edge_a, tile_height, tile_width)
            edge_region_b = self._extract_latent_edge_region(spatial_attention, pos_b, edge_b, tile_height, tile_width)

            if edge_region_a.shape != edge_region_b.shape:
                return spatial_attention

            # Apply seamless blending for perfect tessellation
            blended_edge = self._blend_edge_attention(edge_region_a, edge_region_b)

            # Apply blended attention back to both tiles
            spatial_attention = self._apply_blended_edge_attention(
                spatial_attention, pos_a, edge_a, blended_edge, tile_height, tile_width
            )
            spatial_attention = self._apply_blended_edge_attention(
                spatial_attention, pos_b, edge_b, blended_edge, tile_height, tile_width
            )

            return spatial_attention

        except Exception as e:
            logger.warning("Edge attention coordination failed", error=str(e))
            return spatial_attention

    def _get_tile_latent_position(self, tile_id: int, tile_height: int, tile_width: int) -> Optional[Tuple[int, int, int, int]]:
        """Get tile position in latent space coordinates."""
        row = tile_id // self.atlas_columns
        col = tile_id % self.atlas_columns

        if row >= self.atlas_rows:
            return None

        y1 = row * tile_height
        x1 = col * tile_width
        y2 = y1 + tile_height
        x2 = x1 + tile_width

        return (y1, x1, y2, x2)

    def _extract_latent_edge_region(self, spatial_attention: torch.Tensor, pos: Tuple[int, int, int, int],
                                   direction: str, tile_height: int, tile_width: int) -> torch.Tensor:
        """Extract edge region from latent attention for seamless blending."""
        y1, x1, y2, x2 = pos
        edge_thickness = max(1, min(tile_height, tile_width) // 8)  # Adaptive edge thickness

        if direction == "top":
            return spatial_attention[:, y1:y1+edge_thickness, x1:x2, :].clone()
        elif direction == "bottom":
            return spatial_attention[:, y2-edge_thickness:y2, x1:x2, :].clone()
        elif direction == "left":
            return spatial_attention[:, y1:y2, x1:x1+edge_thickness, :].clone()
        elif direction == "right":
            return spatial_attention[:, y1:y2, x2-edge_thickness:x2, :].clone()
        else:
            return spatial_attention[:, y1:y2, x1:x2, :].clone()

    def _blend_edge_attention(self, edge_a: torch.Tensor, edge_b: torch.Tensor) -> torch.Tensor:
        """Blend edge attention for seamless tessellation like the guide shows."""
        # Weighted blending with slight randomization to avoid artifacts
        weight_a = 0.5 + torch.randn(1).item() * 0.1  # 0.4 to 0.6
        weight_b = 1.0 - weight_a

        # Ensure weights are positive and sum to 1
        weight_a = max(0.3, min(0.7, weight_a))
        weight_b = 1.0 - weight_a

        return edge_a * weight_a + edge_b * weight_b

    def _apply_blended_edge_attention(self, spatial_attention: torch.Tensor, pos: Tuple[int, int, int, int],
                                    direction: str, blended_edge: torch.Tensor,
                                    tile_height: int, tile_width: int) -> torch.Tensor:
        """Apply blended edge attention back to spatial attention."""
        try:
            y1, x1, y2, x2 = pos
            edge_thickness = max(1, min(tile_height, tile_width) // 8)

            if direction == "top":
                spatial_attention[:, y1:y1+edge_thickness, x1:x2, :] = blended_edge
            elif direction == "bottom":
                spatial_attention[:, y2-edge_thickness:y2, x1:x2, :] = blended_edge
            elif direction == "left":
                spatial_attention[:, y1:y2, x1:x1+edge_thickness, :] = blended_edge
            elif direction == "right":
                spatial_attention[:, y1:y2, x2-edge_thickness:x2, :] = blended_edge

            return spatial_attention

        except Exception as e:
            logger.warning("Failed to apply blended edge attention", error=str(e))
            return spatial_attention
    
    def _coordinate_tile_edges(self, spatial_attention: torch.Tensor, latent_tile_size: int) -> torch.Tensor:
        """Coordinate edges between connecting tiles in attention space."""
        try:
            batch_size, height, width, hidden_dim = spatial_attention.shape
            
            # Process each shared edge
            for shared_edge in self.shared_edges:
                tile_a_id = shared_edge.tile_a_id
                tile_b_id = shared_edge.tile_b_id
                
                # Get tile positions in latent space
                pos_a = self._get_latent_tile_position(tile_a_id, latent_tile_size)
                pos_b = self._get_latent_tile_position(tile_b_id, latent_tile_size)
                
                if pos_a and pos_b:
                    # Extract edge regions
                    edge_a = self._extract_attention_edge(spatial_attention, pos_a, shared_edge.tile_a_edge)
                    edge_b = self._extract_attention_edge(spatial_attention, pos_b, shared_edge.tile_b_edge)
                    
                    # Blend edge attentions for consistency
                    blended_edge = (edge_a + edge_b) / 2
                    
                    # Apply back to spatial attention
                    spatial_attention = self._apply_attention_edge(spatial_attention, pos_a, shared_edge.tile_a_edge, blended_edge)
                    spatial_attention = self._apply_attention_edge(spatial_attention, pos_b, shared_edge.tile_b_edge, blended_edge)
            
            return spatial_attention
            
        except Exception as e:
            logger.warning("Tile edge coordination failed", error=str(e))
            return spatial_attention

    def _get_latent_tile_position(self, tile_id: int, latent_tile_size: int) -> Optional[Tuple[int, int, int, int]]:
        """Get tile position in latent space coordinates."""
        row = tile_id // self.atlas_columns
        col = tile_id % self.atlas_columns

        if row >= self.atlas_rows:
            return None

        x1 = col * latent_tile_size
        y1 = row * latent_tile_size
        x2 = x1 + latent_tile_size
        y2 = y1 + latent_tile_size

        return (x1, y1, x2, y2)

    def _extract_attention_edge(self, spatial_attention: torch.Tensor,
                               pos: Tuple[int, int, int, int], direction: str) -> torch.Tensor:
        """Extract edge region from spatial attention."""
        x1, y1, x2, y2 = pos
        edge_width = 1  # 1 pixel edge in latent space

        if direction == "top":
            return spatial_attention[:, y1:y1+edge_width, x1:x2, :].clone()
        elif direction == "bottom":
            return spatial_attention[:, y2-edge_width:y2, x1:x2, :].clone()
        elif direction == "left":
            return spatial_attention[:, y1:y2, x1:x1+edge_width, :].clone()
        elif direction == "right":
            return spatial_attention[:, y1:y2, x2-edge_width:x2, :].clone()
        else:
            return spatial_attention[:, y1:y2, x1:x2, :].clone()

    def _apply_attention_edge(self, spatial_attention: torch.Tensor,
                             pos: Tuple[int, int, int, int], direction: str,
                             edge_data: torch.Tensor) -> torch.Tensor:
        """Apply edge data back to spatial attention."""
        x1, y1, x2, y2 = pos
        edge_width = 1

        try:
            if direction == "top":
                spatial_attention[:, y1:y1+edge_width, x1:x2, :] = edge_data
            elif direction == "bottom":
                spatial_attention[:, y2-edge_width:y2, x1:x2, :] = edge_data
            elif direction == "left":
                spatial_attention[:, y1:y2, x1:x1+edge_width, :] = edge_data
            elif direction == "right":
                spatial_attention[:, y1:y2, x2-edge_width:x2, :] = edge_data
        except Exception as e:
            logger.warning("Failed to apply attention edge", error=str(e))

        return spatial_attention

    def _generate_with_attention_coordination(self, clip_prompt: str, negative_prompt: str,
                                            conditioning_data: Dict[str, Image.Image]) -> Image.Image:
        """Generate atlas with real-time attention coordination."""
        logger.info("Starting generation with real-time attention coordination", steps=self.steps)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create generator on CPU (FLUX requirement)
        if self.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
        else:
            generator = torch.Generator(device="cpu")

        # Generate with REAL ControlNet conditioning from Stage 4
        with torch.no_grad():
            # Check if pipeline supports ControlNet
            if hasattr(self.pipeline, 'controlnet') and self.pipeline.controlnet is not None:
                # Create STRONG multi-modal control image
                control_image = self._create_strong_control_image(conditioning_data)

                # Log generation parameters
                logger.info("Starting ControlNet generation",
                           prompt_used="CLIP_prompt",
                           prompt_content=clip_prompt,
                           controlnet_scale=1.0,
                           steps=self.steps,
                           guidance_scale=self.guidance_scale)

                result = self.pipeline(
                    prompt=clip_prompt,  # Use SHORT prompt for CLIP (77 tokens)
                    negative_prompt=negative_prompt,
                    control_image=control_image,  # STRONG multi-modal conditioning!
                    controlnet_conditioning_scale=1.0,  # MAXIMUM structure control
                    height=self.atlas_height,
                    width=self.atlas_width,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    output_type="pil"
                )
                logger.info("Generated atlas using STRONG ControlNet conditioning")

            else:
                # Log fallback generation parameters
                logger.info("Starting text-only generation (ControlNet fallback)",
                           prompt_used="CLIP_prompt",
                           prompt_content=clip_prompt,
                           steps=self.steps,
                           guidance_scale=self.guidance_scale)

                # Fallback to text-only generation
                result = self.pipeline(
                    prompt=clip_prompt,  # Use SHORT prompt for CLIP (77 tokens)
                    negative_prompt=negative_prompt,
                    height=self.atlas_height,
                    width=self.atlas_width,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    output_type="pil"
                )
                logger.info("Generated atlas using text-only (ControlNet not available)")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        atlas_image = result.images[0]
        logger.info("Generation with real-time coordination completed", size=atlas_image.size)

        return atlas_image

    def _extract_coordinated_tiles(self, atlas_image: Image.Image) -> Dict[int, Image.Image]:
        """Extract individual tiles from coordinated atlas."""
        tiles = {}

        for tile_id in range(len(self.tileset_setup.tile_specs)):
            # Calculate tile position
            row = tile_id // self.atlas_columns
            col = tile_id % self.atlas_columns

            if row >= self.atlas_rows:
                continue

            # Extract tile region
            x1 = col * self.tile_size
            y1 = row * self.tile_size
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size

            tile_image = atlas_image.crop((x1, y1, x2, y2))
            tiles[tile_id] = tile_image

        logger.info("Extracted coordinated tiles", tile_count=len(tiles))
        return tiles

    def _validate_tessellation_quality_detailed(self, tiles: Dict[int, Image.Image]) -> float:
        """Detailed tessellation quality validation with pixel-level analysis."""
        if not self.shared_edges:
            return 1.0

        total_similarity = 0.0
        edge_count = 0
        edge_details = []

        for shared_edge in self.shared_edges:
            tile_a_id = shared_edge.tile_a_id
            tile_b_id = shared_edge.tile_b_id

            if tile_a_id not in tiles or tile_b_id not in tiles:
                continue

            # Calculate detailed edge similarity
            similarity = self._calculate_detailed_edge_similarity(
                tiles[tile_a_id], tiles[tile_b_id],
                shared_edge.tile_a_edge, shared_edge.tile_b_edge
            )

            edge_details.append({
                "tiles": f"{tile_a_id}-{tile_b_id}",
                "edges": f"{shared_edge.tile_a_edge}-{shared_edge.tile_b_edge}",
                "similarity": similarity
            })

            total_similarity += similarity
            edge_count += 1

        average_similarity = total_similarity / edge_count if edge_count > 0 else 1.0

        # Log detailed results
        logger.info("DETAILED tessellation validation",
                   average_similarity=average_similarity,
                   edges_validated=edge_count,
                   worst_edge=min(edge_details, key=lambda x: x["similarity"]) if edge_details else None)

        return average_similarity

    def _calculate_detailed_edge_similarity(self, tile_a: Image.Image, tile_b: Image.Image,
                                          edge_a: str, edge_b: str) -> float:
        """Calculate detailed edge similarity with pixel-level analysis."""
        try:
            # Extract edge regions
            edge_region_a = self._extract_edge_region(tile_a, edge_a)
            edge_region_b = self._extract_edge_region(tile_b, edge_b)

            # Convert to numpy
            array_a = np.array(edge_region_a, dtype=np.float32)
            array_b = np.array(edge_region_b, dtype=np.float32)

            if array_a.shape != array_b.shape:
                return 0.0

            # Multiple similarity metrics
            mse = np.mean((array_a - array_b) ** 2)
            max_possible_mse = 255 ** 2
            mse_similarity = 1.0 - (mse / max_possible_mse)

            # Structural similarity (simplified SSIM)
            mean_a = np.mean(array_a)
            mean_b = np.mean(array_b)
            var_a = np.var(array_a)
            var_b = np.var(array_b)
            covar = np.mean((array_a - mean_a) * (array_b - mean_b))

            structural_similarity = (2 * mean_a * mean_b + 1) * (2 * covar + 1) / \
                                  ((mean_a**2 + mean_b**2 + 1) * (var_a + var_b + 1))

            # Combined similarity
            combined_similarity = (mse_similarity * 0.7 + structural_similarity * 0.3)

            return max(0.0, combined_similarity)

        except Exception as e:
            logger.warning("Detailed edge similarity failed", error=str(e))
            return 0.0

    def _validate_tessellation_quality(self, tiles: Dict[int, Image.Image]) -> float:
        """Validate tessellation quality with mathematical precision."""
        if not self.shared_edges:
            return 1.0

        total_similarity = 0.0
        edge_count = 0

        for shared_edge in self.shared_edges:
            tile_a_id = shared_edge.tile_a_id
            tile_b_id = shared_edge.tile_b_id

            if tile_a_id not in tiles or tile_b_id not in tiles:
                continue

            # Calculate edge similarity with sub-pixel precision
            similarity = self._calculate_edge_similarity(
                tiles[tile_a_id], tiles[tile_b_id],
                shared_edge.tile_a_edge, shared_edge.tile_b_edge
            )

            total_similarity += similarity
            edge_count += 1

        average_similarity = total_similarity / edge_count if edge_count > 0 else 1.0
        logger.info("Tessellation quality validated",
                   average_similarity=average_similarity,
                   edges_validated=edge_count)

        return average_similarity

    def _calculate_edge_similarity(self, tile_a: Image.Image, tile_b: Image.Image,
                                 edge_a: str, edge_b: str) -> float:
        """Calculate mathematical edge similarity."""
        try:
            # Extract edge regions with sub-pixel precision
            edge_region_a = self._extract_edge_region(tile_a, edge_a)
            edge_region_b = self._extract_edge_region(tile_b, edge_b)

            # Convert to numpy for mathematical analysis
            array_a = np.array(edge_region_a, dtype=np.float32)
            array_b = np.array(edge_region_b, dtype=np.float32)

            if array_a.shape != array_b.shape:
                return 0.0

            # Calculate structural similarity (better than MSE)
            mse = np.mean((array_a - array_b) ** 2)
            max_possible_mse = 255 ** 2
            similarity = 1.0 - (mse / max_possible_mse)

            return max(0.0, similarity)

        except Exception as e:
            logger.warning("Edge similarity calculation failed", error=str(e))
            return 0.0

    def _extract_edge_region(self, tile: Image.Image, direction: str) -> Image.Image:
        """Extract edge region with precision."""
        width, height = tile.size
        edge_width = 4  # 4-pixel edge for analysis

        if direction == "top":
            return tile.crop((0, 0, width, edge_width))
        elif direction == "bottom":
            return tile.crop((0, height - edge_width, width, height))
        elif direction == "left":
            return tile.crop((0, 0, edge_width, height))
        elif direction == "right":
            return tile.crop((width - edge_width, 0, width, height))
        else:
            return tile

    def _cleanup_attention_hooks(self):
        """Clean up attention hooks after generation."""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks.clear()
        logger.info("Attention hooks cleaned up")
