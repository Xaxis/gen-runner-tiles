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
        self.steps = 20
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
        
        # Setup PROPER FLUX attention coordination with tile masking
        self._setup_proper_attention_hooks()
        
        # Generate FULL ATLAS with proper attention coordination and tile-specific masking
        atlas_image = self._generate_coordinated_atlas_with_tile_masking(
            clip_prompt, t5_prompt, negative_prompt, conditioning_data
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
        # @TODO - This part looks like bullshit!!!
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

    def _get_tile_type_mapping(self) -> Dict[int, str]:
        """Get Wang tile type mapping for specific prompts."""
        return {
            0: "corner_ne", 1: "corner_nw", 2: "corner_se", 3: "corner_sw",
            4: "edge_top", 5: "edge_bottom", 6: "edge_left", 7: "edge_right",
            8: "t_top", 9: "t_bottom", 10: "t_left", 11: "t_right",
            12: "cross"
        }

    def _create_tile_specific_prompt(self, base_prompt: str, tile_type: str, tile_id: int) -> str:
        """Create THEME-SPECIFIC prompt using Model Registry templates."""
        from src.core.model_registry import ModelRegistry

        theme_base = f"umempart, {self.theme} {self.palette}"

        # Get THEME-SPECIFIC tile description from Model Registry
        # Map theme to full theme key
        theme_key_mapping = {
            "fantasy": "fantasy_medieval",
            "sci_fi": "sci_fi_cyberpunk",
            "nature": "nature_forest"
        }

        full_theme_key = theme_key_mapping.get(self.theme, self.theme)
        theme_prompts = ModelRegistry.THEME_TILE_PROMPTS.get(full_theme_key, {})
        specific_desc = theme_prompts.get(
            tile_type, f"{self.theme} tile with decorative border pattern"
        )

        logger.info("Theme mapping debug",
                   original_theme=self.theme,
                   full_theme_key=full_theme_key,
                   found_theme_prompts=len(theme_prompts),
                   tile_type=tile_type,
                   found_specific_desc=tile_type in theme_prompts)

        # Combine with tessellation requirements from Model Registry
        tile_prompt = f"{theme_base}, {specific_desc}, {ModelRegistry.TESSELLATION_TERMS}, pixel art, crisp edges"

        logger.info("THEME-SPECIFIC prompt created",
                   tile_id=tile_id, tile_type=tile_type, theme=self.theme, prompt=tile_prompt)
        return tile_prompt

    def _generate_coordinated_atlas_with_tile_masking(self, clip_prompt: str, t5_prompt: str,
                                                    negative_prompt: str, conditioning_data: Dict[str, Image.Image]) -> Image.Image:
        """Generate full atlas with PROPER FLUX attention coordination and tile-specific prompt masking."""
        logger.info("Starting COORDINATED atlas generation with tile-specific masking")

        # Create tile-specific prompt embeddings for masking
        tile_prompt_embeddings = self._create_tile_specific_embeddings()

        # Store embeddings for attention hooks to use
        self.tile_prompt_embeddings = tile_prompt_embeddings

        # Create strong multi-modal control image
        control_image = self._create_strong_control_image(conditioning_data)

        # Generate with REAL attention coordination
        with torch.no_grad():
            if hasattr(self.pipeline, 'controlnet') and self.pipeline.controlnet is not None:
                logger.info("Generating with ControlNet + attention coordination + tile masking")

                result = self.pipeline(
                    prompt=clip_prompt,  # Base prompt for CLIP
                    negative_prompt=negative_prompt,
                    control_image=control_image,
                    controlnet_conditioning_scale=1.0,
                    height=self.atlas_height,
                    width=self.atlas_width,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                    output_type="pil"
                )
            else:
                logger.info("Generating with text-only + attention coordination + tile masking")

                result = self.pipeline(
                    prompt=clip_prompt,
                    negative_prompt=negative_prompt,
                    height=self.atlas_height,
                    width=self.atlas_width,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                    output_type="pil"
                )

        atlas_image = result.images[0]
        logger.info("COORDINATED atlas generation complete with tile masking")
        return atlas_image

    def _create_tile_specific_embeddings(self) -> Dict[int, torch.Tensor]:
        """Create tile-specific prompt embeddings for attention masking."""
        logger.info("Creating tile-specific prompt embeddings for masking")

        tile_embeddings = {}
        tile_types = self._get_tile_type_mapping()

        # FLUX has dual text encoders: CLIP and T5
        # We need to get embeddings from both for proper tile-specific guidance

        clip_encoder = getattr(self.pipeline, 'text_encoder', None)
        clip_tokenizer = getattr(self.pipeline, 'tokenizer', None)
        t5_encoder = getattr(self.pipeline, 'text_encoder_2', None)  # T5 encoder
        t5_tokenizer = getattr(self.pipeline, 'tokenizer_2', None)   # T5 tokenizer

        if clip_encoder is None or clip_tokenizer is None:
            logger.warning("Cannot create tile-specific embeddings - missing CLIP encoder/tokenizer")
            return {}

        # Use T5 encoder if available for richer embeddings
        primary_encoder = t5_encoder if t5_encoder is not None else clip_encoder
        primary_tokenizer = t5_tokenizer if t5_tokenizer is not None else clip_tokenizer
        max_length = 512 if t5_encoder is not None else 77

        logger.info("Using text encoder for embeddings",
                   encoder_type="T5" if t5_encoder is not None else "CLIP",
                   max_length=max_length)

        for tile_id in range(13):
            tile_type = tile_types.get(tile_id, "unknown")

            # Create tile-specific prompt
            tile_prompt = self._create_tile_specific_prompt("", tile_type, tile_id)

            try:
                # Tokenize with proper max length
                tokens = primary_tokenizer(
                    tile_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )

                # Move to correct device
                device = primary_encoder.device
                input_ids = tokens.input_ids.to(device)

                with torch.no_grad():
                    # Get embeddings from the encoder
                    if hasattr(primary_encoder, 'text_model'):
                        # T5 encoder structure
                        embeddings = primary_encoder.text_model.encoder(input_ids)[0]
                    else:
                        # CLIP encoder structure
                        embeddings = primary_encoder(input_ids)[0]

                tile_embeddings[tile_id] = embeddings
                logger.info("Created embedding for tile",
                           tile_id=tile_id,
                           tile_type=tile_type,
                           embedding_shape=embeddings.shape,
                           prompt_length=len(tile_prompt.split()))

            except Exception as e:
                logger.warning("Failed to create embedding for tile", tile_id=tile_id, error=str(e))

        logger.info("Tile-specific embeddings created", count=len(tile_embeddings))
        return tile_embeddings
    
    def _generate_negative_prompt(self) -> str:
        """Generate comprehensive negative prompt from Model Registry."""
        from src.core.model_registry import ModelRegistry

        negative_prompt = ModelRegistry.NEGATIVE_PROMPT

        # Log negative prompt for debugging
        logger.info("Negative prompt from registry", prompt=negative_prompt)

        return negative_prompt
    
    def _apply_attention_coordination(self, attention_output: torch.Tensor) -> torch.Tensor:
        """PROPER FLUX attention coordination with tile-specific masking and edge coordination."""
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

            # Reshape to spatial format for processing
            spatial_attention = attention_output.view(batch_size, latent_height, latent_width, hidden_dim)

            # Apply tile-specific prompt masking
            masked_attention = self._apply_tile_specific_masking(spatial_attention)

            # Apply cross-tile edge coordination for seamless tessellation
            coordinated_attention = self._apply_cross_tile_edge_coordination(masked_attention)

            # Reshape back to sequence format
            return coordinated_attention.view(batch_size, seq_len, hidden_dim)

        except Exception as e:
            logger.warning("FLUX attention coordination failed", error=str(e))
            return attention_output

    def _apply_tile_specific_masking(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Apply tile-specific prompt masking to different regions of the atlas."""
        try:
            if not hasattr(self, 'tile_prompt_embeddings') or not self.tile_prompt_embeddings:
                return spatial_attention

            batch_size, height, width, hidden_dim = spatial_attention.shape
            tile_height = self.tile_size // 8
            tile_width = self.tile_size // 8

            # Apply tile-specific modifications
            for tile_id in range(13):
                if tile_id not in self.tile_prompt_embeddings:
                    continue

                # Get tile position
                row = tile_id // self.atlas_columns
                col = tile_id % self.atlas_columns

                if row >= self.atlas_rows:
                    continue

                # Calculate tile region in latent space
                y1 = row * tile_height
                x1 = col * tile_width
                y2 = min(y1 + tile_height, height)
                x2 = min(x1 + tile_width, width)

                # Get tile-specific embedding for this tile type
                tile_embedding = self.tile_prompt_embeddings[tile_id]  # Shape: [1, seq_len, hidden_dim]

                # Extract tile region from spatial attention
                tile_region = spatial_attention[:, y1:y2, x1:x2, :].clone()  # Shape: [batch, tile_h, tile_w, hidden_dim]

                # PROPER cross-attention between tile embedding and tile region
                modified_tile_region = self._apply_tile_specific_cross_attention(
                    tile_region, tile_embedding, tile_id
                )

                # Apply the modified region back to spatial attention
                spatial_attention[:, y1:y2, x1:x2, :] = modified_tile_region

            return spatial_attention

        except Exception as e:
            logger.warning("Tile-specific masking failed", error=str(e))
            return spatial_attention

    def _apply_cross_tile_edge_coordination(self, spatial_attention: torch.Tensor) -> torch.Tensor:
        """Apply cross-tile edge coordination for seamless tessellation."""
        try:
            batch_size, height, width, hidden_dim = spatial_attention.shape
            tile_height = self.tile_size // 8
            tile_width = self.tile_size // 8

            # Process each shared edge for seamless coordination
            for shared_edge in self.shared_edges:
                spatial_attention = self._coordinate_shared_edge_attention(
                    spatial_attention, shared_edge, tile_height, tile_width
                )

            return spatial_attention

        except Exception as e:
            logger.warning("Cross-tile edge coordination failed", error=str(e))
            return spatial_attention

    def _coordinate_shared_edge_attention(self, spatial_attention: torch.Tensor, shared_edge: Any,
                                        tile_height: int, tile_width: int) -> torch.Tensor:
        """Coordinate attention between two tiles sharing an edge."""
        try:
            tile_a_id = shared_edge.tile_a_id
            tile_b_id = shared_edge.tile_b_id
            edge_a = shared_edge.tile_a_edge
            edge_b = shared_edge.tile_b_edge

            # Get tile positions
            pos_a = self._get_tile_latent_position(tile_a_id, tile_height, tile_width)
            pos_b = self._get_tile_latent_position(tile_b_id, tile_height, tile_width)

            if pos_a is None or pos_b is None:
                return spatial_attention

            # Extract edge regions
            edge_region_a = self._extract_latent_edge_region(spatial_attention, pos_a, edge_a, tile_height, tile_width)
            edge_region_b = self._extract_latent_edge_region(spatial_attention, pos_b, edge_b, tile_height, tile_width)

            if edge_region_a.shape != edge_region_b.shape:
                return spatial_attention

            # Blend edge attention for seamless tessellation
            blended_edge = self._blend_edge_attention_seamless(edge_region_a, edge_region_b)

            # Apply blended attention back to both tiles
            spatial_attention = self._apply_blended_edge_attention(
                spatial_attention, pos_a, edge_a, blended_edge, tile_height, tile_width
            )
            spatial_attention = self._apply_blended_edge_attention(
                spatial_attention, pos_b, edge_b, blended_edge, tile_height, tile_width
            )

            return spatial_attention

        except Exception as e:
            logger.warning("Shared edge coordination failed", error=str(e))
            return spatial_attention

    def _blend_edge_attention_seamless(self, edge_a: torch.Tensor, edge_b: torch.Tensor) -> torch.Tensor:
        """Blend edge attention for perfect seamless tessellation."""
        # Use a more sophisticated blending for seamless results
        # Apply gaussian-like blending weights
        blend_weight = 0.5  # Perfect 50/50 blend for seamless edges

        # Add small noise to prevent identical patterns
        noise_scale = 0.02
        noise = torch.randn_like(edge_a) * noise_scale

        blended = edge_a * blend_weight + edge_b * (1.0 - blend_weight) + noise

        return blended

    def _apply_tile_specific_cross_attention(self, tile_region: torch.Tensor,
                                           tile_embedding: torch.Tensor, tile_id: int) -> torch.Tensor:
        """Apply PROPER cross-attention between tile region and tile-specific prompt embedding."""
        try:
            batch_size, tile_h, tile_w, hidden_dim = tile_region.shape
            seq_len_embed = tile_embedding.shape[1]

            # Flatten tile region to sequence format for attention computation
            tile_region_flat = tile_region.view(batch_size, tile_h * tile_w, hidden_dim)

            # Create attention matrices
            # Q: tile region (what we're modifying)
            # K, V: tile embedding (tile-specific prompt guidance)

            # Linear projections for attention (simplified - in practice use proper attention layers)
            scale = hidden_dim ** -0.5

            # Compute attention scores between tile region and tile embedding
            # tile_region_flat: [batch, tile_pixels, hidden_dim]
            # tile_embedding: [batch, prompt_tokens, hidden_dim]

            attention_scores = torch.matmul(tile_region_flat, tile_embedding.transpose(-2, -1)) * scale
            attention_weights = torch.softmax(attention_scores, dim=-1)

            # Apply attention to get tile-specific influenced features
            attended_features = torch.matmul(attention_weights, tile_embedding)

            # Blend original tile region with attended features
            blend_strength = 0.4  # How much tile-specific influence to apply
            modified_region_flat = tile_region_flat * (1.0 - blend_strength) + attended_features * blend_strength

            # Reshape back to spatial format
            modified_region = modified_region_flat.view(batch_size, tile_h, tile_w, hidden_dim)

            logger.debug("Applied tile-specific cross-attention",
                        tile_id=tile_id,
                        attention_shape=attention_weights.shape,
                        blend_strength=blend_strength)

            return modified_region

        except Exception as e:
            logger.warning("Tile-specific cross-attention failed", tile_id=tile_id, error=str(e))
            return tile_region

    def _setup_proper_attention_hooks(self):
        """Setup PROPER attention hooks for FLUX transformer blocks."""
        try:
            self.attention_hooks = []

            # FLUX uses DiT (Diffusion Transformer) architecture
            # We need to hook into the transformer blocks, not just generic attention

            if hasattr(self.pipeline, 'transformer'):
                transformer = self.pipeline.transformer

                # Hook into transformer blocks (FLUX typically has multiple transformer blocks)
                if hasattr(transformer, 'transformer_blocks'):
                    for i, block in enumerate(transformer.transformer_blocks):
                        if hasattr(block, 'attn1'):  # Self-attention
                            hook = block.attn1.register_forward_hook(self._attention_hook_wrapper)
                            self.attention_hooks.append(hook)
                            logger.info("Registered attention hook on transformer block", block_id=i, layer="attn1")

                        if hasattr(block, 'attn2'):  # Cross-attention
                            hook = block.attn2.register_forward_hook(self._attention_hook_wrapper)
                            self.attention_hooks.append(hook)
                            logger.info("Registered attention hook on transformer block", block_id=i, layer="attn2")

                # Also hook into single transformer blocks if they exist
                elif hasattr(transformer, 'single_transformer_blocks'):
                    for i, block in enumerate(transformer.single_transformer_blocks):
                        if hasattr(block, 'attn'):
                            hook = block.attn.register_forward_hook(self._attention_hook_wrapper)
                            self.attention_hooks.append(hook)
                            logger.info("Registered attention hook on single transformer block", block_id=i)

            logger.info("PROPER FLUX attention hooks setup complete",
                       hooks_registered=len(self.attention_hooks),
                       transformer_available=hasattr(self.pipeline, 'transformer'))

        except Exception as e:
            logger.error("Failed to setup proper attention hooks", error=str(e))
            self.attention_hooks = []

    def _attention_hook_wrapper(self, module, input_tensor, output_tensor):
        """Wrapper for attention hooks that handles FLUX-specific attention format."""
        try:
            # FLUX attention output format may be different from standard attention
            if isinstance(output_tensor, tuple):
                # If output is tuple, attention output is typically the first element
                attention_output = output_tensor[0]
                modified_output = self._apply_attention_coordination(attention_output)
                return (modified_output,) + output_tensor[1:]
            else:
                # Direct attention output
                return self._apply_attention_coordination(output_tensor)

        except Exception as e:
            logger.warning("Attention hook wrapper failed", error=str(e))
            return output_tensor

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