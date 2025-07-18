"""
Stage 3: Perspective Setup
Sets up tessellation-aware perspective and lighting for consistent tile generation.
Integrates with Stage 2 tessellation data and prepares parameters for Stage 4 references.
"""

import math
from typing import Dict, Any, Tuple
import structlog

from ..core.pipeline_context import PipelineContext

logger = structlog.get_logger()


def execute(context: PipelineContext) -> Dict[str, Any]:
    """Execute perspective setup stage with tessellation integration."""
    try:
        logger.info("Starting perspective setup", job_id=context.get_job_id())
        
        # Validate context state (needs Stages 1-2 complete)
        if not context.validate_context_state(3):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Get required data from context
        if not context.universal_tileset:
            context.add_error(3, "Missing universal tileset from Stage 2")
            return {"success": False, "errors": ["Missing universal tileset from Stage 2"]}
        
        extracted_config = getattr(context, 'extracted_config', {})
        if not extracted_config:
            context.add_error(3, "Missing extracted configuration")
            return {"success": False, "errors": ["Missing extracted configuration"]}
        
        # Create tessellation-aware perspective setup
        perspective_setup = TessellationPerspectiveSetup(
            tileset_setup=context.universal_tileset,
            config=extracted_config
        )
        
        # Generate perspective and lighting configurations
        setup_result = perspective_setup.setup_tessellation_perspective()
        
        # Store in context for Stage 4
        context.perspective_config = setup_result["perspective_config"]
        context.lighting_config = setup_result["lighting_config"]
        context.tessellation_perspective = setup_result["tessellation_perspective"]
        
        # Update context stage
        context.current_stage = 3
        
        logger.info("Perspective setup completed",
                   job_id=context.get_job_id(),
                   camera_elevation=setup_result["perspective_config"]["camera_elevation"],
                   lighting_type=setup_result["lighting_config"]["lighting_type"],
                   view_angle=setup_result["perspective_config"]["view_angle"],
                   sub_tile_precision=setup_result["tessellation_perspective"]["sub_tile_precision"])
        
        return {
            "success": True,
            "data": setup_result
        }
        
    except Exception as e:
        error_msg = f"Perspective setup failed: {str(e)}"
        logger.error("Perspective setup failed", error=error_msg, job_id=context.get_job_id())
        context.add_error(3, error_msg)
        return {"success": False, "errors": [error_msg]}


class TessellationPerspectiveSetup:
    """Sets up perspective and lighting with tessellation awareness."""
    
    def __init__(self, tileset_setup, config: Dict[str, Any]):
        self.tileset_setup = tileset_setup
        self.config = config
        
        # Get tessellation parameters
        self.tile_size = tileset_setup.tile_size
        self.sub_tile_size = tileset_setup.sub_tile_size
        self.sub_tiles_per_row = tileset_setup.sub_tiles_per_row
        self.sub_tiles_per_col = tileset_setup.sub_tiles_per_col
        self.tileset_type = tileset_setup.tileset_type
        
        # Get theme and style preferences from config
        self.theme = config.get("theme", "fantasy")
        self.palette = config.get("palette", "medieval")

        logger.info("Perspective setup initialized",
                   tile_size=self.tile_size,
                   sub_tile_size=self.sub_tile_size,
                   sub_tile_grid=f"{self.sub_tiles_per_row}x{self.sub_tiles_per_col}",
                   tileset_type=self.tileset_type,
                   theme=self.theme)
        
    def setup_tessellation_perspective(self) -> Dict[str, Any]:
        """Set up perspective and lighting with tessellation integration."""
        
        # Generate tessellation-aware perspective configuration
        perspective_config = self._generate_tessellation_perspective()
        
        # Generate structure-aware lighting configuration
        lighting_config = self._generate_structure_lighting()
        
        # Generate per-tile-type perspective adjustments
        tessellation_perspective = self._generate_per_tile_perspective()
        
        # Generate depth and normal calculation parameters
        depth_parameters = self._generate_depth_parameters()
        
        # Generate lighting parameters for each structure type
        structure_lighting = self._generate_structure_lighting_parameters()
        
        return {
            "perspective_config": perspective_config,
            "lighting_config": lighting_config,
            "tessellation_perspective": tessellation_perspective,
            "depth_parameters": depth_parameters,
            "structure_lighting": structure_lighting,
            "setup_summary": {
                "tile_size": self.tile_size,
                "sub_tile_size": self.sub_tile_size,
                "sub_tile_precision": f"{self.sub_tiles_per_row}x{self.sub_tiles_per_col}",
                "perspective_type": "tessellation_aware",
                "lighting_type": lighting_config["lighting_type"]
            }
        }
    
    def _generate_tessellation_perspective(self) -> Dict[str, Any]:
        """Generate perspective configuration optimized for tessellation."""
        
        # Calculate camera parameters based on tile and sub-tile sizes
        # Camera should capture tile with sub-tile precision
        camera_distance = self._calculate_optimal_camera_distance()
        camera_elevation = self._calculate_tessellation_elevation()
        field_of_view = self._calculate_tessellation_fov()
        
        # Perspective should be consistent across all tiles
        perspective_config = {
            "view_angle": "top-down",  # Optimal for tessellation
            "camera_elevation": camera_elevation,
            "camera_distance": camera_distance,
            "field_of_view": field_of_view,
            "projection_type": "orthographic",  # Better for tessellation consistency
            
            # Sub-tile precision parameters
            "sub_tile_precision": True,
            "sub_tile_size": self.sub_tile_size,
            "sub_tiles_per_row": self.sub_tiles_per_row,
            "sub_tiles_per_col": self.sub_tiles_per_col,
            
            # Tessellation-specific parameters
            "edge_precision": self.sub_tile_size,  # Edge matching precision
            "seamless_perspective": True,  # Ensure seamless edge matching
            "tile_boundary_handling": "precise"  # Precise tile boundary alignment
        }
        
        return perspective_config
    
    def _calculate_optimal_camera_distance(self) -> float:
        """Calculate camera distance for optimal tile capture."""
        # Distance should capture entire tile with sub-tile detail
        base_distance = self.tile_size * 2.0
        
        # Adjust for sub-tile precision requirements
        sub_tile_factor = self.tile_size / self.sub_tile_size
        precision_adjustment = 1.0 + (sub_tile_factor * 0.1)
        
        return base_distance * precision_adjustment
    
    def _calculate_tessellation_elevation(self) -> float:
        """Calculate camera elevation for tessellation consistency."""
        # High elevation for top-down view, but not completely orthogonal
        # This allows for some depth perception while maintaining tessellation
        base_elevation = 85.0  # Near top-down but with slight angle
        
        # Adjust based on tileset complexity
        complexity_adjustments = {
            "minimal": 0.0,    # Standard elevation
            "extended": 5.0,   # Slightly higher for more detail
            "full": 10.0       # Higher for complex patterns
        }
        
        adjustment = complexity_adjustments.get(self.tileset_type, 0.0)
        return min(95.0, base_elevation + adjustment)  # Cap at 95 degrees
    
    def _calculate_tessellation_fov(self) -> float:
        """Calculate field of view for tessellation capture."""
        # FOV should capture tile with minimal distortion
        base_fov = 45.0
        
        # Adjust for tile size - larger tiles need wider FOV
        size_factor = self.tile_size / 64.0  # Normalize to 64px baseline
        fov_adjustment = math.log(size_factor) * 5.0 if size_factor > 0 else 0.0
        
        return max(30.0, min(60.0, base_fov + fov_adjustment))  # Clamp to reasonable range
    
    def _generate_structure_lighting(self) -> Dict[str, Any]:
        """Generate lighting configuration aware of structure types."""
        
        # Base lighting configuration
        lighting_config = {
            "lighting_type": "directional",
            "primary_direction": {"x": -0.5, "y": -0.7, "z": 0.5},  # Top-left directional
            "intensity": 0.8,
            "ambient_intensity": 0.3,
            "shadow_softness": 0.4,
            
            # Theme-specific lighting adjustments
            "theme_adjustments": self._get_theme_lighting_adjustments(),
            
            # Sub-tile lighting precision
            "sub_tile_lighting": True,
            "sub_tile_shadow_precision": self.sub_tile_size,
            
            # Structure-aware lighting
            "structure_lighting": True,
            "per_structure_adjustments": True
        }
        
        return lighting_config
    
    def _get_theme_lighting_adjustments(self) -> Dict[str, Any]:
        """Get theme-specific lighting adjustments."""
        theme_lighting = {
            "fantasy": {
                "color_temperature": "warm",
                "base_color": (255, 240, 220),
                "intensity_multiplier": 1.0,
                "shadow_color": (100, 80, 60)
            },
            "sci_fi": {
                "color_temperature": "cool",
                "base_color": (220, 240, 255),
                "intensity_multiplier": 1.2,
                "shadow_color": (60, 80, 100)
            },
            "modern": {
                "color_temperature": "neutral",
                "base_color": (240, 240, 240),
                "intensity_multiplier": 0.9,
                "shadow_color": (80, 80, 80)
            }
        }
        
        return theme_lighting.get(self.theme, theme_lighting["fantasy"])
    
    def _generate_per_tile_perspective(self) -> Dict[str, Any]:
        """Generate perspective adjustments per tile type."""
        
        tessellation_perspective = {
            "sub_tile_precision": f"{self.sub_tiles_per_row}x{self.sub_tiles_per_col}",
            "edge_alignment_precision": self.sub_tile_size,
            
            # Per-tile-type perspective adjustments
            "tile_type_adjustments": {
                "corner": {
                    "depth_emphasis": 1.2,  # Emphasize corner depth
                    "edge_sharpness": 1.1,
                    "shadow_intensity": 1.0
                },
                "edge": {
                    "depth_emphasis": 0.9,  # Flatter for edge tiles
                    "edge_sharpness": 1.2,  # Sharp edges important
                    "shadow_intensity": 0.8
                },
                "t_junction": {
                    "depth_emphasis": 1.0,  # Balanced depth
                    "edge_sharpness": 1.3,  # Very sharp for junctions
                    "shadow_intensity": 1.1
                },
                "cross": {
                    "depth_emphasis": 0.8,  # Flatter center
                    "edge_sharpness": 1.0,
                    "shadow_intensity": 0.7
                }
            },
            
            # Seamless edge requirements
            "seamless_edge_handling": {
                "edge_precision": self.sub_tile_size,
                "overlap_handling": "precise_alignment",
                "pattern_consistency": "enforced"
            }
        }
        
        return tessellation_perspective
    
    def _generate_depth_parameters(self) -> Dict[str, Any]:
        """Generate depth calculation parameters for Stage 4."""
        
        return {
            "depth_range": {
                "min_depth": 0,
                "max_depth": 255,
                "neutral_depth": 128
            },
            
            # Structure-specific depth values
            "structure_depths": {
                "CORNER": 180,      # Raised corners
                "EDGE": 140,        # Medium edges
                "T_JUNCTION": 120,  # Lower junctions
                "CENTER": 100,      # Lowest centers
                "GENERIC": 128      # Neutral
            },
            
            # Sub-tile depth transitions
            "sub_tile_transitions": {
                "enabled": True,
                "transition_width": max(2, self.sub_tile_size // 16),
                "smoothing": "linear"
            },
            
            # Tessellation depth consistency
            "edge_depth_matching": True,
            "seamless_depth_transitions": True
        }
    
    def _generate_structure_lighting_parameters(self) -> Dict[str, Any]:
        """Generate lighting parameters for each structure type."""
        
        base_theme_color = self._get_theme_lighting_adjustments()["base_color"]
        
        return {
            "structure_lighting_values": {
                "CORNER": {
                    "base_color": tuple(min(255, c + 20) for c in base_theme_color),  # Brighter
                    "intensity": 1.2,
                    "shadow_depth": 0.8
                },
                "EDGE": {
                    "base_color": tuple(min(255, c + 10) for c in base_theme_color),  # Slightly brighter
                    "intensity": 1.1,
                    "shadow_depth": 0.6
                },
                "T_JUNCTION": {
                    "base_color": base_theme_color,  # Base lighting
                    "intensity": 1.0,
                    "shadow_depth": 0.7
                },
                "CENTER": {
                    "base_color": tuple(max(0, c - 20) for c in base_theme_color),  # Darker
                    "intensity": 0.8,
                    "shadow_depth": 0.4
                },
                "GENERIC": {
                    "base_color": base_theme_color,  # Neutral
                    "intensity": 1.0,
                    "shadow_depth": 0.5
                }
            },
            
            # Sub-tile lighting precision
            "sub_tile_lighting_precision": self.sub_tile_size,
            "lighting_consistency": "enforced",
            "seamless_lighting_transitions": True
        }
