"""
Stage 03: Perspective Setup
Sets up ONLY global camera parameters and lighting direction.
Establishes unified perspective and lighting that will be consistent across all tiles.
"""

import math
from typing import Dict, Any, Tuple, List
import numpy as np
import structlog

from ..core.pipeline_context import PipelineContext

logger = structlog.get_logger()

class PerspectiveType:
    """Perspective type constants."""
    TOP_DOWN = "top-down"
    ISOMETRIC = "isometric"
    SIDE_VIEW = "side-view"

class LightingType:
    """Lighting type constants."""
    AMBIENT = "ambient"
    DIRECTIONAL = "directional"
    POINT = "point"
    MIXED = "mixed"

def execute(context: PipelineContext) -> Dict[str, Any]:
    """
    Execute Stage 03: Perspective Setup
    
    Sets up ONLY global camera parameters and lighting direction.
    These will be used consistently across all tiles for visual coherence.
    
    Args:
        context: Pipeline context with validated job spec and tileset setup
        
    Returns:
        Dict with camera and lighting configuration
    """
    logger.info("Starting perspective setup", job_id=context.get_job_id())
    
    try:
        # Validate context state
        if not context.validate_context_state(3):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Get configuration
        extracted_config = getattr(context, 'extracted_config', None)
        if not extracted_config:
            context.add_error(3, "No extracted configuration found")
            return {"success": False, "errors": ["No extracted configuration found"]}
        
        # Get view angle and theme
        view_angle = context.job_spec.get("viewAngle", "top-down")
        theme = extracted_config["theme"]
        
        # Setup camera parameters
        camera_params = _setup_camera_parameters(view_angle, theme, extracted_config)
        
        # Setup lighting configuration
        lighting_config = _setup_lighting_configuration(view_angle, theme, extracted_config)
        
        # Setup depth system
        depth_config = _setup_depth_configuration(view_angle, extracted_config)
        
        # Store in context
        context.global_camera_params = camera_params
        context.lighting_config = lighting_config
        
        # Update context stage
        context.current_stage = 3

        logger.info("Perspective setup completed",
                   job_id=context.get_job_id(),
                   view_angle=view_angle,
                   lighting_type=lighting_config["type"],
                   camera_elevation=camera_params["elevation"])

        return {
            "success": True,
            "camera_params": camera_params,
            "lighting_config": lighting_config,
            "depth_config": depth_config,
            "view_angle": view_angle,
            "theme": theme
        }
        
    except Exception as e:
        error_msg = f"Perspective setup failed: {str(e)}"
        logger.error("Perspective setup failed", job_id=context.get_job_id(), error=error_msg)
        context.add_error(3, error_msg)
        return {"success": False, "errors": [error_msg]}

def _setup_camera_parameters(view_angle: str, theme: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup camera parameters based on view angle and theme."""
    
    if view_angle == PerspectiveType.TOP_DOWN:
        camera_params = {
            "type": "orthographic",
            "elevation": 90.0,  # Looking straight down
            "azimuth": 0.0,     # North-facing
            "distance": 10.0,   # Distance from subject
            "field_of_view": 45.0,
            "up_vector": [0, 1, 0],
            "look_at": [0, 0, 0],
            "projection_matrix": _calculate_orthographic_matrix(config["tile_size"])
        }
        
    elif view_angle == PerspectiveType.ISOMETRIC:
        # Classic isometric: 30° rotation, 35.264° elevation
        camera_params = {
            "type": "orthographic",
            "elevation": 35.264,  # arcsin(tan(30°))
            "azimuth": 45.0,      # 45° rotation for isometric
            "distance": 10.0,
            "field_of_view": 45.0,
            "up_vector": [0, 1, 0],
            "look_at": [0, 0, 0],
            "projection_matrix": _calculate_isometric_matrix(config["tile_size"])
        }
        
    elif view_angle == PerspectiveType.SIDE_VIEW:
        camera_params = {
            "type": "orthographic",
            "elevation": 0.0,     # Horizontal view
            "azimuth": 90.0,      # Side view
            "distance": 10.0,
            "field_of_view": 45.0,
            "up_vector": [0, 1, 0],
            "look_at": [0, 0, 0],
            "projection_matrix": _calculate_orthographic_matrix(config["tile_size"])
        }
        
    else:
        # Default to top-down
        camera_params = {
            "type": "orthographic",
            "elevation": 90.0,
            "azimuth": 0.0,
            "distance": 10.0,
            "field_of_view": 45.0,
            "up_vector": [0, 1, 0],
            "look_at": [0, 0, 0],
            "projection_matrix": _calculate_orthographic_matrix(config["tile_size"])
        }
    
    # Add theme-specific camera adjustments
    if theme == "fantasy":
        # Slightly elevated for dramatic effect
        camera_params["elevation"] += 5.0
    elif theme == "sci-fi":
        # Clean, precise angles
        pass  # Use defaults
    elif theme == "pixel":
        # Ensure pixel-perfect alignment
        camera_params["distance"] = 8.0
    elif theme == "nature":
        # Slightly lower for ground-level feel
        camera_params["elevation"] -= 3.0
    
    return camera_params

def _setup_lighting_configuration(view_angle: str, theme: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup lighting configuration based on view angle and theme."""
    
    # Base lighting setup
    if view_angle == PerspectiveType.TOP_DOWN:
        # Top-down lighting from above
        light_direction = [0.3, -1.0, 0.3]  # Slightly angled for depth
        shadow_direction = [0.3, 0.3]       # 2D shadow projection
        
    elif view_angle == PerspectiveType.ISOMETRIC:
        # Isometric lighting from top-left
        light_direction = [-0.5, -1.0, -0.5]  # Classic isometric lighting
        shadow_direction = [0.5, 0.5]         # Diagonal shadows
        
    elif view_angle == PerspectiveType.SIDE_VIEW:
        # Side lighting
        light_direction = [1.0, -0.5, 0.0]    # From the side
        shadow_direction = [0.0, 0.5]         # Horizontal shadows
        
    else:
        # Default lighting
        light_direction = [0.0, -1.0, 0.0]
        shadow_direction = [0.0, 0.0]
    
    # Theme-specific lighting adjustments
    lighting_config = {
        "type": LightingType.DIRECTIONAL,
        "direction": light_direction,
        "shadow_direction": shadow_direction,
        "intensity": 1.0,
        "ambient_strength": 0.3,
        "shadow_strength": 0.7,
        "color_temperature": 5500,  # Neutral daylight
    }
    
    if theme == "fantasy":
        # Warm, torch-like lighting
        lighting_config.update({
            "color_temperature": 3200,  # Warm light
            "ambient_strength": 0.4,    # More ambient for mystical feel
            "shadow_strength": 0.8,     # Strong shadows for drama
            "additional_lights": [
                {
                    "type": "point",
                    "position": [0, 2, 0],
                    "color": [1.0, 0.8, 0.6],  # Warm orange
                    "intensity": 0.5
                }
            ]
        })
        
    elif theme == "sci-fi":
        # Cool, artificial lighting
        lighting_config.update({
            "color_temperature": 6500,  # Cool blue light
            "ambient_strength": 0.5,    # High ambient for tech feel
            "shadow_strength": 0.6,     # Softer shadows
            "additional_lights": [
                {
                    "type": "ambient",
                    "color": [0.6, 0.8, 1.0],  # Cool blue ambient
                    "intensity": 0.3
                }
            ]
        })
        
    elif theme == "pixel":
        # Flat, even lighting for pixel art
        lighting_config.update({
            "type": LightingType.AMBIENT,
            "ambient_strength": 0.8,    # Very flat lighting
            "shadow_strength": 0.3,     # Minimal shadows
            "direction": [0, -1, 0],    # Straight down
        })
        
    elif theme == "nature":
        # Natural sunlight
        lighting_config.update({
            "color_temperature": 5800,  # Natural sunlight
            "ambient_strength": 0.6,    # Soft ambient
            "shadow_strength": 0.5,     # Natural shadows
            "additional_lights": [
                {
                    "type": "ambient",
                    "color": [0.8, 0.9, 1.0],  # Sky blue ambient
                    "intensity": 0.4
                }
            ]
        })
    
    return lighting_config

def _setup_depth_configuration(view_angle: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup depth and layer configuration for consistent z-ordering."""
    
    depth_config = {
        "layers": {
            "background": {"z_order": 0, "depth_range": [0.0, 0.2]},
            "ground": {"z_order": 1, "depth_range": [0.2, 0.4]},
            "objects": {"z_order": 2, "depth_range": [0.4, 0.7]},
            "foreground": {"z_order": 3, "depth_range": [0.7, 1.0]},
        },
        "depth_precision": 256,  # 8-bit depth maps
        "near_plane": 0.1,
        "far_plane": 100.0,
    }
    
    if view_angle == PerspectiveType.ISOMETRIC:
        # Isometric needs more precise depth handling
        depth_config.update({
            "depth_precision": 1024,  # Higher precision for isometric
            "layers": {
                "background": {"z_order": 0, "depth_range": [0.0, 0.15]},
                "ground": {"z_order": 1, "depth_range": [0.15, 0.35]},
                "walls": {"z_order": 2, "depth_range": [0.35, 0.65]},
                "objects": {"z_order": 3, "depth_range": [0.65, 0.85]},
                "foreground": {"z_order": 4, "depth_range": [0.85, 1.0]},
            }
        })
        
    elif view_angle == PerspectiveType.SIDE_VIEW:
        # Side view has different depth considerations
        depth_config.update({
            "layers": {
                "far_background": {"z_order": 0, "depth_range": [0.0, 0.1]},
                "background": {"z_order": 1, "depth_range": [0.1, 0.3]},
                "midground": {"z_order": 2, "depth_range": [0.3, 0.6]},
                "foreground": {"z_order": 3, "depth_range": [0.6, 0.9]},
                "near_foreground": {"z_order": 4, "depth_range": [0.9, 1.0]},
            }
        })
    
    return depth_config

def _calculate_orthographic_matrix(tile_size: int) -> List[List[float]]:
    """Calculate orthographic projection matrix for tile rendering."""
    # Simple orthographic projection for tile_size x tile_size viewport
    left, right = -tile_size/2, tile_size/2
    bottom, top = -tile_size/2, tile_size/2
    near, far = -100, 100
    
    matrix = [
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1]
    ]
    
    return matrix

def _calculate_isometric_matrix(tile_size: int) -> List[List[float]]:
    """Calculate isometric projection matrix."""
    # Isometric projection matrix
    # Combines rotation and orthographic projection
    
    # Rotation angles for isometric view
    alpha = math.radians(35.264)  # Elevation
    beta = math.radians(45.0)     # Azimuth
    
    # Rotation matrices
    cos_alpha, sin_alpha = math.cos(alpha), math.sin(alpha)
    cos_beta, sin_beta = math.cos(beta), math.sin(beta)
    
    # Combined isometric transformation
    matrix = [
        [cos_beta, sin_alpha * sin_beta, 0, 0],
        [0, cos_alpha, 0, 0],
        [sin_beta, -sin_alpha * cos_beta, 0, 0],
        [0, 0, 0, 1]
    ]
    
    # Scale to fit tile size
    scale = 2.0 / tile_size
    for i in range(3):
        for j in range(3):
            matrix[i][j] *= scale
    
    return matrix
