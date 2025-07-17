"""
Stage 04: Reference Synthesis
Generates ControlNet conditioning images with shared edge coordination.
Creates reference maps that ensure adjacent tiles have matching edge content.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Any, List, Tuple
import cv2
import structlog

from ..core.pipeline_context import PipelineContext

logger = structlog.get_logger()

def execute(context: PipelineContext) -> Dict[str, Any]:
    """
    Execute Stage 04: Reference Synthesis
    
    Generates ControlNet conditioning images with shared edge coordination:
    - Creates reference maps for each tile using camera/lighting from Stage 03
    - Ensures shared edges have matching reference content
    - Prepares conditioning images for ControlNet guidance
    
    Args:
        context: Pipeline context with tileset setup and perspective config
        
    Returns:
        Dict with reference maps and control images
    """
    logger.info("Starting reference synthesis", job_id=context.get_job_id())
    
    try:
        # Validate context state
        if not context.validate_context_state(4):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Get required data from context
        tileset_setup = context.universal_tileset
        camera_params = context.global_camera_params
        lighting_config = context.lighting_config
        extracted_config = getattr(context, 'extracted_config', None)
        
        if not tileset_setup:
            context.add_error(4, "Tileset setup not found")
            return {"success": False, "errors": ["Tileset setup not found"]}
        
        if not camera_params or not lighting_config:
            context.add_error(4, "Perspective configuration not found")
            return {"success": False, "errors": ["Perspective configuration not found"]}
        
        # Initialize reference synthesizer
        synthesizer = ReferenceSynthesizer(
            tileset_setup=tileset_setup,
            camera_params=camera_params,
            lighting_config=lighting_config,
            config=extracted_config
        )
        
        # Generate reference maps with shared edge coordination
        synthesis_result = synthesizer.generate_coordinated_references()
        
        # Store in context
        context.reference_maps = synthesis_result["reference_maps"]
        context.control_images = synthesis_result["control_images"]
        
        logger.info("Reference synthesis completed", 
                   job_id=context.get_job_id(),
                   tiles_processed=len(synthesis_result["reference_maps"]),
                   shared_edges_coordinated=len(synthesis_result["shared_edge_matches"]))
        
        return {
            "success": True,
            "reference_maps": synthesis_result["reference_maps"],
            "control_images": synthesis_result["control_images"],
            "shared_edge_matches": synthesis_result["shared_edge_matches"],
            "synthesis_summary": {
                "tiles_processed": len(synthesis_result["reference_maps"]),
                "shared_edges_coordinated": len(synthesis_result["shared_edge_matches"]),
                "controlnet_model": extracted_config["models"]["controlnet_model"]
            }
        }
        
    except Exception as e:
        error_msg = f"Reference synthesis failed: {str(e)}"
        logger.error("Reference synthesis failed", job_id=context.get_job_id(), error=error_msg)
        context.add_error(4, error_msg)
        return {"success": False, "errors": [error_msg]}

class ReferenceSynthesizer:
    """Synthesizes reference maps with shared edge coordination."""
    
    def __init__(self, tileset_setup: Any, camera_params: Dict[str, Any], 
                 lighting_config: Dict[str, Any], config: Dict[str, Any]):
        self.tileset_setup = tileset_setup
        self.camera_params = camera_params
        self.lighting_config = lighting_config
        self.config = config
        
        self.tile_size = tileset_setup.tile_size
        self.controlnet_model = config["models"]["controlnet_model"]
        
        # Get adjacency and shared edge data
        self.adjacency_graph = tileset_setup.get_adjacency_graph()
        self.shared_edges = tileset_setup.get_shared_edges()
    
    def generate_coordinated_references(self) -> Dict[str, Any]:
        """Generate reference maps with shared edge coordination."""
        reference_maps = {}
        control_images = {}
        shared_edge_matches = []
        
        # Step 1: Generate individual reference maps for each tile
        for tile_id in range(self.tileset_setup.tile_count):
            tile_spec = self.tileset_setup.get_tile_spec(tile_id)
            
            if not tile_spec:
                logger.warning("No tile spec found", tile_id=tile_id)
                continue
            
            # Generate reference maps for this tile
            tile_references = self._generate_tile_references(tile_spec)
            reference_maps[tile_id] = tile_references
        
        # Step 2: Coordinate shared edges to ensure matching content
        shared_edge_matches = self._coordinate_shared_edges(reference_maps)
        
        # Step 3: Generate control images for ControlNet
        for tile_id in reference_maps:
            control_image = self._generate_control_image(reference_maps[tile_id])
            control_images[tile_id] = control_image
        
        return {
            "reference_maps": reference_maps,
            "control_images": control_images,
            "shared_edge_matches": shared_edge_matches
        }
    
    def _generate_tile_references(self, tile_spec: Any) -> Dict[str, Image.Image]:
        """Generate all reference map types for a single tile."""
        references = {}
        
        # Generate structural reference (edges, corners, borders)
        references["structural"] = self._generate_structural_reference(tile_spec)
        
        # Generate depth reference based on camera perspective
        references["depth"] = self._generate_depth_reference(tile_spec)
        
        # Generate normal map reference
        references["normal"] = self._generate_normal_reference(tile_spec)
        
        # Generate edge constraint reference
        references["edge_constraints"] = self._generate_edge_constraints_reference(tile_spec)
        
        # Generate lighting reference
        references["lighting"] = self._generate_lighting_reference(tile_spec)
        
        return references
    
    def _generate_structural_reference(self, tile_spec: Any) -> Image.Image:
        """Generate structural reference showing tile composition."""
        img = Image.new("RGB", (self.tile_size, self.tile_size), "black")
        draw = ImageDraw.Draw(img)
        
        composition = tile_spec.structure_composition
        half_size = self.tile_size // 2
        
        # Define colors for different structure types
        structure_colors = {
            "center": (128, 128, 128),           # Gray
            "border_top": (255, 255, 255),      # White
            "border_right": (255, 255, 255),
            "border_bottom": (255, 255, 255),
            "border_left": (255, 255, 255),
            "edge_ne": (200, 200, 255),         # Light blue
            "edge_nw": (200, 200, 255),
            "edge_se": (200, 200, 255),
            "edge_sw": (200, 200, 255),
            "corner_ne": (255, 200, 200),       # Light red
            "corner_nw": (255, 200, 200),
            "corner_se": (255, 200, 200),
            "corner_sw": (255, 200, 200),
        }
        
        # Draw each quadrant
        quadrants = [
            ("top_left", 0, 0),
            ("top_right", half_size, 0),
            ("bottom_left", 0, half_size),
            ("bottom_right", half_size, half_size)
        ]
        
        for quadrant, x, y in quadrants:
            structure = composition.get(quadrant, "center")
            color = structure_colors.get(structure, (64, 64, 64))
            
            # Fill quadrant with base color
            draw.rectangle([x, y, x + half_size, y + half_size], fill=color)
            
            # Add structure-specific patterns
            self._draw_structure_pattern(draw, structure, x, y, half_size)
        
        # Draw sub-tile boundaries
        draw.line([half_size, 0, half_size, self.tile_size], fill="white", width=1)
        draw.line([0, half_size, self.tile_size, half_size], fill="white", width=1)
        
        return img
    
    def _draw_structure_pattern(self, draw: ImageDraw.Draw, structure: str, 
                               x: int, y: int, size: int):
        """Draw structure-specific patterns within a quadrant."""
        
        if "border" in structure:
            # Draw border indication
            border_width = max(2, size // 8)
            
            if "top" in structure:
                draw.rectangle([x, y, x + size, y + border_width], fill="white")
            elif "bottom" in structure:
                draw.rectangle([x, y + size - border_width, x + size, y + size], fill="white")
            elif "left" in structure:
                draw.rectangle([x, y, x + border_width, y + size], fill="white")
            elif "right" in structure:
                draw.rectangle([x + size - border_width, y, x + size, y + size], fill="white")
        
        elif "edge" in structure:
            # Draw external corner
            corner_size = size // 3
            
            if "ne" in structure:
                points = [(x + size, y), (x + size, y + corner_size), (x + size - corner_size, y)]
                draw.polygon(points, fill="white")
            elif "nw" in structure:
                points = [(x, y), (x + corner_size, y), (x, y + corner_size)]
                draw.polygon(points, fill="white")
            elif "se" in structure:
                points = [(x + size, y + size), (x + size - corner_size, y + size), (x + size, y + size - corner_size)]
                draw.polygon(points, fill="white")
            elif "sw" in structure:
                points = [(x, y + size), (x, y + size - corner_size), (x + corner_size, y + size)]
                draw.polygon(points, fill="white")
        
        elif "corner" in structure:
            # Draw internal corner (L-shape)
            corner_size = size // 2
            
            if "ne" in structure:
                draw.rectangle([x, y + corner_size, x + corner_size, y + size], fill="white")
                draw.rectangle([x + corner_size, y, x + size, y + corner_size], fill="white")
            elif "nw" in structure:
                draw.rectangle([x + corner_size, y + corner_size, x + size, y + size], fill="white")
                draw.rectangle([x, y, x + corner_size, y + corner_size], fill="white")
            elif "se" in structure:
                draw.rectangle([x, y, x + corner_size, y + corner_size], fill="white")
                draw.rectangle([x + corner_size, y + corner_size, x + size, y + size], fill="white")
            elif "sw" in structure:
                draw.rectangle([x + corner_size, y, x + size, y + corner_size], fill="white")
                draw.rectangle([x, y + corner_size, x + corner_size, y + size], fill="white")
    
    def _generate_depth_reference(self, tile_spec: Any) -> Image.Image:
        """Generate depth reference based on camera perspective."""
        img = Image.new("L", (self.tile_size, self.tile_size), 128)  # Grayscale
        draw = ImageDraw.Draw(img)
        
        elevation = self.camera_params.get("elevation", 90)
        composition = tile_spec.structure_composition
        
        # Base depth values based on structure type
        depth_values = {
            "center": 128,        # Mid-depth
            "border_top": 96,     # Slightly recessed
            "border_right": 96,
            "border_bottom": 96,
            "border_left": 96,
            "edge_ne": 64,        # More recessed (external corners)
            "edge_nw": 64,
            "edge_se": 64,
            "edge_sw": 64,
            "corner_ne": 192,     # Raised (internal corners)
            "corner_nw": 192,
            "corner_se": 192,
            "corner_sw": 192,
        }
        
        half_size = self.tile_size // 2
        quadrants = [
            ("top_left", 0, 0),
            ("top_right", half_size, 0),
            ("bottom_left", 0, half_size),
            ("bottom_right", half_size, half_size)
        ]
        
        for quadrant, x, y in quadrants:
            structure = composition.get(quadrant, "center")
            base_depth = depth_values.get(structure, 128)
            
            # Create gradient based on camera elevation
            if elevation > 45:  # Top-down or high angle
                # Uniform depth with slight variation
                for dy in range(half_size):
                    depth = base_depth + int((dy / half_size) * 16 - 8)
                    depth = max(0, min(255, depth))
                    draw.line([x, y + dy, x + half_size, y + dy], fill=depth)
            else:  # Isometric or side view
                # More pronounced depth gradient
                for dy in range(half_size):
                    for dx in range(half_size):
                        depth = base_depth + int((dy / half_size) * 32 - 16)
                        depth = max(0, min(255, depth))
                        draw.point([x + dx, y + dy], fill=depth)
        
        return img.convert("RGB")
    
    def _generate_normal_reference(self, tile_spec: Any) -> Image.Image:
        """Generate normal map reference for surface details."""
        # Create a basic normal map (pointing up = blue)
        img = Image.new("RGB", (self.tile_size, self.tile_size), (128, 128, 255))
        
        # Add subtle normal variations based on lighting direction
        light_dir = self.lighting_config.get("direction", [0, -1, 0])
        
        # #TODO - Simple normal map generation based on lighting
        # This is simplified - could be enhanced with actual normal calculations
        
        return img
    
    def _generate_edge_constraints_reference(self, tile_spec: Any) -> Image.Image:
        """Generate edge constraints reference for seamless tiling."""
        img = Image.new("RGB", (self.tile_size, self.tile_size), "black")
        draw = ImageDraw.Draw(img)
        
        seamless_edges = tile_spec.seamless_edges
        
        # Mark edges that need to be seamless
        edge_width = 2
        
        if "top" in seamless_edges:
            draw.rectangle([0, 0, self.tile_size, edge_width], fill="white")
        
        if "right" in seamless_edges:
            draw.rectangle([self.tile_size - edge_width, 0, self.tile_size, self.tile_size], fill="white")
        
        if "bottom" in seamless_edges:
            draw.rectangle([0, self.tile_size - edge_width, self.tile_size, self.tile_size], fill="white")
        
        if "left" in seamless_edges:
            draw.rectangle([0, 0, edge_width, self.tile_size], fill="white")
        
        return img
    
    def _generate_lighting_reference(self, tile_spec: Any) -> Image.Image:
        """Generate lighting reference based on global lighting configuration."""
        img = Image.new("RGB", (self.tile_size, self.tile_size), "black")
        draw = ImageDraw.Draw(img)
        
        # Get lighting direction
        light_dir = self.lighting_config.get("direction", [0, -1, 0])
        
        # Create simple lighting gradient
        for y in range(self.tile_size):
            for x in range(self.tile_size):
                # Calculate lighting intensity based on position and light direction
                intensity = 128 + int(64 * (light_dir[0] * (x / self.tile_size - 0.5) + 
                                           light_dir[1] * (y / self.tile_size - 0.5)))
                intensity = max(0, min(255, intensity))
                draw.point([x, y], fill=(intensity, intensity, intensity))
        
        return img
    
    def _coordinate_shared_edges(self, reference_maps: Dict[int, Dict[str, Image.Image]]) -> List[Dict[str, Any]]:
        """Coordinate shared edges to ensure matching reference content."""
        shared_edge_matches = []
        
        for edge in self.shared_edges:
            tile_a_id = edge.tile_a_id
            tile_b_id = edge.tile_b_id
            edge_type = edge.edge_type
            
            if tile_a_id not in reference_maps or tile_b_id not in reference_maps:
                continue
            
            # Get reference maps for both tiles
            tile_a_refs = reference_maps[tile_a_id]
            tile_b_refs = reference_maps[tile_b_id]
            
            # Coordinate each reference type
            for ref_type in tile_a_refs:
                if ref_type in tile_b_refs:
                    self._match_edge_content(
                        tile_a_refs[ref_type], 
                        tile_b_refs[ref_type], 
                        edge_type
                    )
            
            shared_edge_matches.append({
                "tile_a_id": tile_a_id,
                "tile_b_id": tile_b_id,
                "edge_type": edge_type,
                "coordinated": True
            })
        
        return shared_edge_matches
    
    def _match_edge_content(self, img_a: Image.Image, img_b: Image.Image, edge_type: str):
        """Make edge content match between two reference images."""
        # Convert to numpy arrays for easier manipulation
        arr_a = np.array(img_a)
        arr_b = np.array(img_b)
        
        if edge_type == "right":  # tile_a's right edge = tile_b's left edge
            # Copy tile_a's right edge to tile_b's left edge
            arr_b[:, :2] = arr_a[:, -2:]
        elif edge_type == "bottom":  # tile_a's bottom edge = tile_b's top edge
            # Copy tile_a's bottom edge to tile_b's top edge
            arr_b[:2, :] = arr_a[-2:, :]
        
        # Update the image
        img_b.paste(Image.fromarray(arr_b))
    
    def _generate_control_image(self, references: Dict[str, Image.Image]) -> Image.Image:
        """Generate primary control image for ControlNet based on model type."""
        
        if self.controlnet_model == "flux-controlnet-union":
            # Union model can handle multiple control types
            return self._generate_union_control_image(references)
        elif self.controlnet_model == "flux-controlnet-depth":
            # Use depth reference
            return references["depth"]
        elif self.controlnet_model == "flux-controlnet-canny":
            # Generate canny edge detection from structural reference
            return self._generate_canny_control_image(references["structural"])
        else:
            # Default to structural reference
            return references["structural"]
    
    def _generate_union_control_image(self, references: Dict[str, Image.Image]) -> Image.Image:
        """Generate multi-channel control image for Union ControlNet."""
        structural = references["structural"]
        depth = references["depth"]
        edge_constraints = references["edge_constraints"]
        
        # Convert to numpy arrays
        struct_array = np.array(structural)
        depth_array = np.array(depth)
        edge_array = np.array(edge_constraints)
        
        # Combine channels: R=structural, G=depth, B=edge_constraints
        combined = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        combined[:, :, 0] = struct_array[:, :, 0]  # Red channel from structural
        combined[:, :, 1] = depth_array[:, :, 0]   # Green channel from depth
        combined[:, :, 2] = edge_array[:, :, 0]    # Blue channel from edge constraints
        
        return Image.fromarray(combined)
    
    def _generate_canny_control_image(self, structural_ref: Image.Image) -> Image.Image:
        """Generate Canny edge detection control image."""
        # Convert to grayscale
        gray = structural_ref.convert("L")
        gray_array = np.array(gray)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_array, 50, 150)
        
        # Convert back to RGB
        edge_rgb = np.stack([edges, edges, edges], axis=2)
        
        return Image.fromarray(edge_rgb)
