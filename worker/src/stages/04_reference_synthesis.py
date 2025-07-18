"""
Stage 4: Reference Synthesis
Generates reference images using REAL tessellation data from Stage 2.
Creates structure-aware, pattern-consistent references for proper Wang tile generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import structlog

from ..core.pipeline_context import PipelineContext

logger = structlog.get_logger()


def execute(context: PipelineContext) -> Dict[str, Any]:
    """Execute reference synthesis stage using real tessellation data."""
    try:
        logger.info("Starting reference synthesis", job_id=context.get_job_id())
        
        # Validate context state
        if not context.validate_context_state(4):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Get required data from context
        if not context.universal_tileset:
            context.add_error(4, "Missing universal tileset from Stage 2")
            return {"success": False, "errors": ["Missing universal tileset from Stage 2"]}
        
        extracted_config = getattr(context, 'extracted_config', {})
        if not extracted_config:
            context.add_error(4, "Missing extracted configuration")
            return {"success": False, "errors": ["Missing extracted configuration"]}
        
        # Create reference synthesizer with REAL tessellation data
        synthesizer = TessellationReferenceSynthesizer(
            tileset_setup=context.universal_tileset,
            config=extracted_config,
            output_path=context.output_path
        )
        
        # Generate structure-aware, pattern-consistent references
        synthesis_result = synthesizer.generate_tessellation_references()
        
        # Save reference images with proper tessellation naming
        synthesizer.save_tessellation_references(context, synthesis_result)
        
        # Store results in context for Stage 5
        context.reference_maps = synthesis_result["reference_maps"]
        context.control_images = synthesis_result["control_images"]
        
        # Update context stage
        context.current_stage = 4
        
        logger.info("Reference synthesis completed", 
                   job_id=context.get_job_id(),
                   tiles_processed=len(synthesis_result["reference_maps"]),
                   edge_patterns=len(synthesis_result["edge_pattern_summary"]),
                   seamless_edges=synthesis_result["tessellation_summary"]["total_seamless_edges"])
        
        return {
            "success": True,
            "data": {
                "reference_maps": synthesis_result["reference_maps"],
                "control_images": synthesis_result["control_images"],
                "edge_pattern_summary": synthesis_result["edge_pattern_summary"],
                "tessellation_summary": synthesis_result["tessellation_summary"]
            }
        }
        
    except Exception as e:
        error_msg = f"Reference synthesis failed: {str(e)}"
        logger.error("Reference synthesis failed", error=error_msg, job_id=context.get_job_id())
        context.add_error(4, error_msg)
        return {"success": False, "errors": [error_msg]}


class TessellationReferenceSynthesizer:
    """Synthesizes references using real tessellation data from Stage 2."""
    
    def __init__(self, tileset_setup, config: Dict[str, Any], output_path: Path):
        self.tileset_setup = tileset_setup
        self.config = config
        self.output_path = output_path
        
        # Get actual configuration values
        self.tile_size = tileset_setup.tile_size
        self.sub_tile_size = tileset_setup.sub_tile_size
        self.sub_tiles_per_row = tileset_setup.sub_tiles_per_row
        self.sub_tiles_per_col = tileset_setup.sub_tiles_per_col
        
        # Get theme from config (not tessellation parameters)
        self.theme = config.get("theme", "fantasy")
        self.palette = config.get("palette", "medieval")

        # Pattern consistency tracking
        self.pattern_colors = {}  # pattern_id -> RGB color for consistency
        self.structure_colors = {  # structure type -> RGB color
            "CORNER": (255, 100, 100),     # Red
            "EDGE": (100, 255, 100),       # Green  
            "T_JUNCTION": (100, 100, 255), # Blue
            "CENTER": (255, 255, 100),     # Yellow
            "GENERIC": (128, 128, 128)     # Gray
        }

        logger.info("Reference synthesizer initialized",
                   tile_size=self.tile_size,
                   sub_tile_size=self.sub_tile_size,
                   sub_tile_grid=f"{self.sub_tiles_per_row}x{self.sub_tiles_per_col}",
                   theme=self.theme,
                   palette=self.palette)
        
    def generate_tessellation_references(self) -> Dict[str, Any]:
        """Generate references using real tessellation data."""
        reference_maps = {}
        control_images = {}
        edge_pattern_summary = {}
        tessellation_stats = {
            "total_seamless_edges": 0,
            "pattern_usage": {},
            "structure_distribution": {}
        }
        
        # Process each tile using REAL tessellation data
        for tile_id, tile_spec in self.tileset_setup.tile_specs.items():
            # Generate structure-aware references
            tile_references = self._generate_structure_aware_references(tile_spec)
            reference_maps[tile_id] = tile_references
            
            # Generate pattern-consistent control image
            control_image = self._generate_pattern_control_image(tile_spec)
            control_images[tile_id] = control_image
            
            # Track edge patterns for consistency
            self._track_edge_patterns(tile_spec, edge_pattern_summary)
            
            # Update tessellation statistics
            self._update_tessellation_stats(tile_spec, tessellation_stats)
        
        return {
            "reference_maps": reference_maps,
            "control_images": control_images,
            "edge_pattern_summary": edge_pattern_summary,
            "tessellation_summary": tessellation_stats
        }
    
    def _generate_structure_aware_references(self, tile_spec) -> Dict[str, Image.Image]:
        """Generate references that understand sub-tile structure."""
        return {
            "structural": self._generate_structural_reference(tile_spec),
            "depth": self._generate_structure_depth_reference(tile_spec),
            "normal": self._generate_structure_normal_reference(tile_spec),
            "edge_constraints": self._generate_seamless_edge_constraints(tile_spec),
            "lighting": self._generate_structure_lighting_reference(tile_spec)
        }
    
    def _generate_structural_reference(self, tile_spec) -> Image.Image:
        """Generate structural reference showing sub-tile composition."""
        img = Image.new("RGB", (self.tile_size, self.tile_size), "black")
        draw = ImageDraw.Draw(img)
        
        # Draw sub-tile grid with structure-specific colors
        for row in range(self.sub_tiles_per_row):
            for col in range(self.sub_tiles_per_col):
                quadrant_key = f"sub_{row}_{col}"
                structure_type = tile_spec.structure_composition.get(quadrant_key, "GENERIC")
                
                # Calculate sub-tile bounds
                x1 = col * self.sub_tile_size
                y1 = row * self.sub_tile_size
                x2 = x1 + self.sub_tile_size
                y2 = y1 + self.sub_tile_size
                
                # Fill with structure-specific color
                color = self.structure_colors[structure_type]
                draw.rectangle([x1, y1, x2-1, y2-1], fill=color, outline="white", width=1)
                
                # Add structure type label
                try:
                    # Try to add text label (may fail if no font available)
                    text = structure_type[:3]  # First 3 chars
                    text_x = x1 + self.sub_tile_size // 4
                    text_y = y1 + self.sub_tile_size // 4
                    draw.text((text_x, text_y), text, fill="black")
                except:
                    pass  # Skip text if font not available
        
        # Draw connectivity points using REAL connectivity pattern
        self._draw_connectivity_points(draw, tile_spec)
        
        return img
    
    def _draw_connectivity_points(self, draw, tile_spec):
        """Draw connection points based on REAL connectivity pattern."""
        connection_size = max(4, self.sub_tile_size // 8)
        
        for direction in tile_spec.connectivity_pattern:
            # Get edge pattern for this direction
            pattern_id = tile_spec.edge_patterns.get(direction, "free")
            if pattern_id == "free":
                continue
                
            # Get consistent color for this pattern
            pattern_color = self._get_pattern_color(pattern_id)
            
            # Draw connection points at sub-tile boundaries
            if direction == "top":
                for i in range(1, self.sub_tiles_per_row):
                    x = i * self.sub_tile_size
                    draw.rectangle([x-connection_size//2, 0, x+connection_size//2, connection_size], 
                                 fill=pattern_color, outline="white")
            elif direction == "bottom":
                for i in range(1, self.sub_tiles_per_row):
                    x = i * self.sub_tile_size
                    draw.rectangle([x-connection_size//2, self.tile_size-connection_size, 
                                  x+connection_size//2, self.tile_size], fill=pattern_color, outline="white")
            elif direction == "left":
                for i in range(1, self.sub_tiles_per_col):
                    y = i * self.sub_tile_size
                    draw.rectangle([0, y-connection_size//2, connection_size, y+connection_size//2], 
                                 fill=pattern_color, outline="white")
            elif direction == "right":
                for i in range(1, self.sub_tiles_per_col):
                    y = i * self.sub_tile_size
                    draw.rectangle([self.tile_size-connection_size, y-connection_size//2, 
                                  self.tile_size, y+connection_size//2], fill=pattern_color, outline="white")
    
    def _generate_structure_depth_reference(self, tile_spec) -> Image.Image:
        """Generate depth map based on structure composition."""
        img = Image.new("L", (self.tile_size, self.tile_size), 128)  # Neutral gray
        
        # Vary depth per sub-tile based on structure type
        depth_values = {
            "CORNER": 180,      # Raised
            "EDGE": 140,        # Medium-high
            "T_JUNCTION": 120,  # Medium-low
            "CENTER": 100,      # Lowered
            "GENERIC": 128      # Neutral
        }
        
        for row in range(self.sub_tiles_per_row):
            for col in range(self.sub_tiles_per_col):
                quadrant_key = f"sub_{row}_{col}"
                structure_type = tile_spec.structure_composition.get(quadrant_key, "GENERIC")
                
                # Calculate sub-tile bounds
                x1 = col * self.sub_tile_size
                y1 = row * self.sub_tile_size
                x2 = x1 + self.sub_tile_size
                y2 = y1 + self.sub_tile_size
                
                # Create sub-image with appropriate depth
                depth_value = depth_values[structure_type]
                sub_img = Image.new("L", (self.sub_tile_size, self.sub_tile_size), depth_value)
                img.paste(sub_img, (x1, y1))
        
        return img
    
    def _generate_structure_normal_reference(self, tile_spec) -> Image.Image:
        """Generate normal map based on structure composition."""
        img = Image.new("RGB", (self.tile_size, self.tile_size), (128, 128, 255))  # Neutral normal
        
        # Vary normals per sub-tile based on structure type
        normal_values = {
            "CORNER": (140, 140, 255),      # Slightly angled
            "EDGE": (128, 140, 255),        # Edge-oriented
            "T_JUNCTION": (140, 128, 255),  # Junction-oriented
            "CENTER": (128, 128, 255),      # Flat
            "GENERIC": (128, 128, 255)      # Neutral
        }
        
        for row in range(self.sub_tiles_per_row):
            for col in range(self.sub_tiles_per_col):
                quadrant_key = f"sub_{row}_{col}"
                structure_type = tile_spec.structure_composition.get(quadrant_key, "GENERIC")
                
                # Calculate sub-tile bounds
                x1 = col * self.sub_tile_size
                y1 = row * self.sub_tile_size
                x2 = x1 + self.sub_tile_size
                y2 = y1 + self.sub_tile_size
                
                # Create sub-image with appropriate normal
                normal_color = normal_values[structure_type]
                sub_img = Image.new("RGB", (self.sub_tile_size, self.sub_tile_size), normal_color)
                img.paste(sub_img, (x1, y1))
        
        return img

    def _generate_seamless_edge_constraints(self, tile_spec) -> Image.Image:
        """Generate edge constraints using REAL seamless_edges data."""
        img = Image.new("RGB", (self.tile_size, self.tile_size), "black")
        draw = ImageDraw.Draw(img)

        edge_width = max(4, self.sub_tile_size // 8)

        # Only mark edges that are actually in seamless_edges list
        for direction in ["top", "bottom", "left", "right"]:
            if direction in tile_spec.seamless_edges:
                # This edge MUST match neighbors - mark as RED
                pattern_id = tile_spec.edge_patterns.get(direction, "free")
                pattern_color = self._get_pattern_color(pattern_id)

                if direction == "top":
                    draw.rectangle([0, 0, self.tile_size, edge_width], fill="red", outline=pattern_color, width=2)
                elif direction == "bottom":
                    draw.rectangle([0, self.tile_size-edge_width, self.tile_size, self.tile_size],
                                 fill="red", outline=pattern_color, width=2)
                elif direction == "left":
                    draw.rectangle([0, 0, edge_width, self.tile_size], fill="red", outline=pattern_color, width=2)
                elif direction == "right":
                    draw.rectangle([self.tile_size-edge_width, 0, self.tile_size, self.tile_size],
                                 fill="red", outline=pattern_color, width=2)
            else:
                # This edge is FREE - mark as BLUE
                if direction == "top":
                    draw.rectangle([0, 0, self.tile_size, edge_width], fill="blue")
                elif direction == "bottom":
                    draw.rectangle([0, self.tile_size-edge_width, self.tile_size, self.tile_size], fill="blue")
                elif direction == "left":
                    draw.rectangle([0, 0, edge_width, self.tile_size], fill="blue")
                elif direction == "right":
                    draw.rectangle([self.tile_size-edge_width, 0, self.tile_size, self.tile_size], fill="blue")

        # Mark sub-tile boundaries for precision
        for i in range(1, self.sub_tiles_per_row):
            x = i * self.sub_tile_size
            draw.line([(x, 0), (x, self.tile_size)], fill="gray", width=1)
        for i in range(1, self.sub_tiles_per_col):
            y = i * self.sub_tile_size
            draw.line([(0, y), (self.tile_size, y)], fill="gray", width=1)

        return img

    def _generate_structure_lighting_reference(self, tile_spec) -> Image.Image:
        """Generate lighting reference based on structure composition."""
        img = Image.new("RGB", (self.tile_size, self.tile_size), (200, 180, 140))  # Warm base

        # Vary lighting per sub-tile based on structure type
        lighting_values = {
            "CORNER": (240, 220, 180),      # Brighter (raised)
            "EDGE": (220, 200, 160),        # Medium-bright
            "T_JUNCTION": (200, 180, 140),  # Medium
            "CENTER": (180, 160, 120),      # Darker (lowered)
            "GENERIC": (200, 180, 140)      # Neutral
        }

        for row in range(self.sub_tiles_per_row):
            for col in range(self.sub_tiles_per_col):
                quadrant_key = f"sub_{row}_{col}"
                structure_type = tile_spec.structure_composition.get(quadrant_key, "GENERIC")

                # Calculate sub-tile bounds
                x1 = col * self.sub_tile_size
                y1 = row * self.sub_tile_size
                x2 = x1 + self.sub_tile_size
                y2 = y1 + self.sub_tile_size

                # Create sub-image with appropriate lighting
                lighting_color = lighting_values[structure_type]
                sub_img = Image.new("RGB", (self.sub_tile_size, self.sub_tile_size), lighting_color)
                img.paste(sub_img, (x1, y1))

        return img

    def _generate_pattern_control_image(self, tile_spec) -> Image.Image:
        """Generate control image for ControlNet using edge patterns."""
        # Use edge constraints as base
        control_img = self._generate_seamless_edge_constraints(tile_spec)

        # Add pattern-specific markers for ControlNet guidance
        draw = ImageDraw.Draw(control_img)

        # Add tile type indicator in center
        center_x = self.tile_size // 2
        center_y = self.tile_size // 2
        tile_type_color = {
            "corner": (255, 0, 0),
            "edge": (0, 255, 0),
            "t_junction": (0, 0, 255),
            "cross": (255, 255, 0)
        }.get(tile_spec.tile_type, (128, 128, 128))

        marker_size = self.sub_tile_size // 4
        draw.ellipse([center_x-marker_size, center_y-marker_size,
                     center_x+marker_size, center_y+marker_size],
                    fill=tile_type_color, outline="white", width=2)

        return control_img

    def _get_pattern_color(self, pattern_id: str) -> Tuple[int, int, int]:
        """Get consistent color for edge pattern."""
        if pattern_id not in self.pattern_colors:
            # Generate deterministic color from pattern_id hash
            hash_val = hash(pattern_id) % (256 * 256 * 256)
            r = (hash_val >> 16) & 255
            g = (hash_val >> 8) & 255
            b = hash_val & 255
            # Ensure colors are bright enough to see
            r = max(r, 100)
            g = max(g, 100)
            b = max(b, 100)
            self.pattern_colors[pattern_id] = (r, g, b)

        return self.pattern_colors[pattern_id]

    def _track_edge_patterns(self, tile_spec, edge_pattern_summary):
        """Track edge patterns for consistency validation."""
        for direction, pattern_id in tile_spec.edge_patterns.items():
            if pattern_id not in edge_pattern_summary:
                edge_pattern_summary[pattern_id] = {
                    "tiles_using": [],
                    "directions": [],
                    "color": self._get_pattern_color(pattern_id)
                }

            edge_pattern_summary[pattern_id]["tiles_using"].append(tile_spec.tile_id)
            edge_pattern_summary[pattern_id]["directions"].append(f"{tile_spec.tile_id}_{direction}")

    def _update_tessellation_stats(self, tile_spec, tessellation_stats):
        """Update tessellation statistics."""
        # Count seamless edges
        tessellation_stats["total_seamless_edges"] += len(tile_spec.seamless_edges)

        # Track pattern usage
        for pattern_id in tile_spec.edge_patterns.values():
            tessellation_stats["pattern_usage"][pattern_id] = \
                tessellation_stats["pattern_usage"].get(pattern_id, 0) + 1

        # Track structure distribution
        for structure_type in tile_spec.structure_composition.values():
            tessellation_stats["structure_distribution"][structure_type] = \
                tessellation_stats["structure_distribution"].get(structure_type, 0) + 1

    def save_tessellation_references(self, context: PipelineContext, synthesis_result: Dict[str, Any]):
        """Save reference images with proper tessellation-based naming."""
        # Create references directory
        references_dir = self.output_path / "references"
        references_dir.mkdir(parents=True, exist_ok=True)

        # Save reference maps for each tile
        for tile_id, reference_maps in synthesis_result["reference_maps"].items():
            tile_spec = self.tileset_setup.get_tile_spec(tile_id)
            if not tile_spec:
                logger.warning("No tile spec found for reference saving", tile_id=tile_id)
                continue

            # Create descriptive tile name using REAL tessellation data
            tile_name = self._create_tessellation_tile_name(tile_spec)

            # Create tile directory
            tile_dir = references_dir / tile_name
            tile_dir.mkdir(exist_ok=True)

            # Save each reference type
            for ref_type, image in reference_maps.items():
                if image:
                    image_path = tile_dir / f"{ref_type}.png"
                    image.save(image_path)
                    logger.debug(f"Saved reference image",
                               tile_id=tile_id,
                               tile_name=tile_name,
                               ref_type=ref_type,
                               path=str(image_path))

        # Save control images
        control_dir = references_dir / "control_images"
        control_dir.mkdir(exist_ok=True)

        for tile_id, control_image in synthesis_result["control_images"].items():
            if control_image:
                tile_spec = self.tileset_setup.get_tile_spec(tile_id)
                if tile_spec:
                    tile_name = self._create_tessellation_tile_name(tile_spec)
                    control_path = control_dir / f"{tile_name}_control.png"
                    control_image.save(control_path)
                    logger.debug(f"Saved control image",
                               tile_id=tile_id,
                               tile_name=tile_name,
                               path=str(control_path))

        # Save comprehensive tessellation summary
        summary_path = references_dir / "tessellation_summary.json"
        summary_data = {
            "job_id": context.get_job_id(),
            "tileset_configuration": {
                "tile_size": self.tile_size,
                "sub_tile_size": self.sub_tile_size,
                "sub_tiles_per_row": self.sub_tiles_per_row,
                "sub_tiles_per_col": self.sub_tiles_per_col,
                "tileset_type": self.config.get("tileset_type")
            },
            "tessellation_summary": synthesis_result["tessellation_summary"],
            "edge_pattern_summary": synthesis_result["edge_pattern_summary"],
            "reference_types": ["structural", "depth", "normal", "edge_constraints", "lighting"],
            "generated_at": context.job_spec.get("createdAt")
        }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        logger.info("Tessellation references saved",
                   job_id=context.get_job_id(),
                   references_dir=str(references_dir),
                   total_tiles=len(synthesis_result["reference_maps"]),
                   edge_patterns=len(synthesis_result["edge_pattern_summary"]))

    def _create_tessellation_tile_name(self, tile_spec):
        """Create descriptive tile name using REAL tessellation data."""
        tile_id = tile_spec.tile_id
        tile_type = tile_spec.tile_type
        connectivity = tile_spec.connectivity_pattern

        # Create name based on actual tessellation properties
        if tile_type == "corner":
            # Determine corner orientation from connectivity
            if "top" in connectivity and "right" in connectivity:
                return f"tile_corner_ne_{tile_id:02d}"
            elif "top" in connectivity and "left" in connectivity:
                return f"tile_corner_nw_{tile_id:02d}"
            elif "bottom" in connectivity and "right" in connectivity:
                return f"tile_corner_se_{tile_id:02d}"
            elif "bottom" in connectivity and "left" in connectivity:
                return f"tile_corner_sw_{tile_id:02d}"
            else:
                return f"tile_corner_{tile_id:02d}"

        elif tile_type == "edge":
            # Single connection edge
            direction = list(connectivity)[0] if connectivity else "unknown"
            return f"tile_edge_{direction}_{tile_id:02d}"

        elif tile_type == "t_junction":
            # Determine T orientation from missing direction
            all_directions = {"top", "bottom", "left", "right"}
            missing = all_directions - connectivity
            missing_dir = list(missing)[0] if missing else "unknown"
            return f"tile_t_{missing_dir}_{tile_id:02d}"

        elif tile_type == "cross":
            return f"tile_cross_{tile_id:02d}"

        else:
            return f"tile_{tile_type}_{tile_id:02d}"
