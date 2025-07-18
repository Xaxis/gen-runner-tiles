"""
Stage 7: Final Integration and Output - REAL IMPLEMENTATION
Finalizes the tileset and saves all outputs.
NO PLACEHOLDERS - Real file output and atlas generation.
"""

import os
from PIL import Image
from typing import Dict, Any, List
import structlog
import json

from ..core.pipeline_context import PipelineContext

logger = structlog.get_logger()


def execute(context: PipelineContext) -> Dict[str, Any]:
    """Execute Stage 7: Final integration and output generation."""
    try:
        logger.info("Starting final integration and output", job_id=context.get_job_id())
        
        # Validate context state
        if not context.validate_context_state(7):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Get final tiles (validated tiles from Stage 6, or generated tiles from Stage 5)
        final_tiles = getattr(context, 'validated_tiles', None) or context.generated_tiles
        tileset_setup = context.universal_tileset
        atlas_layout = context.atlas_layout
        
        # Validate required data
        if not all([final_tiles, tileset_setup]):
            context.add_error(7, "Missing required data for final output")
            return {"success": False, "errors": ["Missing required data for final output"]}
        
        # Initialize output generator
        output_generator = TilesetOutputGenerator(
            final_tiles=final_tiles,
            tileset_setup=tileset_setup,
            atlas_layout=atlas_layout,
            job_id=context.get_job_id()
        )
        
        # Generate all outputs
        output_result = output_generator.generate_final_outputs()
        
        # Store output paths in context
        context.output_files = output_result["output_files"]
        context.atlas_image_path = output_result["atlas_path"]
        context.individual_tiles_dir = output_result["tiles_dir"]
        
        # Update context stage
        context.current_stage = 7
        
        logger.info("Final integration completed", 
                   job_id=context.get_job_id(),
                   output_files=len(output_result["output_files"]),
                   atlas_generated=bool(output_result["atlas_path"]))
        
        return {
            "success": True,
            "output_files": output_result["output_files"],
            "atlas_path": output_result["atlas_path"],
            "tiles_dir": output_result["tiles_dir"],
            "summary": output_result["summary"]
        }
        
    except Exception as e:
        error_msg = f"Final integration failed: {str(e)}"
        logger.error("Final integration failed", error=error_msg, job_id=context.get_job_id())
        context.add_error(7, error_msg)
        return {"success": False, "errors": [error_msg]}


class TilesetOutputGenerator:
    """REAL implementation of tileset output generation."""
    
    def __init__(self, final_tiles: Dict[int, Image.Image], tileset_setup: Any,
                 atlas_layout: Dict[str, Any], job_id: str):
        self.final_tiles = final_tiles
        self.tileset_setup = tileset_setup
        self.atlas_layout = atlas_layout
        self.job_id = job_id
        
        # Output directory structure
        self.base_output_dir = f"../jobs/output/{job_id}"
        self.tiles_dir = os.path.join(self.base_output_dir, "tiles")
        self.atlas_dir = os.path.join(self.base_output_dir, "atlas")
        self.metadata_dir = os.path.join(self.base_output_dir, "metadata")
        
        # Ensure directories exist
        os.makedirs(self.tiles_dir, exist_ok=True)
        os.makedirs(self.atlas_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        logger.info("Tileset output generator initialized",
                   job_id=job_id,
                   tile_count=len(final_tiles),
                   output_dir=self.base_output_dir)
    
    def generate_final_outputs(self) -> Dict[str, Any]:
        """Generate all final outputs for the tileset."""
        output_files = []
        
        # 1. Save individual tiles
        individual_tile_paths = self._save_individual_tiles()
        output_files.extend(individual_tile_paths)
        
        # 2. Generate and save atlas
        atlas_path = self._generate_atlas()
        if atlas_path:
            output_files.append(atlas_path)
        
        # 3. Save tileset metadata
        metadata_path = self._save_tileset_metadata()
        if metadata_path:
            output_files.append(metadata_path)
        
        # 4. Generate summary
        summary = self._generate_summary()
        
        logger.info("Final outputs generated",
                   total_files=len(output_files),
                   individual_tiles=len(individual_tile_paths))
        
        return {
            "output_files": output_files,
            "atlas_path": atlas_path,
            "tiles_dir": self.tiles_dir,
            "summary": summary
        }
    
    def _save_individual_tiles(self) -> List[str]:
        """Save each tile as individual PNG files."""
        tile_paths = []
        
        for tile_id, tile_image in self.final_tiles.items():
            # Get tile spec for naming
            tile_spec = self.tileset_setup.tile_specs.get(tile_id)
            tile_type = tile_spec.tile_type if tile_spec else "unknown"
            
            # Generate filename
            filename = f"tile_{tile_id:02d}_{tile_type}.png"
            filepath = os.path.join(self.tiles_dir, filename)
            
            # Save tile
            tile_image.save(filepath, "PNG")
            tile_paths.append(filepath)
            
            logger.debug("Saved individual tile", 
                        tile_id=tile_id, 
                        tile_type=tile_type,
                        filepath=filename)
        
        return tile_paths
    
    def _generate_atlas(self) -> str:
        """Generate and save the complete tileset atlas."""
        # Get atlas dimensions
        atlas_columns = self.atlas_layout.get("columns", 4)
        atlas_rows = self.atlas_layout.get("rows", 4)
        tile_size = self.atlas_layout.get("tile_size", 32)
        
        atlas_width = atlas_columns * tile_size
        atlas_height = atlas_rows * tile_size
        
        # Create atlas image
        atlas_image = Image.new("RGB", (atlas_width, atlas_height), (0, 0, 0))
        
        # Place tiles in atlas
        for tile_id, tile_image in self.final_tiles.items():
            # Calculate position in atlas
            row = tile_id // atlas_columns
            col = tile_id % atlas_columns
            
            if row >= atlas_rows:
                continue  # Skip tiles that don't fit
            
            # Calculate pixel position
            x = col * tile_size
            y = row * tile_size
            
            # Resize tile if needed
            if tile_image.size != (tile_size, tile_size):
                tile_image = tile_image.resize((tile_size, tile_size), Image.LANCZOS)
            
            # Paste tile into atlas
            atlas_image.paste(tile_image, (x, y))
        
        # Save atlas
        atlas_filename = f"tileset_atlas_{atlas_columns}x{atlas_rows}.png"
        atlas_path = os.path.join(self.atlas_dir, atlas_filename)
        atlas_image.save(atlas_path, "PNG")
        
        logger.info("Atlas generated", 
                   atlas_size=f"{atlas_columns}x{atlas_rows}",
                   atlas_path=atlas_filename)
        
        return atlas_path
    
    def _save_tileset_metadata(self) -> str:
        """Save tileset metadata as JSON."""
        metadata = {
            "job_id": self.job_id,
            "tileset_info": {
                "tile_count": len(self.final_tiles),
                "tile_size": self.tileset_setup.tile_size,
                "sub_tile_size": self.tileset_setup.sub_tile_size,
                "tileset_type": self.tileset_setup.tileset_type,
                "atlas_layout": self.atlas_layout
            },
            "tiles": {}
        }
        
        # Add individual tile metadata
        for tile_id, tile_image in self.final_tiles.items():
            tile_spec = self.tileset_setup.tile_specs.get(tile_id)
            if tile_spec:
                metadata["tiles"][tile_id] = {
                    "tile_type": tile_spec.tile_type,
                    "connectivity": list(tile_spec.connectivity_pattern),
                    "edge_patterns": tile_spec.edge_patterns,
                    "neighbors": tile_spec.neighbors,
                    "size": tile_image.size
                }
        
        # Save metadata
        metadata_filename = "tileset_metadata.json"
        metadata_path = os.path.join(self.metadata_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Metadata saved", metadata_path=metadata_filename)
        return metadata_path
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the tileset generation."""
        tile_types = {}
        for tile_id, tile_image in self.final_tiles.items():
            tile_spec = self.tileset_setup.tile_specs.get(tile_id)
            if tile_spec:
                tile_type = tile_spec.tile_type
                tile_types[tile_type] = tile_types.get(tile_type, 0) + 1
        
        summary = {
            "total_tiles": len(self.final_tiles),
            "tile_types": tile_types,
            "atlas_size": f"{self.atlas_layout.get('columns', 4)}x{self.atlas_layout.get('rows', 4)}",
            "tile_size": self.tileset_setup.tile_size,
            "output_directory": self.base_output_dir,
            "generation_complete": True
        }
        
        return summary
