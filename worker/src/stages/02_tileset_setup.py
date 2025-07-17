"""
Stage 02: Tileset Setup
Sets up the complete tileset structure including:
- Universal building blocks (13 types)
- Tile adjacency graph (which tiles are neighbors)
- Shared edge definitions between adjacent tiles
- Atlas layout mapping for multi-tile diffusion
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import structlog

from ..core.pipeline_context import PipelineContext

logger = structlog.get_logger()

class TileStructure(Enum):
    """Universal tile structures - theme agnostic."""
    CENTER = "center"
    BORDER_TOP = "border_top"
    BORDER_RIGHT = "border_right"
    BORDER_BOTTOM = "border_bottom"
    BORDER_LEFT = "border_left"
    EDGE_NE = "edge_ne"
    EDGE_NW = "edge_nw"
    EDGE_SE = "edge_se"
    EDGE_SW = "edge_sw"
    CORNER_NE = "corner_ne"
    CORNER_NW = "corner_nw"
    CORNER_SE = "corner_se"
    CORNER_SW = "corner_sw"

@dataclass
class TilePosition:
    """Position of a tile in the atlas grid."""
    tile_id: int
    row: int
    col: int
    x_pixel: int
    y_pixel: int

@dataclass
class SharedEdge:
    """Defines a shared edge between two adjacent tiles."""
    tile_a_id: int
    tile_b_id: int
    edge_type: str  # "top", "right", "bottom", "left"
    edge_length: int  # Length in pixels
    must_match: bool  # Whether this edge must be identical

@dataclass
class TileSpec:
    """Complete specification for a single tile."""
    tile_id: int
    position: TilePosition
    structure_composition: Dict[str, str]  # quadrant -> structure type
    prompt_components: List[str]
    seamless_edges: List[str]  # Which edges must be seamless
    shared_edges: List[SharedEdge]  # Edges shared with neighbors
    neighbors: Dict[str, int]  # direction -> neighbor_tile_id

class TilesetSetup:
    """Sets up the complete tileset structure for multi-tile diffusion."""

    def __init__(self, config: Dict[str, Any]):
        self.theme = config["theme"]
        self.tile_count = config["tile_count"]
        self.tile_size = config["tile_size"]
        self.sub_tile_size = config["sub_tile_size"]

        # Atlas configuration
        atlas_config = config.get("atlas", {})
        self.atlas_columns = atlas_config.get("columns", int(np.sqrt(self.tile_count)))
        self.atlas_rows = atlas_config.get("rows", int(np.ceil(self.tile_count / self.atlas_columns)))
        self.atlas_padding = atlas_config.get("padding", 1)

        # Tileset configuration
        tileset_config = config.get("tileset", {})
        self.building_blocks = tileset_config.get("building_blocks", ["center"])
        self.variations_per_block = tileset_config.get("variations_per_block", 3)

        # Generated data structures
        self.tile_specs: Dict[int, TileSpec] = {}
        self.adjacency_graph: Dict[int, Dict[str, int]] = {}  # tile_id -> {direction: neighbor_id}
        self.shared_edges: List[SharedEdge] = []
        self.atlas_layout: Dict[int, TilePosition] = {}

        self._setup_tileset()

    def _setup_tileset(self):
        """Set up the complete tileset structure."""
        logger.info("Setting up tileset structure",
                   tile_count=self.tile_count,
                   atlas_size=f"{self.atlas_columns}x{self.atlas_rows}")

        # 1. Create atlas layout (tile positions)
        self._create_atlas_layout()

        # 2. Build adjacency graph (which tiles are neighbors)
        self._build_adjacency_graph()

        # 3. Generate tile structure compositions
        self._generate_tile_compositions()

        # 4. Define shared edges between adjacent tiles
        self._define_shared_edges()

        # 5. Create complete tile specifications
        self._create_tile_specifications()

    def _create_atlas_layout(self):
        """Create the atlas layout mapping tile IDs to grid positions."""
        for tile_id in range(self.tile_count):
            row = tile_id // self.atlas_columns
            col = tile_id % self.atlas_columns

            # Calculate pixel positions in the atlas
            x_pixel = col * (self.tile_size + self.atlas_padding)
            y_pixel = row * (self.tile_size + self.atlas_padding)

            position = TilePosition(
                tile_id=tile_id,
                row=row,
                col=col,
                x_pixel=x_pixel,
                y_pixel=y_pixel
            )

            self.atlas_layout[tile_id] = position

    def _build_adjacency_graph(self):
        """Build the adjacency graph showing which tiles are neighbors."""
        for tile_id in range(self.tile_count):
            position = self.atlas_layout[tile_id]
            neighbors = {}

            # Check all four directions
            directions = [
                ("top", -1, 0),
                ("right", 0, 1),
                ("bottom", 1, 0),
                ("left", 0, -1)
            ]

            for direction, row_offset, col_offset in directions:
                neighbor_row = position.row + row_offset
                neighbor_col = position.col + col_offset

                # Check if neighbor exists in atlas
                if (0 <= neighbor_row < self.atlas_rows and
                    0 <= neighbor_col < self.atlas_columns):

                    neighbor_id = neighbor_row * self.atlas_columns + neighbor_col
                    if neighbor_id < self.tile_count:
                        neighbors[direction] = neighbor_id

            self.adjacency_graph[tile_id] = neighbors

    def _generate_tile_compositions(self):
        """Generate structure compositions for each tile based on position and neighbors."""
        # Define composition patterns based on tile position in atlas
        for tile_id in range(self.tile_count):
            position = self.atlas_layout[tile_id]
            neighbors = self.adjacency_graph[tile_id]

            # Determine tile role based on position and neighbors
            tile_role = self._determine_tile_role(position, neighbors)

            # Generate structure composition based on role
            composition = self._generate_composition_for_role(tile_role, neighbors)

            # Store composition (will be used in tile spec creation)
            setattr(self, f'_composition_{tile_id}', composition)

    def _determine_tile_role(self, position: TilePosition, neighbors: Dict[str, int]) -> str:
        """Determine the role of a tile based on its position and neighbors."""
        neighbor_count = len(neighbors)

        # Corner tiles (2 neighbors)
        if neighbor_count == 2:
            if "right" in neighbors and "bottom" in neighbors:
                return "corner_nw"  # Top-left corner
            elif "left" in neighbors and "bottom" in neighbors:
                return "corner_ne"  # Top-right corner
            elif "right" in neighbors and "top" in neighbors:
                return "corner_sw"  # Bottom-left corner
            elif "left" in neighbors and "top" in neighbors:
                return "corner_se"  # Bottom-right corner

        # Edge tiles (3 neighbors)
        elif neighbor_count == 3:
            if "top" not in neighbors:
                return "edge_top"
            elif "right" not in neighbors:
                return "edge_right"
            elif "bottom" not in neighbors:
                return "edge_bottom"
            elif "left" not in neighbors:
                return "edge_left"

        # Interior tiles (4 neighbors) or isolated tiles
        else:
            return "interior"

    def _generate_composition_for_role(self, role: str, neighbors: Dict[str, int]) -> Dict[str, str]:
        """Generate sub-tile composition based on tile role."""
        # Default to all center
        composition = {
            "top_left": TileStructure.CENTER.value,
            "top_right": TileStructure.CENTER.value,
            "bottom_left": TileStructure.CENTER.value,
            "bottom_right": TileStructure.CENTER.value
        }

        # Modify based on role
        if role == "corner_nw":
            composition.update({
                "top_left": TileStructure.EDGE_NW.value,
                "top_right": TileStructure.BORDER_TOP.value,
                "bottom_left": TileStructure.BORDER_LEFT.value,
                "bottom_right": TileStructure.CENTER.value
            })
        elif role == "corner_ne":
            composition.update({
                "top_left": TileStructure.BORDER_TOP.value,
                "top_right": TileStructure.EDGE_NE.value,
                "bottom_left": TileStructure.CENTER.value,
                "bottom_right": TileStructure.BORDER_RIGHT.value
            })
        elif role == "corner_sw":
            composition.update({
                "top_left": TileStructure.BORDER_LEFT.value,
                "top_right": TileStructure.CENTER.value,
                "bottom_left": TileStructure.EDGE_SW.value,
                "bottom_right": TileStructure.BORDER_BOTTOM.value
            })
        elif role == "corner_se":
            composition.update({
                "top_left": TileStructure.CENTER.value,
                "top_right": TileStructure.BORDER_RIGHT.value,
                "bottom_left": TileStructure.BORDER_BOTTOM.value,
                "bottom_right": TileStructure.EDGE_SE.value
            })
        elif role == "edge_top":
            composition.update({
                "top_left": TileStructure.BORDER_TOP.value,
                "top_right": TileStructure.BORDER_TOP.value
            })
        elif role == "edge_right":
            composition.update({
                "top_right": TileStructure.BORDER_RIGHT.value,
                "bottom_right": TileStructure.BORDER_RIGHT.value
            })
        elif role == "edge_bottom":
            composition.update({
                "bottom_left": TileStructure.BORDER_BOTTOM.value,
                "bottom_right": TileStructure.BORDER_BOTTOM.value
            })
        elif role == "edge_left":
            composition.update({
                "top_left": TileStructure.BORDER_LEFT.value,
                "bottom_left": TileStructure.BORDER_LEFT.value
            })

        return composition

    def _define_shared_edges(self):
        """Define shared edges between adjacent tiles."""
        for tile_id, neighbors in self.adjacency_graph.items():
            for direction, neighbor_id in neighbors.items():
                # Only create edge once (from lower ID to higher ID)
                if tile_id < neighbor_id:
                    edge = SharedEdge(
                        tile_a_id=tile_id,
                        tile_b_id=neighbor_id,
                        edge_type=direction,
                        edge_length=self.tile_size,
                        must_match=True  # All adjacent edges must match for seamless tiling
                    )
                    self.shared_edges.append(edge)

    def _create_tile_specifications(self):
        """Create complete specifications for each tile."""
        for tile_id in range(self.tile_count):
            position = self.atlas_layout[tile_id]
            neighbors = self.adjacency_graph[tile_id]
            composition = getattr(self, f'_composition_{tile_id}')

            # Generate prompt components
            prompt_components = self._generate_prompt_components(composition)

            # Determine seamless edges
            seamless_edges = list(neighbors.keys())  # All edges with neighbors must be seamless

            # Find shared edges for this tile
            tile_shared_edges = [
                edge for edge in self.shared_edges
                if edge.tile_a_id == tile_id or edge.tile_b_id == tile_id
            ]

            # Create tile specification
            tile_spec = TileSpec(
                tile_id=tile_id,
                position=position,
                structure_composition=composition,
                prompt_components=prompt_components,
                seamless_edges=seamless_edges,
                shared_edges=tile_shared_edges,
                neighbors=neighbors
            )

            self.tile_specs[tile_id] = tile_spec

    def _generate_prompt_components(self, composition: Dict[str, str]) -> List[str]:
        """Generate theme-agnostic prompt components based on composition."""
        components = []

        # Count structure types
        structure_counts = {}
        for structure in composition.values():
            structure_counts[structure] = structure_counts.get(structure, 0) + 1

        # Add components based on dominant structures
        for structure, count in structure_counts.items():
            if structure == TileStructure.CENTER.value:
                components.append("solid interior texture")
            elif "border" in structure:
                components.append("border transition")
            elif "edge" in structure:
                components.append("external corner")
            elif "corner" in structure:
                components.append("internal corner")

        # Add general components
        components.extend([
            "seamless tileable texture",
            "consistent lighting",
            "high quality detailed"
        ])

        return components

    def get_tile_spec(self, tile_id: int) -> Optional[TileSpec]:
        """Get the complete specification for a tile."""
        return self.tile_specs.get(tile_id)

    def get_adjacency_graph(self) -> Dict[int, Dict[str, int]]:
        """Get the complete adjacency graph."""
        return self.adjacency_graph

    def get_shared_edges(self) -> List[SharedEdge]:
        """Get all shared edges between tiles."""
        return self.shared_edges

    def get_atlas_layout(self) -> Dict[int, TilePosition]:
        """Get the atlas layout mapping."""
        return self.atlas_layout

    def get_setup_summary(self) -> Dict[str, Any]:
        """Get a summary of the tileset setup."""
        return {
            "theme": self.theme,
            "tile_count": self.tile_count,
            "atlas_dimensions": f"{self.atlas_columns}x{self.atlas_rows}",
            "tile_size": self.tile_size,
            "sub_tile_size": self.sub_tile_size,
            "total_shared_edges": len(self.shared_edges),
            "adjacency_connections": sum(len(neighbors) for neighbors in self.adjacency_graph.values()),
            "building_blocks_used": self.building_blocks,
            "variations_per_block": self.variations_per_block
        }

def execute(context: PipelineContext) -> Dict[str, Any]:
    """
    Execute Stage 02: Tileset Setup

    Sets up the complete tileset structure including adjacency graph,
    shared edges, and atlas layout for multi-tile diffusion.

    Args:
        context: Pipeline context with validated job spec and extracted config

    Returns:
        Dict with tileset setup results
    """
    logger.info("Starting tileset setup", job_id=context.get_job_id())

    try:
        # Validate context state
        if not context.validate_context_state(2):
            return {"success": False, "errors": context.pipeline_errors}

        # Get extracted configuration
        extracted_config = getattr(context, 'extracted_config', None)
        if not extracted_config:
            context.add_error(2, "No extracted configuration found")
            return {"success": False, "errors": ["No extracted configuration found"]}

        # Create tileset setup
        tileset_setup = TilesetSetup(extracted_config)

        # Store in context for later stages
        context.universal_tileset = tileset_setup  # Keep same name for compatibility
        context.tileset_summary = tileset_setup.get_setup_summary()

        logger.info("Tileset setup completed",
                   job_id=context.get_job_id(),
                   tile_count=tileset_setup.tile_count,
                   shared_edges=len(tileset_setup.shared_edges),
                   atlas_size=f"{tileset_setup.atlas_columns}x{tileset_setup.atlas_rows}")

        return {
            "success": True,
            "tileset_setup": tileset_setup,
            "setup_summary": context.tileset_summary,
            "adjacency_graph": tileset_setup.get_adjacency_graph(),
            "shared_edges": tileset_setup.get_shared_edges(),
            "atlas_layout": tileset_setup.get_atlas_layout()
        }

    except Exception as e:
        error_msg = f"Tileset setup failed: {str(e)}"
        logger.error("Tileset setup failed", job_id=context.get_job_id(), error=error_msg)
        context.add_error(2, error_msg)
        return {"success": False, "errors": [error_msg]}