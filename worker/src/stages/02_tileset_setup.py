"""
Stage 2: Tileset Setup
Sets up proper tessellation tileset structure based on Wang tiles/blob tiles methodology.
Creates tile specifications with correct connectivity patterns for seamless tessellation.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import structlog

from ..core.pipeline_context import PipelineContext

logger = structlog.get_logger()


@dataclass
class TilePosition:
    """Position of a tile in the atlas grid."""
    x: int
    y: int
    atlas_index: int


@dataclass
class SharedEdge:
    """Specification for an edge shared between two tiles."""
    tile_a_id: int
    tile_a_edge: str  # "top", "bottom", "left", "right"
    tile_b_id: int
    tile_b_edge: str
    edge_length: int  # Length in pixels
    pattern_id: str   # Unique identifier for the edge pattern


@dataclass
class TileSpec:
    """Complete specification for a tessellation tile."""
    tile_id: int
    position: TilePosition
    tile_type: str  # "corner", "edge", "t_junction", "cross", "end"
    connectivity_pattern: Set[str]  # Which directions connect: {"top", "right", "bottom", "left"}
    structure_composition: Dict[str, str]  # sub-tile quadrant -> structure type
    seamless_edges: List[str]  # Which edges must match neighbors exactly
    shared_edges: List[SharedEdge]  # Specific edge sharing requirements
    neighbors: Dict[str, int]  # direction -> neighbor_tile_id
    edge_patterns: Dict[str, str]  # direction -> pattern_id for Wang tile matching


class TilesetSetup:
    """Sets up proper tessellation tileset based on Wang tiles methodology."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tile_size = config.get("tile_size", 64)  # Match JobSpec field name
        self.tileset_type = config.get("tileset_type", "minimal")

        # Calculate optimal sub_tile_size based on tileset_type and tile_size
        self.sub_tile_size = self._calculate_optimal_sub_tile_size()
        
        # Calculate sub-tile grid dimensions
        self.sub_tiles_per_row = self.tile_size // self.sub_tile_size
        self.sub_tiles_per_col = self.tile_size // self.sub_tile_size

        logger.info("Calculated tessellation parameters",
                   tile_size=self.tile_size,
                   sub_tile_size=self.sub_tile_size,
                   tileset_type=self.tileset_type,
                   sub_tile_grid=f"{self.sub_tiles_per_row}x{self.sub_tiles_per_col}")
        
        # Tessellation specifications
        self.tile_count = self._get_tessellation_tile_count()
        self.atlas_columns, self.atlas_rows = self._calculate_atlas_dimensions()
        
        # Tile specifications and relationships
        self.tile_specs: Dict[int, TileSpec] = {}
        self.shared_edges: List[SharedEdge] = []
        self.edge_patterns: Dict[str, List[int]] = {}  # pattern_id -> list of tile_ids using it

    def _calculate_optimal_sub_tile_size(self) -> int:
        """Calculate optimal sub-tile size based on tileset_type and tile_size."""

        if self.tileset_type == "minimal":
            # Minimal: 2x2 sub-tile grid for simple tessellation
            return self.tile_size // 2
        elif self.tileset_type == "extended":
            # Extended: 4x4 sub-tile grid for more precision
            return self.tile_size // 4
        elif self.tileset_type == "full":
            # Full: 8x8 sub-tile grid for maximum precision
            return self.tile_size // 8
        else:
            # Default to minimal approach
            return self.tile_size // 2
        
    def _get_tessellation_tile_count(self) -> int:
        """Get the number of tiles needed for proper tessellation."""
        tessellation_counts = {
            "minimal": 13,    # 4 corners + 4 edges + 4 t-junctions + 1 cross
            "extended": 47,   # Includes variations and transitions
            "full": 256       # Complete Wang tile set
        }
        return tessellation_counts.get(self.tileset_type, 13)
    
    def _calculate_atlas_dimensions(self) -> Tuple[int, int]:
        """Calculate optimal atlas dimensions for the tile count."""
        # Find dimensions that minimize wasted space
        import math
        sqrt_count = math.sqrt(self.tile_count)
        columns = math.ceil(sqrt_count)
        rows = math.ceil(self.tile_count / columns)
        return columns, rows
    
    def setup_tessellation_tileset(self) -> Dict[str, Any]:
        """Set up complete tessellation tileset with proper connectivity."""
        logger.info("Setting up tessellation tileset",
                   tileset_type=self.tileset_type,
                   tile_count=self.tile_count,
                   tile_size=self.tile_size,
                   sub_tile_size=self.sub_tile_size)
        
        # Generate tessellation patterns based on type
        if self.tileset_type == "minimal":
            self._setup_minimal_tessellation()
        elif self.tileset_type == "extended":
            self._setup_extended_tessellation()
        elif self.tileset_type == "full":
            self._setup_full_wang_tiles()
        else:
            raise ValueError(f"Unknown tileset type: {self.tileset_type}")
        
        # Generate shared edges and connectivity
        self._generate_shared_edges()
        
        # Validate tessellation completeness
        self._validate_tessellation()
        
        return {
            "tile_specs": self.tile_specs,
            "shared_edges": self.shared_edges,
            "edge_patterns": self.edge_patterns,
            "atlas_dimensions": (self.atlas_columns, self.atlas_rows),
            "tessellation_summary": {
                "total_tiles": len(self.tile_specs),
                "tile_types": self._count_tile_types(),
                "edge_patterns": len(self.edge_patterns),
                "shared_edges": len(self.shared_edges)
            }
        }
    
    def _setup_minimal_tessellation(self):
        """Set up minimal 13-tile tessellation set."""
        tile_id = 0
        
        # 4 Corner tiles (2 connections each, perpendicular)
        corners = [
            ({"top", "right"}, "corner_ne"),
            ({"top", "left"}, "corner_nw"), 
            ({"bottom", "right"}, "corner_se"),
            ({"bottom", "left"}, "corner_sw")
        ]
        
        for connectivity, corner_type in corners:
            self._create_tile_spec(tile_id, "corner", connectivity, corner_type)
            tile_id += 1
        
        # 4 Edge tiles (1 connection each, dead ends)
        edges = [
            ({"top"}, "edge_north"),
            ({"bottom"}, "edge_south"),
            ({"left"}, "edge_west"), 
            ({"right"}, "edge_east")
        ]
        
        for connectivity, edge_type in edges:
            self._create_tile_spec(tile_id, "edge", connectivity, edge_type)
            tile_id += 1
        
        # 4 T-junction tiles (3 connections each)
        t_junctions = [
            ({"left", "right", "bottom"}, "t_north"),  # Missing top
            ({"left", "right", "top"}, "t_south"),     # Missing bottom
            ({"top", "bottom", "right"}, "t_west"),    # Missing left
            ({"top", "bottom", "left"}, "t_east")      # Missing right
        ]
        
        for connectivity, t_type in t_junctions:
            self._create_tile_spec(tile_id, "t_junction", connectivity, t_type)
            tile_id += 1
        
        # 1 Cross tile (4 connections, all directions)
        self._create_tile_spec(tile_id, "cross", {"top", "bottom", "left", "right"}, "cross_center")
    
    def _create_tile_spec(self, tile_id: int, tile_type: str, connectivity: Set[str], type_name: str):
        """Create a complete tile specification with proper tessellation data."""
        # Calculate atlas position
        atlas_x = tile_id % self.atlas_columns
        atlas_y = tile_id // self.atlas_columns
        position = TilePosition(atlas_x, atlas_y, tile_id)
        
        # Generate structure composition based on connectivity and sub-tile grid
        structure_composition = self._generate_structure_composition(connectivity, tile_type)
        
        # Determine seamless edges (all connected edges must be seamless)
        seamless_edges = list(connectivity)
        
        # Generate edge patterns for Wang tile matching
        edge_patterns = self._generate_edge_patterns(tile_id, connectivity, tile_type)
        
        # Create neighbors dict (will be populated when generating shared edges)
        neighbors = {}
        
        # Create tile specification
        tile_spec = TileSpec(
            tile_id=tile_id,
            position=position,
            tile_type=tile_type,
            connectivity_pattern=connectivity,
            structure_composition=structure_composition,
            seamless_edges=seamless_edges,
            shared_edges=[],  # Will be populated in _generate_shared_edges
            neighbors=neighbors,
            edge_patterns=edge_patterns
        )
        
        self.tile_specs[tile_id] = tile_spec
        logger.debug("Created tile spec", 
                    tile_id=tile_id, 
                    tile_type=tile_type,
                    connectivity=list(connectivity))
    
    def _generate_structure_composition(self, connectivity: Set[str], tile_type: str) -> Dict[str, str]:
        """Generate structure composition for sub-tile quadrants."""
        # Map sub-tile positions to structure types based on connectivity
        composition = {}
        
        # For dynamic sub-tile grid
        for row in range(self.sub_tiles_per_row):
            for col in range(self.sub_tiles_per_col):
                quadrant_key = f"sub_{row}_{col}"
                
                # Determine structure type based on position and connectivity
                if tile_type == "corner":
                    # Corner tiles have corner structure in the connecting quadrant
                    if (row == 0 and col == 0 and {"top", "left"}.issubset(connectivity)) or \
                       (row == 0 and col == self.sub_tiles_per_col-1 and {"top", "right"}.issubset(connectivity)) or \
                       (row == self.sub_tiles_per_row-1 and col == 0 and {"bottom", "left"}.issubset(connectivity)) or \
                       (row == self.sub_tiles_per_row-1 and col == self.sub_tiles_per_col-1 and {"bottom", "right"}.issubset(connectivity)):
                        composition[quadrant_key] = "CORNER"
                    else:
                        composition[quadrant_key] = "EDGE"
                        
                elif tile_type == "edge":
                    # Edge tiles have edge structure
                    composition[quadrant_key] = "EDGE"
                    
                elif tile_type == "t_junction":
                    # T-junction tiles have mixed structure
                    composition[quadrant_key] = "T_JUNCTION"
                    
                elif tile_type == "cross":
                    # Cross tiles have center structure
                    composition[quadrant_key] = "CENTER"
                    
                else:
                    composition[quadrant_key] = "GENERIC"
        
        return composition
    
    def _generate_edge_patterns(self, tile_id: int, connectivity: Set[str], tile_type: str) -> Dict[str, str]:
        """Generate edge patterns for Wang tile matching."""
        edge_patterns = {}
        
        # Assign pattern IDs based on tile type and connectivity
        for direction in ["top", "bottom", "left", "right"]:
            if direction in connectivity:
                # Connected edges get specific patterns based on tile type
                pattern_id = f"{tile_type}_{direction}_{len(connectivity)}"
                edge_patterns[direction] = pattern_id
                
                # Track which tiles use this pattern
                if pattern_id not in self.edge_patterns:
                    self.edge_patterns[pattern_id] = []
                self.edge_patterns[pattern_id].append(tile_id)
            else:
                # Non-connected edges get "free" pattern
                edge_patterns[direction] = "free"
        
        return edge_patterns

    def _generate_shared_edges(self):
        """Generate shared edges between tiles that must match exactly."""
        # For minimal tessellation, define explicit connectivity patterns
        # This ensures proper tessellation where tiles can actually connect

        if self.tileset_type == "minimal":
            self._setup_minimal_connectivity()
        else:
            # For extended/full, use pattern matching
            self._setup_pattern_based_connectivity()

    def _setup_minimal_connectivity(self):
        """Set up explicit connectivity for minimal 13-tile set."""
        # Define which tiles connect to which for proper tessellation
        # Corners connect to edges and T-junctions
        # Edges connect to corners, T-junctions, and cross
        # T-junctions connect to everything
        # Cross connects to everything

        connectivity_rules = {
            # Corners (tile_ids 0-3) - only connect in their connectivity directions
            0: {"right": 7, "top": 4},      # corner_ne -> edge_east, edge_north
            1: {"left": 6, "top": 4},       # corner_nw -> edge_west, edge_north
            2: {"right": 7, "bottom": 5},   # corner_se -> edge_east, edge_south
            3: {"left": 6, "bottom": 5},    # corner_sw -> edge_west, edge_south

            # Edges (tile_ids 4-7) - DON'T connect in their edge direction!
            4: {},  # edge_north (top) - no neighbors, it's a dead end
            5: {},  # edge_south (bottom) - no neighbors, it's a dead end
            6: {},  # edge_west (left) - no neighbors, it's a dead end
            7: {},  # edge_east (right) - no neighbors, it's a dead end

            # T-junctions (tile_ids 8-11) connect to cross and edges
            8: {"left": 6, "right": 7, "bottom": 5},    # t_north -> edges
            9: {"left": 6, "right": 7, "top": 4},       # t_south -> edges
            10: {"top": 4, "bottom": 5, "right": 7},    # t_east -> edges
            11: {"top": 4, "bottom": 5, "left": 6},     # t_west -> edges

            # Cross (tile_id 12) connects to all T-junctions
            12: {"top": 9, "bottom": 8, "left": 11, "right": 10}  # cross -> t-junctions
        }

        # Apply connectivity rules
        for tile_id, connections in connectivity_rules.items():
            if tile_id in self.tile_specs:
                tile_spec = self.tile_specs[tile_id]
                for direction, neighbor_id in connections.items():
                    if direction in tile_spec.connectivity_pattern:
                        # Set neighbor
                        tile_spec.neighbors[direction] = neighbor_id

                        # Create shared edge
                        opposite_direction = self._get_opposite_direction(direction)
                        pattern_id = tile_spec.edge_patterns[direction]

                        shared_edge = SharedEdge(
                            tile_a_id=tile_id,
                            tile_a_edge=direction,
                            tile_b_id=neighbor_id,
                            tile_b_edge=opposite_direction,
                            edge_length=self.tile_size,
                            pattern_id=pattern_id
                        )

                        self.shared_edges.append(shared_edge)
                        tile_spec.shared_edges.append(shared_edge)

    def _setup_pattern_based_connectivity(self):
        """Set up connectivity based on pattern matching for extended/full sets."""
        # This is the original pattern-based approach for complex tilesets
        for tile_id, tile_spec in self.tile_specs.items():
            for direction in tile_spec.connectivity_pattern:
                if direction not in tile_spec.neighbors:  # Only if not already set
                    pattern_id = tile_spec.edge_patterns[direction]

                    # Find matching tiles
                    opposite_direction = self._get_opposite_direction(direction)

                    for other_id, other_spec in self.tile_specs.items():
                        if (other_id != tile_id and
                            opposite_direction in other_spec.connectivity_pattern and
                            opposite_direction not in other_spec.neighbors):

                            other_pattern = other_spec.edge_patterns[opposite_direction]
                            if self._patterns_can_connect(pattern_id, other_pattern):
                                # Create connection
                                tile_spec.neighbors[direction] = other_id
                                other_spec.neighbors[opposite_direction] = tile_id

                                shared_edge = SharedEdge(
                                    tile_a_id=tile_id,
                                    tile_a_edge=direction,
                                    tile_b_id=other_id,
                                    tile_b_edge=opposite_direction,
                                    edge_length=self.tile_size,
                                    pattern_id=pattern_id
                                )

                                self.shared_edges.append(shared_edge)
                                tile_spec.shared_edges.append(shared_edge)
                                other_spec.shared_edges.append(shared_edge)
                                break

    def _get_opposite_direction(self, direction: str) -> str:
        """Get the opposite direction for edge matching."""
        opposites = {
            "top": "bottom",
            "bottom": "top",
            "left": "right",
            "right": "left"
        }
        return opposites[direction]

    def _patterns_can_connect(self, pattern_a: str, pattern_b: str) -> bool:
        """Determine if two edge patterns can connect."""
        # For now, same pattern types can connect
        # In a full implementation, this would have complex matching rules
        if pattern_a == "free" or pattern_b == "free":
            return False

        # Extract tile types from pattern IDs
        type_a = pattern_a.split("_")[0] if "_" in pattern_a else pattern_a
        type_b = pattern_b.split("_")[0] if "_" in pattern_b else pattern_b

        # Define which tile types can connect
        compatible_connections = {
            "corner": ["edge", "t_junction", "cross"],
            "edge": ["corner", "t_junction", "cross"],
            "t_junction": ["corner", "edge", "cross"],
            "cross": ["corner", "edge", "t_junction", "cross"]
        }

        return type_b in compatible_connections.get(type_a, [])

    def _validate_tessellation(self):
        """Validate that the tessellation is complete and correct."""
        errors = []

        # Check that all tiles have proper connectivity
        for tile_id, tile_spec in self.tile_specs.items():
            if not tile_spec.connectivity_pattern:
                errors.append(f"Tile {tile_id} has no connectivity")

            # Check that connected edges have neighbors
            for direction in tile_spec.connectivity_pattern:
                if direction not in tile_spec.neighbors:
                    errors.append(f"Tile {tile_id} missing neighbor in direction {direction}")

        # Check edge pattern consistency
        for pattern_id, tile_list in self.edge_patterns.items():
            if len(tile_list) < 2 and pattern_id != "free":
                errors.append(f"Edge pattern {pattern_id} used by only one tile")

        if errors:
            logger.warning("Tessellation validation issues", errors=errors)
        else:
            logger.info("Tessellation validation passed")

    def _count_tile_types(self) -> Dict[str, int]:
        """Count tiles by type for summary."""
        type_counts = {}
        for tile_spec in self.tile_specs.values():
            tile_type = tile_spec.tile_type
            type_counts[tile_type] = type_counts.get(tile_type, 0) + 1
        return type_counts

    def get_tile_spec(self, tile_id: int) -> Optional[TileSpec]:
        """Get tile specification by ID."""
        return self.tile_specs.get(tile_id)

    def _setup_extended_tessellation(self):
        """Set up extended tessellation with more tile variations."""
        # Start with minimal set
        self._setup_minimal_tessellation()

        # Add variations and transitions (placeholder for now)
        logger.info("Extended tessellation not fully implemented yet")

    def _setup_full_wang_tiles(self):
        """Set up complete Wang tile set."""
        # Full 256-tile Wang set (placeholder for now)
        logger.info("Full Wang tiles not implemented yet")


def execute(context: PipelineContext) -> Dict[str, Any]:
    """Execute tileset setup stage."""
    try:
        logger.info("Starting tileset setup", job_id=context.get_job_id())

        # Validate context state
        if not context.validate_context_state(2):
            return {"success": False, "errors": context.pipeline_errors}

        # Get extracted config
        extracted_config = getattr(context, 'extracted_config', {})
        if not extracted_config:
            context.add_error(2, "Missing extracted configuration")
            return {"success": False, "errors": ["Missing extracted configuration"]}

        # Create tileset setup
        tileset_setup = TilesetSetup(extracted_config)

        # Set up tessellation tileset
        setup_result = tileset_setup.setup_tessellation_tileset()

        # Store in context
        context.universal_tileset = tileset_setup

        # Update context stage
        context.current_stage = 2

        logger.info("Tileset setup completed",
                   job_id=context.get_job_id(),
                   tile_count=setup_result["tessellation_summary"]["total_tiles"],
                   shared_edges=setup_result["tessellation_summary"]["shared_edges"],
                   atlas_size=f"{tileset_setup.atlas_columns}x{tileset_setup.atlas_rows}")

        return {
            "success": True,
            "data": setup_result
        }

    except Exception as e:
        error_msg = f"Tileset setup failed: {str(e)}"
        logger.error("Tileset setup failed", error=error_msg, job_id=context.get_job_id())
        context.add_error(2, error_msg)
        return {"success": False, "errors": [error_msg]}
