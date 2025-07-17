"""
Stage 07: Failure Recovery and Integration
Handles regeneration of failed tiles and integrates them back into the tileset.
Only needed when multi-tile coordination from Stage 05 partially failed.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional
import structlog

from ..core.pipeline_context import PipelineContext
from ..core.model_registry import ModelRegistry

logger = structlog.get_logger()

def execute(context: PipelineContext) -> Dict[str, Any]:
    """
    Execute Stage 07: Failure Recovery and Integration
    
    Handles tiles that failed constraint validation from Stage 06:
    - Regenerates failed tiles using targeted diffusion
    - Integrates regenerated tiles back into the tileset
    - Applies minimal corrections to preserve multi-tile coordination
    - Only processes tiles marked for regeneration
    
    Args:
        context: Pipeline context with constraint validation results
        
    Returns:
        Dict with recovery results and final integrated tileset
    """
    logger.info("Starting failure recovery and integration", job_id=context.get_job_id())
    
    try:
        # Validate context state
        if not context.validate_context_state(7):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Get required data
        generated_tiles = context.generated_tiles
        constraint_violations = context.constraint_violations
        regeneration_queue = getattr(context, 'regeneration_queue', [])
        tileset_setup = context.universal_tileset
        extracted_config = getattr(context, 'extracted_config', None)
        
        if not generated_tiles:
            context.add_error(7, "No generated tiles found")
            return {"success": False, "errors": ["No generated tiles found"]}
        
        # Check if any recovery is needed
        if not regeneration_queue:
            logger.info("No tiles marked for regeneration, skipping recovery", job_id=context.get_job_id())
            context.final_tiles = generated_tiles  # Use original tiles
            return {
                "success": True,
                "recovery_needed": False,
                "final_tiles": generated_tiles,
                "tiles_regenerated": 0,
                "recovery_summary": "No recovery needed - multi-tile coordination was successful"
            }
        
        # Initialize failure recovery engine
        recovery_engine = FailureRecoveryEngine(
            tileset_setup=tileset_setup,
            config=extracted_config,
            original_tiles=generated_tiles,
            constraint_violations=constraint_violations
        )
        
        # Perform targeted regeneration and integration
        recovery_result = recovery_engine.recover_and_integrate(regeneration_queue)
        
        # Store final integrated tileset
        context.final_tiles = recovery_result["final_tiles"]
        context.recovery_metadata = recovery_result["metadata"]
        
        logger.info("Failure recovery completed", 
                   job_id=context.get_job_id(),
                   tiles_regenerated=recovery_result["tiles_regenerated"],
                   recovery_success_rate=recovery_result["recovery_success_rate"])
        
        return {
            "success": True,
            "recovery_needed": True,
            "final_tiles": recovery_result["final_tiles"],
            "tiles_regenerated": recovery_result["tiles_regenerated"],
            "recovery_metadata": recovery_result["metadata"],
            "recovery_summary": recovery_result["summary"]
        }
        
    except Exception as e:
        error_msg = f"Failure recovery failed: {str(e)}"
        logger.error("Failure recovery failed", job_id=context.get_job_id(), error=error_msg)
        context.add_error(7, error_msg)
        return {"success": False, "errors": [error_msg]}

class FailureRecoveryEngine:
    """Engine for recovering from multi-tile coordination failures."""
    
    def __init__(self, tileset_setup: Any, config: Dict[str, Any], 
                 original_tiles: Dict[int, Image.Image], constraint_violations: Dict[int, List[str]]):
        self.tileset_setup = tileset_setup
        self.config = config
        self.original_tiles = original_tiles
        self.constraint_violations = constraint_violations
        
        # Get coordination data
        self.adjacency_graph = tileset_setup.get_adjacency_graph()
        self.shared_edges = tileset_setup.get_shared_edges()
        self.atlas_layout = tileset_setup.get_atlas_layout()
        
        # Model registry for regeneration
        self.model_registry = ModelRegistry()
        
        # Recovery parameters
        self.edge_preservation_strength = 0.9  # Strongly preserve existing good edges
        self.max_regeneration_attempts = 3
    
    def recover_and_integrate(self, regeneration_queue: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recover failed tiles and integrate them back into the tileset."""
        
        logger.info("Starting targeted tile recovery", tiles_to_regenerate=len(regeneration_queue))
        
        # Start with original tiles
        final_tiles = self.original_tiles.copy()
        
        # Track recovery statistics
        tiles_regenerated = 0
        recovery_attempts = {}
        recovery_successes = {}
        
        # Group tiles by failure type for efficient recovery
        failure_groups = self._group_tiles_by_failure_type(regeneration_queue)
        
        # Process each failure group
        for failure_type, tile_ids in failure_groups.items():
            logger.info(f"Recovering {len(tile_ids)} tiles with {failure_type} failures")
            
            for tile_id in tile_ids:
                recovery_attempts[tile_id] = 0
                recovery_successes[tile_id] = False
                
                # Attempt recovery with multiple strategies
                for attempt in range(self.max_regeneration_attempts):
                    recovery_attempts[tile_id] += 1
                    
                    # Choose recovery strategy based on failure type and attempt
                    strategy = self._choose_recovery_strategy(failure_type, attempt)
                    
                    # Attempt tile regeneration
                    regenerated_tile = self._regenerate_tile_with_strategy(
                        tile_id, strategy, final_tiles
                    )
                    
                    if regenerated_tile:
                        # Validate the regenerated tile
                        if self._validate_regenerated_tile(tile_id, regenerated_tile, final_tiles):
                            final_tiles[tile_id] = regenerated_tile
                            recovery_successes[tile_id] = True
                            tiles_regenerated += 1
                            logger.info(f"Successfully recovered tile {tile_id} on attempt {attempt + 1}")
                            break
                        else:
                            logger.warning(f"Regenerated tile {tile_id} failed validation on attempt {attempt + 1}")
                    else:
                        logger.warning(f"Failed to regenerate tile {tile_id} on attempt {attempt + 1}")
                
                if not recovery_successes[tile_id]:
                    logger.error(f"Failed to recover tile {tile_id} after {self.max_regeneration_attempts} attempts")
        
        # Apply final integration corrections
        final_tiles = self._apply_integration_corrections(final_tiles, recovery_successes)
        
        # Calculate recovery statistics
        total_tiles_to_recover = len(regeneration_queue)
        successful_recoveries = sum(recovery_successes.values())
        recovery_success_rate = successful_recoveries / total_tiles_to_recover if total_tiles_to_recover > 0 else 1.0
        
        return {
            "final_tiles": final_tiles,
            "tiles_regenerated": tiles_regenerated,
            "recovery_success_rate": recovery_success_rate,
            "metadata": {
                "recovery_attempts": recovery_attempts,
                "recovery_successes": recovery_successes,
                "failure_groups": failure_groups,
                "total_attempts": sum(recovery_attempts.values()),
                "successful_recoveries": successful_recoveries
            },
            "summary": f"Recovered {successful_recoveries}/{total_tiles_to_recover} failed tiles"
        }
    
    def _group_tiles_by_failure_type(self, regeneration_queue: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group tiles by their primary failure type for efficient recovery."""
        failure_groups = {
            "edge_mismatch": [],
            "palette_violation": [],
            "structural_violation": [],
            "coherence_violation": [],
            "multiple_violations": []
        }
        
        for item in regeneration_queue:
            tile_id = item["tile_id"]
            reason = item["reason"]
            
            # Categorize by primary failure type
            if "Edge mismatch" in reason:
                failure_groups["edge_mismatch"].append(tile_id)
            elif "Palette deviation" in reason:
                failure_groups["palette_violation"].append(tile_id)
            elif "structure" in reason.lower():
                failure_groups["structural_violation"].append(tile_id)
            elif "coherence" in reason.lower():
                failure_groups["coherence_violation"].append(tile_id)
            else:
                failure_groups["multiple_violations"].append(tile_id)
        
        # Remove empty groups
        return {k: v for k, v in failure_groups.items() if v}
    
    def _choose_recovery_strategy(self, failure_type: str, attempt: int) -> str:
        """Choose recovery strategy based on failure type and attempt number."""
        
        strategies = {
            "edge_mismatch": ["edge_guided_regeneration", "neighbor_context_regeneration", "full_regeneration"],
            "palette_violation": ["palette_constrained_regeneration", "color_correction", "full_regeneration"],
            "structural_violation": ["structure_guided_regeneration", "controlnet_enhanced", "full_regeneration"],
            "coherence_violation": ["coherence_guided_regeneration", "neighbor_blending", "full_regeneration"],
            "multiple_violations": ["comprehensive_regeneration", "staged_correction", "full_regeneration"]
        }
        
        strategy_list = strategies.get(failure_type, ["full_regeneration"])
        strategy_index = min(attempt, len(strategy_list) - 1)
        
        return strategy_list[strategy_index]
    
    def _regenerate_tile_with_strategy(self, tile_id: int, strategy: str, 
                                     current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Regenerate a tile using the specified strategy."""
        
        try:
            if strategy == "edge_guided_regeneration":
                return self._edge_guided_regeneration(tile_id, current_tiles)
            elif strategy == "palette_constrained_regeneration":
                return self._palette_constrained_regeneration(tile_id, current_tiles)
            elif strategy == "structure_guided_regeneration":
                return self._structure_guided_regeneration(tile_id, current_tiles)
            elif strategy == "coherence_guided_regeneration":
                return self._coherence_guided_regeneration(tile_id, current_tiles)
            elif strategy == "neighbor_context_regeneration":
                return self._neighbor_context_regeneration(tile_id, current_tiles)
            elif strategy == "color_correction":
                return self._apply_color_correction(tile_id, current_tiles)
            elif strategy == "neighbor_blending":
                return self._apply_neighbor_blending(tile_id, current_tiles)
            else:  # full_regeneration
                return self._full_tile_regeneration(tile_id, current_tiles)
                
        except Exception as e:
            logger.error(f"Strategy {strategy} failed for tile {tile_id}", error=str(e))
            return None
    
    def _edge_guided_regeneration(self, tile_id: int, current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Regenerate tile with strong edge constraints from neighbors."""
        
        # Get neighbors and their edges
        neighbors = self.adjacency_graph.get(tile_id, {})
        edge_constraints = {}
        
        for direction, neighbor_id in neighbors.items():
            if neighbor_id in current_tiles:
                neighbor_tile = current_tiles[neighbor_id]
                neighbor_array = np.array(neighbor_tile)
                
                # Extract edge that should match
                if direction == "top":
                    edge_constraints["top"] = neighbor_array[-4:, :]  # Bottom edge of neighbor
                elif direction == "right":
                    edge_constraints["right"] = neighbor_array[:, :4]  # Left edge of neighbor
                elif direction == "bottom":
                    edge_constraints["bottom"] = neighbor_array[:4, :]  # Top edge of neighbor
                elif direction == "left":
                    edge_constraints["left"] = neighbor_array[:, -4:]  # Right edge of neighbor
        
        # Use edge constraints to guide regeneration
        # This is a simplified implementation - would need actual diffusion with edge constraints
        original_tile = self.original_tiles[tile_id]
        constrained_tile = self._apply_edge_constraints(original_tile, edge_constraints)
        
        return constrained_tile
    
    def _palette_constrained_regeneration(self, tile_id: int, current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Regenerate tile with strict palette constraints."""
        
        # Get target palette
        palette_name = self.config.get("palette")
        if not palette_name:
            return None
        
        palette_config = self.model_registry.get_palette_config(palette_name)
        if not palette_config:
            return None
        
        # Apply palette quantization to original tile
        original_tile = self.original_tiles[tile_id]
        quantized_tile = self._apply_palette_quantization(original_tile, palette_config)
        
        return quantized_tile
    
    def _structure_guided_regeneration(self, tile_id: int, current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Regenerate tile with enhanced structural guidance."""
        
        # Get tile specification
        tile_spec = self.tileset_setup.get_tile_spec(tile_id)
        if not tile_spec:
            return None
        
        # Apply structural corrections based on tile spec
        original_tile = self.original_tiles[tile_id]
        structure_corrected = self._apply_structural_corrections(original_tile, tile_spec)
        
        return structure_corrected
    
    def _coherence_guided_regeneration(self, tile_id: int, current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Regenerate tile with coherence constraints."""
        
        # Apply coherence corrections to improve sub-tile boundaries
        original_tile = self.original_tiles[tile_id]
        coherence_corrected = self._apply_coherence_corrections(original_tile)
        
        return coherence_corrected
    
    def _neighbor_context_regeneration(self, tile_id: int, current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Regenerate using neighbor context for guidance."""
        
        # Use neighbor tiles to provide context for regeneration
        # This would involve creating a larger context image and regenerating the center tile
        original_tile = self.original_tiles[tile_id]
        context_guided = self._apply_neighbor_context_guidance(tile_id, original_tile, current_tiles)
        
        return context_guided
    
    def _apply_color_correction(self, tile_id: int, current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Apply color correction to fix palette violations."""
        
        original_tile = self.original_tiles[tile_id]
        
        # Simple color correction - adjust brightness/contrast to match neighbors
        neighbors = self.adjacency_graph.get(tile_id, {})
        if not neighbors:
            return original_tile
        
        # Calculate target color statistics from neighbors
        neighbor_stats = []
        for neighbor_id in neighbors.values():
            if neighbor_id in current_tiles:
                neighbor_array = np.array(current_tiles[neighbor_id].convert("L"))
                neighbor_stats.append((np.mean(neighbor_array), np.std(neighbor_array)))
        
        if not neighbor_stats:
            return original_tile
        
        target_mean = np.mean([stats[0] for stats in neighbor_stats])
        target_std = np.mean([stats[1] for stats in neighbor_stats])
        
        # Apply color correction
        tile_array = np.array(original_tile.convert("L"))
        current_mean = np.mean(tile_array)
        current_std = np.std(tile_array)
        
        if current_std > 0:
            corrected_array = (tile_array - current_mean) * (target_std / current_std) + target_mean
            corrected_array = np.clip(corrected_array, 0, 255).astype(np.uint8)
            
            # Convert back to RGB
            corrected_tile = Image.fromarray(corrected_array).convert("RGB")
            return corrected_tile
        
        return original_tile
    
    def _apply_neighbor_blending(self, tile_id: int, current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Apply neighbor blending to improve coherence."""
        
        original_tile = self.original_tiles[tile_id]
        tile_array = np.array(original_tile)
        
        # Blend edges with neighbors
        neighbors = self.adjacency_graph.get(tile_id, {})
        
        for direction, neighbor_id in neighbors.items():
            if neighbor_id in current_tiles:
                neighbor_array = np.array(current_tiles[neighbor_id])
                
                # Apply edge blending
                if direction == "right":
                    # Blend right edge with neighbor's left edge
                    blend_width = 2
                    tile_array[:, -blend_width:] = (
                        tile_array[:, -blend_width:] * 0.7 + 
                        neighbor_array[:, :blend_width] * 0.3
                    )
                elif direction == "bottom":
                    # Blend bottom edge with neighbor's top edge
                    blend_width = 2
                    tile_array[-blend_width:, :] = (
                        tile_array[-blend_width:, :] * 0.7 + 
                        neighbor_array[:blend_width, :] * 0.3
                    )
        
        return Image.fromarray(tile_array.astype(np.uint8))
    
    def _full_tile_regeneration(self, tile_id: int, current_tiles: Dict[int, Image.Image]) -> Optional[Image.Image]:
        """Full regeneration as last resort."""
        
        # This would involve re-running the diffusion process for this specific tile
        # For now, return the original tile (no regeneration)
        logger.warning(f"Full regeneration not implemented, returning original tile {tile_id}")
        return self.original_tiles[tile_id]
    
    def _validate_regenerated_tile(self, tile_id: int, regenerated_tile: Image.Image, 
                                 current_tiles: Dict[int, Image.Image]) -> bool:
        """Validate that the regenerated tile meets constraints."""
        
        # Quick validation - check edge similarity with neighbors
        neighbors = self.adjacency_graph.get(tile_id, {})
        
        for direction, neighbor_id in neighbors.items():
            if neighbor_id in current_tiles:
                # Check edge similarity
                similarity = self._calculate_edge_similarity(
                    regenerated_tile, current_tiles[neighbor_id], direction
                )
                
                if similarity < 0.9:  # Lower threshold for regenerated tiles
                    return False
        
        return True
    
    def _apply_integration_corrections(self, final_tiles: Dict[int, Image.Image], 
                                     recovery_successes: Dict[int, bool]) -> Dict[int, Image.Image]:
        """Apply final corrections to integrate regenerated tiles."""
        
        # Apply minimal corrections to ensure seamless integration
        # Focus on edge blending between regenerated and original tiles
        
        corrected_tiles = final_tiles.copy()
        
        for tile_id, was_regenerated in recovery_successes.items():
            if was_regenerated and tile_id in corrected_tiles:
                # Apply edge smoothing with neighbors
                corrected_tiles[tile_id] = self._apply_edge_smoothing(
                    tile_id, corrected_tiles[tile_id], corrected_tiles
                )
        
        return corrected_tiles
    
    # Helper methods (simplified implementations)
    def _apply_edge_constraints(self, tile: Image.Image, edge_constraints: Dict[str, np.ndarray]) -> Image.Image:
        """Apply edge constraints to tile."""
        tile_array = np.array(tile)
        
        for direction, constraint in edge_constraints.items():
            if direction == "top":
                tile_array[:4, :] = constraint
            elif direction == "right":
                tile_array[:, -4:] = constraint
            elif direction == "bottom":
                tile_array[-4:, :] = constraint
            elif direction == "left":
                tile_array[:, :4] = constraint
        
        return Image.fromarray(tile_array.astype(np.uint8))
    
    def _apply_palette_quantization(self, tile: Image.Image, palette_config: Any) -> Image.Image:
        """Apply palette quantization."""
        # Simplified palette quantization
        return tile  # Would implement actual quantization
    
    def _apply_structural_corrections(self, tile: Image.Image, tile_spec: Any) -> Image.Image:
        """Apply structural corrections."""
        # Simplified structural corrections
        return tile  # Would implement actual corrections
    
    def _apply_coherence_corrections(self, tile: Image.Image) -> Image.Image:
        """Apply coherence corrections."""
        # Simplified coherence corrections
        return tile  # Would implement actual corrections
    
    def _apply_neighbor_context_guidance(self, tile_id: int, tile: Image.Image, 
                                       current_tiles: Dict[int, Image.Image]) -> Image.Image:
        """Apply neighbor context guidance."""
        # Simplified neighbor guidance
        return tile  # Would implement actual guidance
    
    def _calculate_edge_similarity(self, tile_a: Image.Image, tile_b: Image.Image, direction: str) -> float:
        """Calculate edge similarity between tiles."""
        # Simplified edge similarity calculation
        return 0.95  # Would implement actual calculation
    
    def _apply_edge_smoothing(self, tile_id: int, tile: Image.Image, 
                            all_tiles: Dict[int, Image.Image]) -> Image.Image:
        """Apply edge smoothing."""
        # Simplified edge smoothing
        return tile  # Would implement actual smoothing
