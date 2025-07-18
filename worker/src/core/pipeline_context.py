"""
Pipeline Context System - Shared state management for the tile generation pipeline.
All data flows through this context object between pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image
import structlog

logger = structlog.get_logger()

@dataclass
class PipelineContext:
    """Central context object that flows through all pipeline stages."""

    # Input job specification
    job_spec: Dict[str, Any]
    output_path: Path
    
    # Stage 01: Job validation results
    validated_job_spec: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = field(default_factory=list)
    
    # Stage 02: Universal tileset structure
    universal_tileset: Optional[Any] = None  # UniversalTileset object
    tileset_summary: Optional[Dict[str, Any]] = None
    
    # Stage 03: Perspective and lighting setup
    perspective_params: Optional[Dict[str, Any]] = None
    lighting_config: Optional[Dict[str, Any]] = None
    
    # Stage 04: Reference synthesis results
    reference_maps: Dict[int, Dict[str, Image.Image]] = field(default_factory=dict)
    control_images: Dict[int, Image.Image] = field(default_factory=dict)
    
    # Stage 05: Diffusion generation results
    generated_tiles: Dict[int, Image.Image] = field(default_factory=dict)
    generation_metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Stage 06: Constraint enforcement results
    constraint_violations: Dict[int, List[str]] = field(default_factory=dict)
    constraint_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Stage 07: Failure recovery results
    final_tiles: Dict[int, Image.Image] = field(default_factory=dict)
    recovery_metadata: Optional[Dict[str, Any]] = None
    
    # Stage 08: Post-processing results
    processed_tiles: Dict[int, Image.Image] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Stage 09: Atlas generation results
    final_atlas: Optional[Image.Image] = None
    atlas_metadata: Optional[Dict[str, Any]] = None
    tile_positions: Dict[int, Dict[str, int]] = field(default_factory=dict)
    
    # Stage 10: Quality validation results
    quality_scores: Dict[str, float] = field(default_factory=dict)
    regeneration_queue: List[Dict[str, Any]] = field(default_factory=list)
    final_validation_passed: bool = False
    
    # Pipeline execution state
    current_stage: int = 0
    stage_timings: Dict[int, float] = field(default_factory=dict)
    pipeline_errors: List[str] = field(default_factory=list)
    
    def get_job_id(self) -> str:
        """Get the job ID from the job spec."""
        return self.job_spec.get("id", "unknown")
    
    def get_theme(self) -> str:
        """Get the theme from the job spec."""
        return self.job_spec.get("theme", "default")
    
    def get_palette(self) -> str:
        """Get the palette from the job spec."""
        return self.job_spec.get("palette", "default")
    
    def get_tile_count(self) -> int:
        """Get the number of tiles to generate."""
        return self.job_spec.get("tileCount", 16)
    
    def get_tile_size(self) -> int:
        """Get the size of each tile in pixels."""
        return self.job_spec.get("tileSize", 32)
    
    def add_error(self, stage: int, error: str):
        """Add an error for a specific stage."""
        error_msg = f"Stage {stage:02d}: {error}"
        self.pipeline_errors.append(error_msg)
        logger.error("Pipeline error", stage=stage, error=error, job_id=self.get_job_id())
    
    def set_stage_timing(self, stage: int, duration: float):
        """Record timing for a pipeline stage."""
        self.stage_timings[stage] = duration
        logger.info("Stage completed", stage=stage, duration=duration, job_id=self.get_job_id())
    
    def has_errors(self) -> bool:
        """Check if there are any pipeline errors."""
        return len(self.pipeline_errors) > 0 or len(self.validation_errors) > 0
    
    def get_latest_tiles(self) -> Dict[int, Image.Image]:
        """Get the most recent version of tiles from the pipeline."""
        if self.processed_tiles:
            return self.processed_tiles
        elif self.final_tiles:
            return self.final_tiles
        elif self.generated_tiles:
            return self.generated_tiles
        else:
            return {}
    
    def get_tile_spec(self, tile_id: int) -> Optional[Dict[str, Any]]:
        """Get the generation specification for a specific tile."""
        if self.universal_tileset:
            return self.universal_tileset.get_tile_generation_spec(tile_id)
        return None
    
    def mark_tile_for_regeneration(self, tile_id: int, reason: str):
        """Mark a tile for regeneration due to quality issues."""
        # Check if tile is already in queue
        existing_entry = next((item for item in self.regeneration_queue if item["tile_id"] == tile_id), None)

        if not existing_entry:
            self.regeneration_queue.append({
                "tile_id": tile_id,
                "reason": reason,
                "attempts": 0
            })
        else:
            # Update reason if more specific
            existing_entry["reason"] = f"{existing_entry['reason']}; {reason}"

        if tile_id not in self.constraint_violations:
            self.constraint_violations[tile_id] = []
        self.constraint_violations[tile_id].append(reason)

        logger.warning("Tile marked for regeneration",
                      tile_id=tile_id,
                      reason=reason,
                      job_id=self.get_job_id())
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire pipeline execution."""
        total_time = sum(self.stage_timings.values())
        
        return {
            "job_id": self.get_job_id(),
            "theme": self.get_theme(),
            "palette": self.get_palette(),
            "tile_count": self.get_tile_count(),
            "tile_size": self.get_tile_size(),
            "current_stage": self.current_stage,
            "total_execution_time": total_time,
            "stage_timings": self.stage_timings,
            "errors": self.pipeline_errors,
            "validation_errors": self.validation_errors,
            "tiles_generated": len(self.generated_tiles),
            "tiles_processed": len(self.processed_tiles),
            "constraint_violations": len([v for violations in self.constraint_violations.values() for v in violations]),
            "tiles_for_regeneration": len(self.regeneration_queue),
            "final_validation_passed": self.final_validation_passed,
            "has_final_atlas": self.final_atlas is not None,
            "output_path": str(self.output_path)
        }
    
    def log_stage_start(self, stage: int, stage_name: str):
        """Log the start of a pipeline stage."""
        self.current_stage = stage
        logger.info("Starting pipeline stage", 
                   stage=stage, 
                   stage_name=stage_name, 
                   job_id=self.get_job_id())
    
    def log_stage_complete(self, stage: int, stage_name: str, duration: float):
        """Log the completion of a pipeline stage."""
        self.set_stage_timing(stage, duration)
        logger.info("Pipeline stage completed", 
                   stage=stage, 
                   stage_name=stage_name, 
                   duration=duration,
                   job_id=self.get_job_id())
    
    def validate_context_state(self, expected_stage: int) -> bool:
        """Validate that the context is in the expected state for a given stage."""
        if self.current_stage < expected_stage - 1:
            self.add_error(expected_stage, f"Context not ready - current stage is {self.current_stage}")
            return False
        
        # Stage-specific validation
        if expected_stage >= 3 and not self.universal_tileset:
            self.add_error(expected_stage, "Universal tileset not generated")
            return False

        if expected_stage >= 5 and not self.reference_maps:
            self.add_error(expected_stage, "Reference maps not generated")
            return False

        if expected_stage >= 7 and not self.generated_tiles:
            self.add_error(expected_stage, "Tiles not generated")
            return False
        
        return True
