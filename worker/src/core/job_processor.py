"""
Job processor that orchestrates the complete tile generation pipeline.
Handles configuration extraction and executes stages 01-10 in sequence.
"""

import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import structlog

from .pipeline_context import PipelineContext

logger = structlog.get_logger()

class JobProcessor:
    """Main job processor that orchestrates the 10-stage tile generation pipeline."""
    
    def __init__(self, jobs_dir: str = "./jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.queue_dir = self.jobs_dir / "queue"
        self.status_dir = self.jobs_dir / "status"
        self.output_dir = self.jobs_dir / "output"
        
        # Ensure directories exist
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline stages in execution order
        self.pipeline_stages = [
            (1, "job_validation", "Validating job specification"),
            (2, "tileset_generation", "Generating universal tileset structure"),
            (3, "perspective_setup", "Setting up perspective and lighting"),
            (4, "reference_synthesis", "Synthesizing reference maps"),
            (5, "diffusion_core", "Running FLUX tile generation"),
            (6, "constraint_enforcement", "Enforcing tile constraints"),
            (7, "palette_harmonization", "Harmonizing palette and style"),
            (8, "post_processing", "Post-processing tiles"),
            (9, "atlas_generation", "Generating final atlas"),
            (10, "quality_validation", "Validating output quality"),
        ]
    
    def run(self, poll_interval: float = 2.0):
        """Main processing loop that polls for jobs and processes them."""
        logger.info("Starting job processor", jobs_dir=str(self.jobs_dir))
        
        while True:
            try:
                job_file = self._get_next_job()
                if job_file:
                    self._process_job(job_file)
                else:
                    time.sleep(poll_interval)
            except KeyboardInterrupt:
                logger.info("Job processor stopped by user")
                break
            except Exception as e:
                logger.error("Unexpected error in job processor", error=str(e))
                time.sleep(poll_interval)
    
    def _get_next_job(self) -> Optional[Path]:
        """Get the next job from the queue (FIFO order)."""
        try:
            job_files = list(self.queue_dir.glob("*.json"))
            if not job_files:
                return None
            
            # Sort by creation time (FIFO)
            job_files.sort(key=lambda f: f.stat().st_ctime)
            return job_files[0]
        except Exception as e:
            logger.error("Error getting next job", error=str(e))
            return None
    
    def _process_job(self, job_file: Path):
        """Process a single job through the complete pipeline."""
        job_id = job_file.stem
        logger.info("Processing job", job_id=job_id)
        
        try:
            # Load and extract configuration from job specification
            with open(job_file, 'r') as f:
                raw_job_spec = json.load(f)
            
            # Extract and validate configuration
            extracted_config = self._extract_configuration(raw_job_spec)
            
            # Create output directory
            output_path = self.output_dir / job_id
            output_path.mkdir(exist_ok=True)
            
            # Create pipeline context with extracted configuration
            context = PipelineContext(
                job_spec=raw_job_spec,
                output_path=output_path
            )
            
            # Add extracted configuration to context
            context.extracted_config = extracted_config
            
            # Update status to running
            self._update_status(job_id, "running", 0, "Starting tile generation pipeline")
            
            # Execute pipeline stages
            for stage_num, stage_name, stage_description in self.pipeline_stages:
                progress = int((stage_num / len(self.pipeline_stages)) * 90)  # Reserve 10% for final steps
                self._update_status(job_id, "running", progress, stage_description)
                
                # Execute stage
                stage_start_time = time.time()
                context.log_stage_start(stage_num, stage_name)
                
                stage_result = self._execute_stage(stage_num, stage_name, context)
                
                stage_duration = time.time() - stage_start_time
                context.log_stage_complete(stage_num, stage_name, stage_duration)
                
                if not stage_result["success"]:
                    raise Exception(f"Stage {stage_num:02d}_{stage_name} failed: {stage_result['error']}")
                
                logger.info("Stage completed", job_id=job_id, stage=f"{stage_num:02d}_{stage_name}")
            
            # Final output preparation
            self._update_status(job_id, "running", 95, "Finalizing output")
            final_output = self._finalize_output(context)
            
            # Mark as completed
            self._update_status(
                job_id, 
                "completed", 
                100, 
                "Tileset generation completed successfully",
                output_path=str(final_output)
            )
            
            # Remove job from queue
            job_file.unlink()
            
            logger.info("Job completed successfully", job_id=job_id, output=str(final_output))
            
        except Exception as e:
            error_msg = f"Job failed: {str(e)}"
            logger.error("Job processing failed", job_id=job_id, error=error_msg, traceback=traceback.format_exc())
            
            self._update_status(job_id, "failed", 0, error_msg, error=error_msg)
            
            # Remove job from queue even if failed
            if job_file.exists():
                job_file.unlink()
    
    def _extract_configuration(self, job_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and process configuration from job specification."""
        logger.info("Extracting configuration", job_id=job_spec.get("id", "unknown"))
        
        # Extract basic parameters
        config = {
            "job_id": job_spec.get("id"),
            "theme": job_spec.get("theme"),
            "palette": job_spec.get("palette"),
            "tile_size": job_spec.get("tileSize", 32),
            "sub_tile_size": job_spec.get("subTileSize", 8),
            "tile_count": job_spec.get("tileCount", 16),
        }
        
        # Extract tileset configuration
        tileset_config = job_spec.get("tilesetConfig", {})
        config["tileset"] = {
            "building_blocks": tileset_config.get("buildingBlocks", ["center"]),
            "variations_per_block": tileset_config.get("variationsPerBlock", 3),
            "composition_rules": tileset_config.get("compositionRules", "basic"),
        }
        
        # Extract model configuration
        model_config = job_spec.get("modelConfig", {})
        config["models"] = {
            "base_model": model_config.get("baseModel", "flux-dev"),
            "controlnet_model": model_config.get("controlnetModel"),
            "use_controlnet": model_config.get("useControlNet", True),
            "precision": model_config.get("precision", "bfloat16"),
            "enable_cpu_offload": model_config.get("enableCpuOffload", True),
        }
        
        # Extract generation parameters
        gen_params = job_spec.get("generationParams", {})
        config["generation"] = {
            "steps": gen_params.get("steps"),
            "guidance_scale": gen_params.get("guidanceScale"),
            "seed": gen_params.get("seed"),
            "batch_size": gen_params.get("batchSize", 1),
        }
        
        # Extract constraints
        constraints = job_spec.get("constraints", {})
        config["constraints"] = {
            "edge_similarity": constraints.get("edgeSimilarity", 0.98),
            "palette_deviation": constraints.get("paletteDeviation", 3.0),
            "structural_compliance": constraints.get("structuralCompliance", 1.0),
            "sub_tile_coherence": constraints.get("subTileCoherence", 0.7),
        }
        
        # Extract atlas configuration
        atlas_config = job_spec.get("atlasLayout", {})
        config["atlas"] = {
            "columns": atlas_config.get("columns"),
            "rows": atlas_config.get("rows"),
            "padding": atlas_config.get("padding", 1),
            "power_of_two": atlas_config.get("powerOfTwo", True),
            "max_size": atlas_config.get("maxSize", 2048),
        }
        
        # Extract options
        options = job_spec.get("options", {})
        config["options"] = {
            "watch": options.get("watch", False),
            "generate_normals": options.get("generateNormals", False),
            "generate_height_maps": options.get("generateHeightMaps", False),
            "enable_dithering": options.get("enableDithering", False),
            "auto_regenerate": options.get("autoRegenerate", True),
            "max_regeneration_attempts": options.get("maxRegenerationAttempts", 3),
            "output_formats": options.get("outputFormats", ["png", "json"]),
        }
        
        logger.info("Configuration extracted", 
                   theme=config["theme"], 
                   tile_count=config["tile_count"],
                   base_model=config["models"]["base_model"])
        
        return config
    
    def _execute_stage(self, stage_num: int, stage_name: str, context: PipelineContext) -> Dict[str, Any]:
        """Execute a specific pipeline stage."""
        try:
            # Import and execute the appropriate stage
            if stage_num == 1:
                from ..stages.stage_01_job_validation import execute
            elif stage_num == 2:
                from ..stages.stage_02_tileset_setup import execute
            elif stage_num == 3:
                from ..stages.stage_03_perspective_setup import execute
            elif stage_num == 4:
                from ..stages.stage_04_reference_synthesis import execute
            elif stage_num == 5:
                from ..stages.stage_05_diffusion_core import execute
            elif stage_num == 6:
                from ..stages.stage_06_constraint_enforcement import execute
            elif stage_num == 7:
                from ..stages.stage_07_palette_harmonization import execute
            elif stage_num == 8:
                from ..stages.stage_08_post_processing import execute
            elif stage_num == 9:
                from ..stages.stage_09_atlas_generation import execute
            elif stage_num == 10:
                from ..stages.stage_10_quality_validation import execute
            else:
                return {"success": False, "error": f"Unknown stage: {stage_num}"}
            
            # Execute the stage
            result = execute(context)
            return {"success": True, "data": result}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _finalize_output(self, context: PipelineContext) -> Path:
        """Finalize output and return path to main atlas file."""
        if context.final_atlas:
            atlas_path = context.output_path / "tileset.png"
            context.final_atlas.save(atlas_path)
            return atlas_path
        else:
            # Fallback - create placeholder
            atlas_path = context.output_path / "tileset.png"
            atlas_path.touch()
            return atlas_path
    
    def _update_status(self, job_id: str, status: str, progress: int, message: str, 
                      output_path: Optional[str] = None, error: Optional[str] = None):
        """Update job status file."""
        status_data = {
            "id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
        }
        
        if status == "running" and "startedAt" not in status_data:
            status_data["startedAt"] = datetime.now().isoformat()
        
        if status in ["completed", "failed", "cancelled"]:
            status_data["completedAt"] = datetime.now().isoformat()
        
        if output_path:
            status_data["outputPath"] = output_path
        
        if error:
            status_data["error"] = error
        
        status_file = self.status_dir / f"{job_id}.json"
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
