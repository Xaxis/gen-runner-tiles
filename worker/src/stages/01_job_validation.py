"""
Stage 01: Job Validation
Validates job specification, extracted configuration, and model availability.
Ensures all requirements are met before proceeding with tile generation.
"""

from typing import Dict, Any, List
import structlog
from pathlib import Path
from datetime import datetime

from ..core.pipeline_context import PipelineContext
from ..core.model_registry import ModelRegistry

logger = structlog.get_logger()

def execute(context: PipelineContext) -> Dict[str, Any]:
    """
    Execute Stage 01: Job Validation
    
    Validates:
    - Job specification format and required fields
    - Extracted configuration parameters
    - Model availability and compatibility
    - Theme and palette configurations
    - Tile size and count constraints
    - Output directory permissions
    
    Args:
        context: Pipeline context with job_spec and extracted_config
        
    Returns:
        Dict with validation results and any errors found
    """
    logger.info("Starting job validation", job_id=context.get_job_id())
    
    validation_errors = []
    validation_warnings = []
    
    try:
        # Validate context state
        if not context.validate_context_state(1):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Extract configuration from job specification
        logger.info("Extracting configuration from job specification")
        extracted_config = _extract_configuration_from_job_spec(context.job_spec)

        # Store extracted config in context for later stages
        context.extracted_config = extracted_config
        
        # 1. Validate basic job specification
        basic_validation = _validate_basic_job_spec(context.job_spec)
        validation_errors.extend(basic_validation["errors"])
        validation_warnings.extend(basic_validation["warnings"])
        
        # 2. Validate extracted configuration parameters
        config_validation = _validate_extracted_config(extracted_config)
        validation_errors.extend(config_validation["errors"])
        validation_warnings.extend(config_validation["warnings"])
        
        # 3. Validate model availability and compatibility
        model_validation = _validate_model_availability(extracted_config)
        validation_errors.extend(model_validation["errors"])
        validation_warnings.extend(model_validation["warnings"])
        
        # 4. Validate theme and palette configurations
        theme_validation = _validate_theme_palette_config(extracted_config)
        validation_errors.extend(theme_validation["errors"])
        validation_warnings.extend(theme_validation["warnings"])
        
        # 5. Validate tile parameters and constraints
        tile_validation = _validate_tile_parameters(extracted_config)
        validation_errors.extend(tile_validation["errors"])
        validation_warnings.extend(tile_validation["warnings"])
        
        # 6. Validate output directory and permissions
        output_validation = _validate_output_directory(context.output_path)
        validation_errors.extend(output_validation["errors"])
        validation_warnings.extend(output_validation["warnings"])
        
        # Store validation results in context
        context.validation_errors = validation_errors
        
        if validation_errors:
            logger.error("Job validation failed", 
                        job_id=context.get_job_id(),
                        error_count=len(validation_errors),
                        errors=validation_errors)
            return {"success": False, "errors": validation_errors, "warnings": validation_warnings}
        
        # Create validated job spec with normalized parameters
        validated_job_spec = _create_validated_job_spec(context.job_spec, extracted_config)
        context.validated_job_spec = validated_job_spec
        
        # Log warnings if any
        if validation_warnings:
            logger.warning("Job validation completed with warnings",
                          job_id=context.get_job_id(),
                          warning_count=len(validation_warnings),
                          warnings=validation_warnings)
        
        # Update context stage
        context.current_stage = 1

        logger.info("Job validation completed successfully",
                   job_id=context.get_job_id(),
                   theme=extracted_config["theme"],
                   tileset_type=extracted_config["tileset_type"])

        return {
            "success": True,
            "validated_job_spec": validated_job_spec,
            "warnings": validation_warnings,
            "validation_summary": {
                "theme": extracted_config["theme"],
                "palette": extracted_config["palette"],
                "tileset_type": extracted_config["tileset_type"],
                "tile_size": extracted_config["tile_size"],
                "base_model": extracted_config["base_model"],
            }
        }
        
    except Exception as e:
        error_msg = f"Job validation failed with exception: {str(e)}"
        logger.error("Job validation exception", job_id=context.get_job_id(), error=error_msg)
        context.add_error(1, error_msg)
        return {"success": False, "errors": [error_msg]}

def _validate_basic_job_spec(job_spec: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate basic job specification format and required fields."""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ["id", "theme", "palette", "tileSize", "tileset_type", "createdAt"]
    for field in required_fields:
        if field not in job_spec:
            errors.append(f"Missing required field: {field}")
    
    # Validate ID format
    if "id" in job_spec and not isinstance(job_spec["id"], str):
        errors.append("Job ID must be a string")
    
    # Validate timestamps
    if "createdAt" in job_spec:
        try:
            from datetime import datetime
            datetime.fromisoformat(job_spec["createdAt"].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            errors.append("Invalid createdAt timestamp format")
    
    return {"errors": errors, "warnings": warnings}

def _validate_extracted_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate extracted configuration parameters."""
    errors = []
    warnings = []
    
    # Validate tile size (sub_tile_size is now calculated by Stage 2)
    tile_size = config.get("tileSize", 0)  # Match JobSpec field name

    # Validate tile size is supported
    valid_tile_sizes = [32, 64, 128, 256, 512]
    if tile_size not in valid_tile_sizes:
        errors.append(f"Unsupported tile size: {tile_size}. Must be one of {valid_tile_sizes}")
    
    # Validate tileset type
    tileset_type = config.get("tileset_type", "")
    valid_tileset_types = ["minimal", "extended", "full"]
    if tileset_type not in valid_tileset_types:
        errors.append(f"Invalid tileset type '{tileset_type}'. Must be one of: {valid_tileset_types}")
    
    # Validate constraints
    constraints = config.get("constraints", {})
    edge_similarity = constraints.get("edge_similarity", 0)
    if not (0 <= edge_similarity <= 1):
        errors.append(f"Edge similarity {edge_similarity} must be between 0 and 1")
    
    palette_deviation = constraints.get("palette_deviation", 0)
    if not (0 <= palette_deviation <= 10):
        errors.append(f"Palette deviation {palette_deviation} must be between 0 and 10")
    
    return {"errors": errors, "warnings": warnings}

def _validate_model_availability(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate that required models are available."""
    errors = []
    warnings = []
    
    try:
        model_registry = ModelRegistry()
        
        # Validate base model
        base_model = config.get("models", {}).get("base_model")
        if not base_model:
            errors.append("No base model specified")
        elif not model_registry.get_model_config(base_model):
            errors.append(f"Base model '{base_model}' not found in registry")
        
        # Validate ControlNet model if enabled
        use_controlnet = config.get("models", {}).get("use_controlnet", False)
        if use_controlnet:
            controlnet_model = config.get("models", {}).get("controlnet_model")
            if not controlnet_model:
                warnings.append("ControlNet enabled but no model specified, will use default")
            elif not model_registry.get_model_config(controlnet_model):
                errors.append(f"ControlNet model '{controlnet_model}' not found in registry")
        
        # Validate theme configuration
        theme = config.get("theme")
        if theme and not model_registry.get_theme_config(theme):
            errors.append(f"Theme '{theme}' not found in registry")
        
        # Validate palette configuration
        palette = config.get("palette")
        if palette and not model_registry.get_palette_config(palette):
            errors.append(f"Palette '{palette}' not found in registry")
            
    except Exception as e:
        errors.append(f"Model validation failed: {str(e)}")
    
    return {"errors": errors, "warnings": warnings}

def _validate_theme_palette_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate theme and palette compatibility."""
    errors = []
    warnings = []
    
    try:
        model_registry = ModelRegistry()
        
        theme_name = config.get("theme")
        palette_name = config.get("palette")
        
        if theme_name and palette_name:
            theme_config = model_registry.get_theme_config(theme_name)
            palette_config = model_registry.get_palette_config(palette_name)
            
            if theme_config and palette_config:
                # Check if theme and palette are compatible
                if theme_name == "pixel" and palette_config.dithering_enabled:
                    warnings.append("Dithering enabled for pixel art theme - may affect sharp edges")
                
                if theme_name == "sci-fi" and palette_name == "earth":
                    warnings.append("Earth palette with sci-fi theme may not match aesthetic")
                
                # Validate generation parameters
                base_model = config.get("models", {}).get("base_model")
                if base_model != theme_config.base_model:
                    warnings.append(f"Using {base_model} instead of recommended {theme_config.base_model} for {theme_name} theme")
                    
    except Exception as e:
        errors.append(f"Theme/palette validation failed: {str(e)}")
    
    return {"errors": errors, "warnings": warnings}

def _validate_tile_parameters(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate tile parameters and building block configuration."""
    errors = []
    warnings = []
    
    # Tileset configuration is now handled by tileset_type
    # No need to validate building blocks since they're determined by tessellation requirements
    
    # Validate atlas configuration
    atlas_config = config.get("atlas", {})
    max_size = atlas_config.get("max_size", 2048)
    if max_size not in [256, 512, 1024, 2048, 4096, 8192]:
        warnings.append(f"Atlas max size {max_size} is not a standard power-of-two size")
    
    return {"errors": errors, "warnings": warnings}

def _validate_output_directory(output_path: Path) -> Dict[str, List[str]]:
    """Validate output directory exists and is writable."""
    errors = []
    warnings = []
    
    try:
        # Check if directory exists
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {str(e)}")
                return {"errors": errors, "warnings": warnings}
        
        # Check if directory is writable
        test_file = output_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            errors.append(f"Output directory is not writable: {str(e)}")
        
        # Check available space (warn if less than 1GB)
        try:
            import shutil
            free_space = shutil.disk_usage(output_path).free
            if free_space < 1024 * 1024 * 1024:  # 1GB
                warnings.append(f"Low disk space: {free_space // (1024*1024)} MB available")
        except Exception:
            warnings.append("Could not check available disk space")
            
    except Exception as e:
        errors.append(f"Output directory validation failed: {str(e)}")
    
    return {"errors": errors, "warnings": warnings}

def _create_validated_job_spec(job_spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a validated and normalized job specification."""
    validated_spec = job_spec.copy()
    
    # Normalize and add computed fields
    validated_spec["normalized_config"] = config
    validated_spec["validation_timestamp"] = str(datetime.now().isoformat())
    validated_spec["pipeline_version"] = "1.0.0"
    
    # Add computed atlas layout if not specified
    if "atlasLayout" not in validated_spec or not validated_spec["atlasLayout"].get("columns"):
        tileset_type = config["tileset_type"]
        tessellation_counts = {
            "minimal": 13,
            "extended": 47,
            "full": 256
        }
        tile_count = tessellation_counts.get(tileset_type, 13)
        columns = int(tile_count ** 0.5) + (1 if tile_count ** 0.5 != int(tile_count ** 0.5) else 0)
        rows = (tile_count + columns - 1) // columns
        
        validated_spec["atlasLayout"] = {
            "columns": columns,
            "rows": rows,
            "padding": config.get("atlas", {}).get("padding", 1),
            "powerOfTwo": config.get("atlas", {}).get("power_of_two", True),
            "maxSize": config.get("atlas", {}).get("max_size", 2048),
        }
    
    return validated_spec

def _extract_configuration_from_job_spec(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize configuration from job specification."""

    # Extract basic configuration (simplified for refactored pipeline)
    extracted_config = {
        "theme": job_spec.get("theme", "fantasy"),
        "palette": job_spec.get("palette", "default"),
        "tileSize": job_spec.get("tileSize", 64),  # Match JobSpec field name
        "tileset_type": job_spec.get("tileset_type", "minimal"),
        "viewAngle": job_spec.get("viewAngle", "top-down"),
        "baseModel": job_spec.get("baseModel", "flux-dev"),
        # sub_tile_size is now calculated by Stage 2

    }

    return extracted_config
