# Worker Architecture Proposal

## Current Structure

```
worker/src/
├── main.py                          # Entry point, starts job processor
├── core/
│   ├── pipeline_context.py          # Shared state between pipeline stages (DONE)
│   ├── job_processor.py             # Main orchestrator, executes pipeline (TODO)
│   └── model_registry.py            # FLUX models and themes (TODO)
└── stages/
    ├── 01_job_validation.py         # Job spec validation (TODO)
    ├── 02_tileset_setup.py          # Universal tileset structure - 13 building blocks (TODO)
    ├── 03_perspective_setup.py      # Global camera/lighting setup (TODO)
    ├── 04_reference_synthesis.py    # ControlNet conditioning images (TODO)
    ├── 05_diffusion_core.py         # FLUX tile generation (TODO)
    ├── 06_constraint_enforcement.py # Real-time validation (TODO)
    ├── 07_palette_harmonization.py  # Color quantization (TODO)
    ├── 08_post_processing.py        # Image cleanup (TODO)
    ├── 09_atlas_generation.py       # Final packing and metadata (TODO)
    └── 10_quality_validation.py     # Quality gates & auto-regen (TODO)
```

## Pipeline Context Purpose

**Central Data Hub**: All pipeline data flows through a single `PipelineContext` object
- **No File I/O Between Stages**: Eliminates intermediate file reads/writes
- **State Management**: Tracks progress, errors, and results from each stage
- **Data Contracts**: Defines exactly what each stage produces and consumes

## File Relationships

```
main.py
└── imports core/job_processor.py
    └── creates PipelineContext from job spec
    └── calls stages/01_job_validation.py(context)
    └── calls stages/02_tileset_generation.py(context)  # Contains 13 building blocks logic
    └── calls stages/03_perspective_setup.py(context)
    └── ... etc (stages 04-10)
    └── each stage modifies the SAME context object
```

## Data Flow Example

1. **main.py** → starts job processor
2. **job_processor.py** → creates PipelineContext from job spec
3. **Stage 01** → validates job, writes to `context.validated_job_spec`
4. **Stage 02** → generates 13 building blocks, writes to `context.universal_tileset`
5. **Stage 03** → sets up camera/lighting, writes to `context.global_camera_params`
6. **Stage 04** → creates reference maps, writes to `context.reference_maps`
7. **Stage 05** → generates tiles, writes to `context.generated_tiles`
8. **etc...**
