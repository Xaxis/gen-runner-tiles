# Stage 5: FLUX Multi-Tile Diffusion Core - EXPERT IMPLEMENTATION PLAN

## 🎯 GOAL
Create a **beyond expert** Stage 5 that generates **usable retro 2D game asset tiles** with **perfect tessellation** using our **research-backed multi-tile strategy**.

---

## 📋 REQUIREMENTS ANALYSIS

### **MUST USE FROM CONTEXT:**
1. **`context.reference_maps`** - Structure-aware reference images from Stage 4
2. **`context.control_images`** - ControlNet conditioning images from Stage 4
3. **`context.shared_edges`** - Tessellation edge constraints from Stage 2
4. **`context.universal_tileset`** - Wang tile specifications from Stage 2
5. **`context.extracted_config`** - User configuration (tileSize, theme, palette)

### **MUST IMPLEMENT:**
1. **Atlas-based generation** - Generate all 13 tiles simultaneously in one atlas
2. **ControlNet integration** - Use reference images for structure control
3. **Edge coordination** - Ensure seamless tessellation between connecting tiles
4. **UmeAiRT LoRA** - Apply `UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art` with trigger word
5. **Research-backed tessellation** - Implement proper Wang tile methodology

---

## 🏗️ DETAILED IMPLEMENTATION PLAN

### **PHASE 1: CONTEXT DATA INTEGRATION**

#### **1.1 Extract Required Data**
```python
# From Stage 4 (Reference Synthesis)
reference_maps = context.reference_maps  # Dict[int, Dict[str, Image.Image]]
control_images = context.control_images  # Dict[int, Image.Image]

# From Stage 2 (Tessellation Setup)
shared_edges = context.shared_edges      # List[SharedEdge]
tileset_setup = context.universal_tileset

# From Stage 1 (Job Validation)
config = context.extracted_config
tile_size = config.get("tileSize", 32)   # USER'S ACTUAL TILE SIZE
theme = config.get("theme", "fantasy")
palette = config.get("palette", "medieval")
```

#### **1.2 Validate Required Data**
```python
# Ensure all required data exists
if not reference_maps:
    raise ValueError("Missing reference_maps from Stage 4")
if not control_images:
    raise ValueError("Missing control_images from Stage 4")
if not shared_edges:
    raise ValueError("Missing shared_edges from Stage 2")
```

### **PHASE 2: ATLAS GENERATION SETUP**

#### **2.1 Calculate Atlas Dimensions**
```python
# Atlas layout from tileset_setup
atlas_columns = tileset_setup.atlas_columns  # 4
atlas_rows = tileset_setup.atlas_rows        # 4
atlas_width = atlas_columns * tile_size      # 4 * user_tile_size
atlas_height = atlas_rows * tile_size        # 4 * user_tile_size
```

#### **2.2 Create Atlas Reference Image**
```python
# Combine all reference images into single atlas
atlas_reference = Image.new("RGB", (atlas_width, atlas_height))
for tile_id, ref_images in reference_maps.items():
    # Position tile in atlas grid
    row = tile_id // atlas_columns
    col = tile_id % atlas_columns
    x = col * tile_size
    y = row * tile_size
    
    # Use structure reference image
    structure_ref = ref_images["structure"]
    atlas_reference.paste(structure_ref, (x, y))
```

#### **2.3 Create Atlas Control Image**
```python
# Combine all control images into single atlas
atlas_control = Image.new("RGB", (atlas_width, atlas_height))
for tile_id, control_img in control_images.items():
    # Position tile in atlas grid
    row = tile_id // atlas_columns
    col = tile_id % atlas_columns
    x = col * tile_size
    y = row * tile_size
    
    atlas_control.paste(control_img, (x, y))
```

### **PHASE 3: FLUX PIPELINE SETUP**

#### **3.1 Load FLUX with ControlNet**
```python
# Load base FLUX pipeline
pipeline = model_registry.load_base_model("flux-dev")

# Load ControlNet for structure control
controlnet = load_controlnet_for_flux()  # Research-backed ControlNet

# Combine pipeline with ControlNet
pipeline = FluxControlNetPipeline(
    transformer=pipeline.transformer,
    controlnet=controlnet,
    scheduler=pipeline.scheduler,
    vae=pipeline.vae,
    text_encoder=pipeline.text_encoder,
    tokenizer=pipeline.tokenizer
)
```

#### **3.2 Apply UmeAiRT LoRA**
```python
# Apply specialized pixel art LoRA
pipeline.load_lora_weights("UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art")
pipeline.set_adapters(["retro_pixel"], adapter_weights=[0.7])
```

### **PHASE 4: PROMPT GENERATION**

#### **4.1 Create Atlas-Wide Prompt**
```python
# Generate comprehensive prompt for entire atlas
atlas_prompt = f"umempart, pixel art tileset, {theme} {palette} style, "
atlas_prompt += "seamless tessellating tiles, retro game assets, "
atlas_prompt += "crisp pixels, no blur, sharp edges, detailed pixel work"

# Add negative prompt
negative_prompt = "blurry, smooth, anti-aliased, photorealistic, 3d render"
```

#### **4.2 Tile-Specific Prompt Regions**
```python
# Create prompt map for different tile regions
prompt_regions = {}
for tile_id, tile_spec in tileset_setup.tile_specs.items():
    structure_prompt = get_structure_prompt(tile_spec.tile_type)
    prompt_regions[tile_id] = f"umempart, {structure_prompt}"
```

### **PHASE 5: COORDINATED GENERATION**

#### **5.1 Atlas Generation with ControlNet**
```python
# Generate entire atlas with structure control
result = pipeline(
    prompt=atlas_prompt,
    negative_prompt=negative_prompt,
    image=atlas_control,           # ControlNet conditioning
    control_image=atlas_reference, # Additional reference
    height=atlas_height,
    width=atlas_width,
    num_inference_steps=20,        # Optimized for UmeAiRT LoRA
    guidance_scale=3.5,            # Optimal for FLUX
    controlnet_conditioning_scale=0.8,  # Strong structure control
    generator=torch.Generator(device="cpu").manual_seed(seed)  # FLUX requirement
)
```

#### **5.2 Edge Coordination During Generation**
```python
# Apply edge blending every few steps
class EdgeCoordinationCallback:
    def __call__(self, step, timestep, latents):
        if step % 3 == 0:  # Every 3 steps
            latents = apply_edge_coordination(latents, shared_edges, tile_size)
        return latents

# Use callback during generation
result = pipeline(..., callback=EdgeCoordinationCallback())
```

### **PHASE 6: EDGE COORDINATION IMPLEMENTATION**

#### **6.1 Shared Edge Processing**
```python
def apply_edge_coordination(latents, shared_edges, tile_size):
    """Apply edge blending between connecting tiles."""
    for shared_edge in shared_edges:
        tile_a_id = shared_edge.tile_a_id
        tile_b_id = shared_edge.tile_b_id
        direction_a = shared_edge.tile_a_edge
        direction_b = shared_edge.tile_b_edge
        
        # Get tile positions in atlas
        pos_a = get_tile_position_in_atlas(tile_a_id, tile_size)
        pos_b = get_tile_position_in_atlas(tile_b_id, tile_size)
        
        # Extract edge regions from latents
        edge_a = extract_edge_region(latents, pos_a, direction_a)
        edge_b = extract_edge_region(latents, pos_b, direction_b)
        
        # Blend edges for seamless transition
        blended_edge = (edge_a + edge_b) / 2
        
        # Apply blended edges back to latents
        apply_edge_to_latents(latents, pos_a, direction_a, blended_edge)
        apply_edge_to_latents(latents, pos_b, direction_b, blended_edge)
    
    return latents
```

### **PHASE 7: TILE EXTRACTION**

#### **7.1 Extract Individual Tiles**
```python
def extract_tiles_from_atlas(atlas_image, tileset_setup, tile_size):
    """Extract individual tiles from generated atlas."""
    tiles = {}
    
    for tile_id in range(len(tileset_setup.tile_specs)):
        # Calculate tile position
        row = tile_id // atlas_columns
        col = tile_id % atlas_columns
        
        # Extract tile region
        x1 = col * tile_size
        y1 = row * tile_size
        x2 = x1 + tile_size
        y2 = y1 + tile_size
        
        tile_image = atlas_image.crop((x1, y1, x2, y2))
        tiles[tile_id] = tile_image
    
    return tiles
```

### **PHASE 8: VALIDATION & QUALITY ASSURANCE**

#### **8.1 Tessellation Validation**
```python
# Validate edge matching between connecting tiles
for shared_edge in shared_edges:
    tile_a = tiles[shared_edge.tile_a_id]
    tile_b = tiles[shared_edge.tile_b_id]
    
    edge_similarity = calculate_edge_similarity(
        tile_a, tile_b, 
        shared_edge.tile_a_edge, shared_edge.tile_b_edge
    )
    
    if edge_similarity < 0.95:  # 95% similarity threshold
        logger.warning(f"Poor edge matching: {edge_similarity}")
```

---

## 🔬 RESEARCH-BACKED FEATURES

### **1. Wang Tile Methodology**
- ✅ **13-tile system** (4 corners, 4 edges, 4 t-junctions, 1 cross)
- ✅ **Proper connectivity patterns** for infinite tessellation
- ✅ **Edge constraint enforcement** during generation

### **2. ControlNet Integration**
- ✅ **Structure-aware generation** using reference images
- ✅ **Depth and normal map conditioning** for 3D consistency
- ✅ **Edge constraint visualization** (red=seamless, blue=free)

### **3. Multi-Tile Coordination**
- ✅ **Atlas-based generation** for tile consistency
- ✅ **Cross-tile attention** through shared latent space
- ✅ **Edge blending** during diffusion process

### **4. Pixel Art Optimization**
- ✅ **UmeAiRT LoRA** with "umempart" trigger word
- ✅ **Optimal parameters** (20 steps, 3.5 guidance, 0.7 LoRA weight)
- ✅ **Pixel-perfect constraints** through prompting

---

## ✅ SUCCESS CRITERIA

1. **Perfect Tessellation** - All connecting edges match seamlessly
2. **Consistent Style** - All tiles maintain retro pixel art aesthetic
3. **Structure Adherence** - Tiles match their Wang tile specifications
4. **Game-Ready Quality** - Tiles are immediately usable in game engines
5. **Performance** - Generation completes in reasonable time (~15 minutes)

---

## 🚨 CRITICAL IMPLEMENTATION NOTES

1. **MUST use context data** - No hardcoded values or ignored context
2. **MUST implement atlas generation** - No individual tile generation
3. **MUST use ControlNet** - No text-only generation
4. **MUST apply edge coordination** - No independent tile generation
5. **MUST validate tessellation** - No assumption of success

---

*This plan represents a comprehensive, research-backed approach to implementing Stage 5 that actually achieves our revolutionary goal of generating usable retro 2D game asset tiles.*
