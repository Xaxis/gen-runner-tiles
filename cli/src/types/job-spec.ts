import { z } from 'zod';

export const JobSpecSchema = z.object({
  id: z.string(),
  theme: z.string(),
  palette: z.string(),

  // Multi-tile size support
  tileSize: z.number().min(16).max(512), // Support larger tiles
  subTileSize: z.number().min(4).max(64).default(8), // 8x8 sub-tiles by default
  tileset_type: z.enum(['minimal', 'extended', 'full']).default('minimal'), // Tessellation type

  // Universal tileset configuration
  tilesetConfig: z.object({
    buildingBlocks: z.array(z.enum([
      'center', 'border_top', 'border_right', 'border_bottom', 'border_left',
      'edge_ne', 'edge_nw', 'edge_se', 'edge_sw',
      'corner_ne', 'corner_nw', 'corner_se', 'corner_sw'
    ])).default(['center', 'border_top', 'border_right', 'border_bottom', 'border_left']),
    variationsPerBlock: z.number().min(1).max(20).default(3),
    compositionRules: z.enum(['basic', 'advanced', 'custom']).default('basic'),
  }),

  // Model configuration
  modelConfig: z.object({
    baseModel: z.enum(['flux-dev', 'flux-schnell']).default('flux-dev'),
    controlnetModel: z.enum(['flux-controlnet-union', 'flux-controlnet-depth', 'flux-controlnet-canny']).optional(),
    useControlNet: z.boolean().default(true),
    precision: z.enum(['float16', 'bfloat16', 'float32']).default('bfloat16'),
    enableCpuOffload: z.boolean().default(true),
  }),

  // Generation parameters
  generationParams: z.object({
    steps: z.number().min(1).max(100).optional(), // Will use theme default if not specified
    guidanceScale: z.number().min(0.1).max(20).optional(), // Will use theme default if not specified
    seed: z.number().optional(),
    batchSize: z.number().min(1).max(16).default(1),
  }),

  resolution: z.object({
    width: z.number(),
    height: z.number(),
  }),
  viewAngle: z.enum(['top-down', 'isometric', 'side-view']).default('top-down'),

  // Atlas configuration
  atlasLayout: z.object({
    columns: z.number(),
    rows: z.number(),
    padding: z.number().default(1),
    powerOfTwo: z.boolean().default(true),
    maxSize: z.number().default(2048),
  }),

  // Enhanced constraints
  constraints: z.object({
    edgeSimilarity: z.number().min(0).max(1).default(0.98),
    paletteDeviation: z.number().min(0).max(10).default(3),
    structuralCompliance: z.number().min(0).max(1).default(1.0),
    subTileCoherence: z.number().min(0).max(1).default(0.7),
  }),

  // Enhanced options
  options: z.object({
    watch: z.boolean().default(false),
    generateNormals: z.boolean().default(false),
    generateHeightMaps: z.boolean().default(false),
    enableDithering: z.boolean().default(false),
    autoRegenerate: z.boolean().default(true),
    maxRegenerationAttempts: z.number().min(1).max(10).default(3),
    outputFormats: z.array(z.enum(['png', 'json', 'tmx', 'unity', 'godot'])).default(['png', 'json']),
  }),

  createdAt: z.string().datetime(),
});

export type JobSpec = z.infer<typeof JobSpecSchema>;

export const JobStatusSchema = z.object({
  id: z.string(),
  status: z.enum(['queued', 'running', 'completed', 'failed', 'cancelled']),
  progress: z.number().min(0).max(100),
  message: z.string().optional(),
  startedAt: z.string().datetime().optional(),
  completedAt: z.string().datetime().optional(),
  outputPath: z.string().optional(),
  error: z.string().optional(),
});

export type JobStatus = z.infer<typeof JobStatusSchema>;
