import { z } from 'zod';

export const JobSpecSchema = z.object({
  id: z.string(),

  // Style configuration
  theme: z.string(),

  // Color palette configuration
  palette: z.string(),

  // Tessellation configuration
  tile_size: z.number().min(32).max(512).refine(val => [32, 64, 128, 256, 512].includes(val), {
    message: "tileSize must be one of: 32, 64, 128, 256, 512"
  }),

  // Tessellation type
  tileset_type: z.enum(['minimal', 'extended', 'full']).default('minimal'),

  // View configuration
  view_angle: z.enum(['top-down', 'isometric', 'side-view']).default('top-down'),

  // Model configuration (simplified)
  base_model: z.enum(['flux-dev', 'flux-schnell']).default('flux-dev'),

  created_at: z.string().datetime(),
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
