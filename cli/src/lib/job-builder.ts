import { JobSpec, JobSpecSchema } from '../types/job-spec';
import { v4 as uuidv4 } from 'uuid';

export interface GenerateOptions {
  theme: string;
  palette: string;
  tileSize: string;
  tilesetType: string;
  baseModel?: string;
}

export class JobBuilder {
  static build(options: GenerateOptions): JobSpec {
    const jobSpec: JobSpec = {
      id: uuidv4(),
      theme: options.theme,
      palette: options.palette,
      tile_size: parseInt(options.tileSize),
      tileset_type: options.tilesetType as 'minimal' | 'extended' | 'full',
      view_angle: 'top-down',
      base_model: (options.baseModel as 'flux-dev' | 'flux-schnell') || 'flux-dev',
      created_at: new Date().toISOString(),
    };

    // Validate the job spec
    const result = JobSpecSchema.safeParse(jobSpec);
    if (!result.success) {
      throw new Error(`Invalid job specification: ${result.error.message}`);
    }

    return result.data;
  }
}
