import { JobSpec, JobSpecSchema } from '../types/job-spec';
import { v4 as uuidv4 } from 'uuid';

export interface GenerateOptions {
  theme: string;
  palette: string;
  size: string;
  tileset: string;
  baseModel?: string;
}

export class JobBuilder {
  static build(options: GenerateOptions): JobSpec {
    const jobSpec: JobSpec = {
      id: uuidv4(),
      theme: options.theme,
      palette: options.palette,
      tileSize: parseInt(options.size, 10),
      tileset_type: options.tileset as 'minimal' | 'extended' | 'full',
      viewAngle: 'top-down',
      baseModel: (options.baseModel as 'flux-dev' | 'flux-schnell') || 'flux-dev',
      createdAt: new Date().toISOString(),
    };

    // Validate the job spec
    const result = JobSpecSchema.safeParse(jobSpec);
    if (!result.success) {
      throw new Error(`Invalid job specification: ${result.error.message}`);
    }

    return result.data;
  }
}
