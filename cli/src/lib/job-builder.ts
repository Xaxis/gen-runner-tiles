import { JobSpec, JobSpecSchema } from '../types/job-spec';
import { v4 as uuidv4 } from 'uuid';

export interface GenerateOptions {
  theme: string;
  palette: string;
  size: string;
  count: string;
  watch?: boolean;
  subTileSize?: string;
  baseModel?: string;
  useControlNet?: boolean;
  steps?: string;
  guidanceScale?: string;
  seed?: string;
}

export class JobBuilder {
  static build(options: GenerateOptions): JobSpec {
    const tileSize = parseInt(options.size, 10);
    const subTileSize = parseInt(options.subTileSize || '8', 10);
    const tileCount = parseInt(options.count, 10);

    // Calculate optimal atlas layout
    const columns = Math.ceil(Math.sqrt(tileCount));
    const rows = Math.ceil(tileCount / columns);

    const jobSpec: JobSpec = {
      id: uuidv4(),
      theme: options.theme,
      palette: options.palette,
      tileSize,
      subTileSize,
      tileCount,

      // Universal tileset configuration
      tilesetConfig: {
        buildingBlocks: ['center', 'border_top', 'border_right', 'border_bottom', 'border_left'],
        variationsPerBlock: 3,
        compositionRules: 'basic',
      },

      // Model configuration
      modelConfig: {
        baseModel: (options.baseModel as 'flux-dev' | 'flux-schnell') || 'flux-dev',
        controlnetModel: 'flux-controlnet-union',
        useControlNet: options.useControlNet !== false,
        precision: 'bfloat16',
        enableCpuOffload: true,
      },

      // Generation parameters
      generationParams: {
        steps: options.steps ? parseInt(options.steps, 10) : undefined,
        guidanceScale: options.guidanceScale ? parseFloat(options.guidanceScale) : undefined,
        seed: options.seed ? parseInt(options.seed, 10) : undefined,
        batchSize: 1,
      },

      resolution: {
        width: tileSize,
        height: tileSize,
      },

      viewAngle: 'top-down',

      atlasLayout: {
        columns,
        rows,
        padding: 1,
        powerOfTwo: true,
        maxSize: 2048,
      },

      constraints: {
        edgeSimilarity: 0.98,
        paletteDeviation: 3,
        structuralCompliance: 1.0,
        subTileCoherence: 0.7,
      },

      options: {
        watch: options.watch || false,
        generateNormals: false,
        generateHeightMaps: false,
        enableDithering: false,
        autoRegenerate: true,
        maxRegenerationAttempts: 3,
        outputFormats: ['png', 'json'],
      },

      createdAt: new Date().toISOString(),
    };

    // Validate the job spec
    const result = JobSpecSchema.safeParse(jobSpec);
    if (!result.success) {
      throw new Error(`Invalid job specification: ${result.error.message}`);
    }

    return result.data;
  }

  static validate(jobSpec: unknown): JobSpec {
    const result = JobSpecSchema.safeParse(jobSpec);
    if (!result.success) {
      throw new Error(`Invalid job specification: ${result.error.message}`);
    }
    return result.data;
  }
}
