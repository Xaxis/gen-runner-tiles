import chalk from 'chalk';
import { JobBuilder, GenerateOptions } from '../lib/job-builder';
import { FileQueue } from '../lib/file-queue';

export async function generateCommand(options: GenerateOptions): Promise<void> {
  try {
    console.log(chalk.blue('🎨 Generating tileset...'));
    
    // Validate required options
    if (!options.theme) {
      console.error(chalk.red('Error: Theme is required. Use --theme <theme>'));
      process.exit(1);
    }
    
    if (!options.palette) {
      console.error(chalk.red('Error: Palette is required. Use --palette <palette>'));
      process.exit(1);
    }

    // Build job specification
    console.log(chalk.gray('Building job specification...'));
    const jobSpec = JobBuilder.build(options);
    
    console.log(chalk.gray(`Job ID: ${jobSpec.id}`));
    console.log(chalk.gray(`Theme: ${jobSpec.theme}`));
    console.log(chalk.gray(`Palette: ${jobSpec.palette}`));
    console.log(chalk.gray(`Tile Size: ${jobSpec.tileSize}px`));
    console.log(chalk.gray(`Tile Count: ${jobSpec.tileCount}`));
    
    // Submit to file queue
    console.log(chalk.gray('Submitting job to queue...'));
    const fileQueue = new FileQueue();
    
    const jobId = await fileQueue.submitJob(jobSpec);
    console.log(chalk.green(`✅ Job submitted successfully!`));
    console.log(chalk.blue(`Job ID: ${jobId}`));
    
    if (options.watch) {
      console.log(chalk.yellow('👀 Watching job progress...'));
      await watchJobProgress(fileQueue, jobId);
    } else {
      console.log(chalk.gray(`Use 'gen-tiles status --job ${jobId}' to check progress`));
      console.log(chalk.gray(`Start the worker with 'pnpm start:worker' to process jobs`));
    }
    
  } catch (error) {
    console.error(chalk.red('Error generating tileset:'), error);
    process.exit(1);
  }
}

async function watchJobProgress(fileQueue: FileQueue, jobId: string): Promise<void> {
  const pollInterval = 2000; // 2 seconds
  
  while (true) {
    const status = await fileQueue.getJobStatus(jobId);
    
    if (!status) {
      console.error(chalk.red('Job not found'));
      break;
    }
    
    console.log(chalk.blue(`Status: ${status.status} | Progress: ${status.progress}%`));
    
    if (status.message) {
      console.log(chalk.gray(`Message: ${status.message}`));
    }
    
    if (status.status === 'completed') {
      console.log(chalk.green('🎉 Tileset generation completed!'));
      if (status.outputPath) {
        console.log(chalk.blue(`Output: ${status.outputPath}`));
      }
      break;
    }
    
    if (status.status === 'failed') {
      console.error(chalk.red('❌ Job failed'));
      if (status.error) {
        console.error(chalk.red(`Error: ${status.error}`));
      }
      break;
    }
    
    if (status.status === 'cancelled') {
      console.log(chalk.yellow('⏹️ Job was cancelled'));
      break;
    }
    
    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }
}
