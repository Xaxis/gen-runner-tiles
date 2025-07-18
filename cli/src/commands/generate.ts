import chalk from 'chalk';
import { JobBuilder, GenerateOptions } from '../lib/job-builder';
import { FileQueue } from '../lib/file-queue';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

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
    console.log(chalk.gray(`Tile Size: ${jobSpec.tile_size}`));
    console.log(chalk.gray(`Tileset Type: ${jobSpec.tileset_type}`));
    console.log(chalk.gray(`View Angle: ${jobSpec.view_angle}`));
    console.log(chalk.gray(`Base Model: ${jobSpec.base_model}`));
    
    // Submit to file queue
    console.log(chalk.gray('Submitting job to queue...'));
    const jobsDir = path.resolve(__dirname, '../../../jobs');
    const fileQueue = new FileQueue(jobsDir);

    const jobId = await fileQueue.submitJob(jobSpec);
    console.log(chalk.green(`✅ Job submitted successfully!`));
    console.log(chalk.blue(`Job ID: ${jobId}`));

    // Wait for the job file to actually exist before starting worker
    console.log(chalk.gray('⏳ Waiting for job file to be created...'));
    await waitForJobFile(jobId);

    // Start worker to process the job
    console.log(chalk.blue('🚀 Starting worker to process job...'));
    await processJobWithWorker(jobId);
    
  } catch (error) {
    console.error(chalk.red('Error generating tileset:'), error);
    process.exit(1);
  }
}

async function waitForJobFile(jobId: string): Promise<void> {
  const jobFilePath = path.resolve(__dirname, '../../../jobs/queue', `${jobId}.json`);
  const maxWait = 10000; // 10 seconds max
  const checkInterval = 100; // Check every 100ms
  let waited = 0;

  while (waited < maxWait) {
    if (fs.existsSync(jobFilePath)) {
      console.log(chalk.green('✅ Job file created successfully'));
      return;
    }

    await new Promise(resolve => setTimeout(resolve, checkInterval));
    waited += checkInterval;
  }

  throw new Error(`Job file not created after ${maxWait}ms`);
}

async function processJobWithWorker(jobId: string): Promise<void> {
  return new Promise((resolve, reject) => {
    // Get the worker path relative to CLI
    const workerPath = path.resolve(__dirname, '../../../worker');
    const jobFilePath = path.resolve(__dirname, '../../../jobs/queue', `${jobId}.json`);

    console.log(chalk.gray(`Starting Python worker in: ${workerPath}`));
    console.log(chalk.gray(`Processing job file: ${jobFilePath}`));

    // Spawn the Python worker process with environment variables
    const worker = spawn('python', ['-m', 'src.main', '--job-file', jobFilePath], {
      cwd: workerPath,
      stdio: 'inherit', // Show worker output directly
      env: {
        ...process.env,
      }
    });

    worker.on('close', (code) => {
      if (code === 0) {
        console.log(chalk.green('🎉 Job completed successfully!'));
        resolve();
      } else {
        console.error(chalk.red(`❌ Worker failed with exit code ${code}`));
        reject(new Error(`Worker failed with exit code ${code}`));
      }
    });

    worker.on('error', (error) => {
      console.error(chalk.red('❌ Failed to start worker:'), error);
      reject(error);
    });
  });
}
