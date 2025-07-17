import chalk from 'chalk';
import { FileQueue } from '../lib/file-queue';

interface StatusOptions {
  job?: string;
}

export async function statusCommand(options: StatusOptions): Promise<void> {
  try {
    const fileQueue = new FileQueue();
    
    if (options.job) {
      // Show specific job status
      const status = await fileQueue.getJobStatus(options.job);
      
      if (!status) {
        console.error(chalk.red(`Job ${options.job} not found`));
        process.exit(1);
      }
      
      console.log(chalk.blue(`Job ${status.id}:`));
      console.log(`  Status: ${getStatusColor(status.status)}`);
      console.log(`  Progress: ${status.progress}%`);
      
      if (status.message) {
        console.log(`  Message: ${chalk.gray(status.message)}`);
      }
      
      if (status.startedAt) {
        console.log(`  Started: ${chalk.gray(new Date(status.startedAt).toLocaleString())}`);
      }
      
      if (status.completedAt) {
        console.log(`  Completed: ${chalk.gray(new Date(status.completedAt).toLocaleString())}`);
      }
      
      if (status.outputPath) {
        console.log(`  Output: ${chalk.blue(status.outputPath)}`);
      }
      
      if (status.error) {
        console.log(`  Error: ${chalk.red(status.error)}`);
      }
      
    } else {
      // Show all jobs
      const jobs = await fileQueue.getAllJobs();
      
      if (jobs.length === 0) {
        console.log(chalk.gray('No jobs found'));
        return;
      }
      
      console.log(chalk.blue('Recent Jobs:'));
      console.log('');
      
      for (const job of jobs.slice(0, 10)) {
        const timeStr = job.startedAt 
          ? new Date(job.startedAt).toLocaleString()
          : 'Not started';
          
        console.log(`${job.id.slice(0, 8)} | ${getStatusColor(job.status)} | ${job.progress}% | ${timeStr}`);
      }
      
      if (jobs.length > 10) {
        console.log(chalk.gray(`... and ${jobs.length - 10} more jobs`));
      }
    }
    
  } catch (error) {
    console.error(chalk.red('Error checking status:'), error);
    process.exit(1);
  }
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'queued':
      return chalk.yellow(status);
    case 'running':
      return chalk.blue(status);
    case 'completed':
      return chalk.green(status);
    case 'failed':
      return chalk.red(status);
    case 'cancelled':
      return chalk.gray(status);
    default:
      return status;
  }
}
