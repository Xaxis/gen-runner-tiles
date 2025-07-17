import fs from 'fs/promises';
import path from 'path';
import { JobSpec, JobStatus } from '../types/job-spec';

export class FileQueue {
  private jobsDir: string;
  private queueDir: string;
  private statusDir: string;
  private outputDir: string;

  constructor(baseDir = './jobs') {
    this.jobsDir = baseDir;
    this.queueDir = path.join(baseDir, 'queue');
    this.statusDir = path.join(baseDir, 'status');
    this.outputDir = path.join(baseDir, 'output');
  }

  async ensureDirectories(): Promise<void> {
    await fs.mkdir(this.queueDir, { recursive: true });
    await fs.mkdir(this.statusDir, { recursive: true });
    await fs.mkdir(this.outputDir, { recursive: true });
  }

  async submitJob(jobSpec: JobSpec): Promise<string> {
    await this.ensureDirectories();
    
    // Write job spec to queue
    const jobFile = path.join(this.queueDir, `${jobSpec.id}.json`);
    await fs.writeFile(jobFile, JSON.stringify(jobSpec, null, 2));
    
    // Create initial status
    const jobStatus: JobStatus = {
      id: jobSpec.id,
      status: 'queued',
      progress: 0,
    };
    
    await this.updateJobStatus(jobStatus);
    
    return jobSpec.id;
  }

  async getJobStatus(jobId: string): Promise<JobStatus | null> {
    try {
      const statusFile = path.join(this.statusDir, `${jobId}.json`);
      const statusData = await fs.readFile(statusFile, 'utf-8');
      return JSON.parse(statusData) as JobStatus;
    } catch (error) {
      return null;
    }
  }

  async updateJobStatus(status: JobStatus): Promise<void> {
    await this.ensureDirectories();
    const statusFile = path.join(this.statusDir, `${status.id}.json`);
    await fs.writeFile(statusFile, JSON.stringify(status, null, 2));
  }

  async getAllJobs(): Promise<JobStatus[]> {
    try {
      await this.ensureDirectories();
      const files = await fs.readdir(this.statusDir);
      const statuses: JobStatus[] = [];
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          const statusData = await fs.readFile(path.join(this.statusDir, file), 'utf-8');
          statuses.push(JSON.parse(statusData) as JobStatus);
        }
      }
      
      return statuses.sort((a, b) => 
        new Date(b.startedAt || '').getTime() - new Date(a.startedAt || '').getTime()
      );
    } catch (error) {
      return [];
    }
  }

  async cancelJob(jobId: string): Promise<boolean> {
    const status = await this.getJobStatus(jobId);
    if (!status || status.status === 'completed' || status.status === 'failed') {
      return false;
    }

    // Remove from queue if still queued
    if (status.status === 'queued') {
      try {
        const queueFile = path.join(this.queueDir, `${jobId}.json`);
        await fs.unlink(queueFile);
      } catch (error) {
        // File might already be picked up by worker
      }
    }

    // Update status to cancelled
    const updatedStatus: JobStatus = {
      ...status,
      status: 'cancelled',
      completedAt: new Date().toISOString(),
    };

    await this.updateJobStatus(updatedStatus);
    return true;
  }

  getOutputPath(jobId: string): string {
    return path.join(this.outputDir, jobId);
  }
}
