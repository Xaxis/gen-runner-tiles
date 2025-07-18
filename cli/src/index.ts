#!/usr/bin/env node

import { Command } from 'commander';
import chalk from 'chalk';
import { generateCommand } from './commands/generate';
import { statusCommand } from './commands/status';
import { configCommand } from './commands/config';

const program = new Command();

program
  .name('gen-tiles')
  .description('AI-powered tileable game asset generator')
  .version('0.1.0');

program
  .command('generate')
  .alias('gen')
  .description('Generate a new tileset')
  .option('-t, --theme <theme>', 'Theme for the tileset (e.g., fantasy, sci-fi)')
  .option('-p, --palette <palette>', 'Color palette name')
  .option('-s, --size <size>', 'Tile size in pixels', '32')
  .option('--tileset <type>', 'Tileset type (minimal, extended, full)', 'minimal')
  .option('--base-model <model>', 'Base model to use (flux-dev, flux-schnell)', 'flux-dev')
  .option('--watch', 'Watch generation progress in real-time')
  .action(generateCommand);

program
  .command('status')
  .description('Check status of running jobs')
  .option('-j, --job <jobId>', 'Check specific job ID')
  .action(statusCommand);

program
  .command('config')
  .description('Manage configuration')
  .option('--list', 'List available themes and palettes')
  .option('--set <key=value>', 'Set configuration value')
  .action(configCommand);

program.parse();

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error(chalk.red('Unhandled Rejection at:'), promise, chalk.red('reason:'), reason);
  process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error(chalk.red('Uncaught Exception:'), error);
  process.exit(1);
});
