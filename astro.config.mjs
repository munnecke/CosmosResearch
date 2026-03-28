import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import node from '@astrojs/node';

export default defineConfig({
  site: 'https://cosmosresearch.center',
  output: 'server',
  adapter: node({ mode: 'standalone' }),
  integrations: [mdx()],
});
