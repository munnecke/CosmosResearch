# Cosmos Research Center

**cosmosresearch.center** — a multi-generational science and ideas platform.

Built with [Astro](https://astro.build), deployed on [Railway](https://railway.app), DNS via Cloudflare.

---

## Local development

```bash
npm install
npm run dev        # starts at http://localhost:4321
```

## Project structure

```
src/
  pages/
    index.astro              ← home page
    about.astro              ← about page
    blog/
      index.astro            ← blog listing
      *.mdx                  ← blog posts (add more here)
    lab/
      index.astro            ← lab listing
    rss.xml.js               ← RSS feed
  layouts/
    Base.astro               ← site shell (nav, footer)
    BlogPost.astro           ← blog post wrapper
  styles/
    global.css               ← full design system
public/
  favicon.svg
railway.toml                 ← Railway deployment config
astro.config.mjs
```

## Writing a new blog post

Create a new `.mdx` file in `src/pages/blog/`:

```mdx
---
layout: ../../layouts/BlogPost.astro
title: Your post title
description: One sentence summary for SEO and RSS.
pubDate: 2026-04-01
author: Tom          # or Abigail
tag: Essay           # Essay | Research | Reflection | Physics | Health
---

Your content here in Markdown...
```

Push to `main` → Railway auto-deploys.

## Deploying to Railway

1. Push this repo to GitHub
2. Create a new Railway project → "Deploy from GitHub repo"
3. Railway auto-detects `railway.toml` — no config needed
4. Add custom domain `cosmosresearch.center` in Railway settings
5. Copy the CNAME record Railway gives you into Cloudflare DNS
6. Set SSL/TLS to "Full" in Cloudflare

## Environment variables

None required for the base site. When adding lab services (FastAPI, etc.), add them in Railway's Variables panel.

## Adding lab experiments

The lab page (`src/pages/lab/index.astro`) currently uses a static array. To add an item, edit the `items` array at the top of that file. Future step: move to a database or CMS when the list grows.
