import rss from '@astrojs/rss';

export async function GET(context) {
  const posts = import.meta.glob('./blog/*.mdx', { eager: true });
  const items = Object.values(posts)
    .sort((a, b) =>
      new Date(b.frontmatter.pubDate).getTime() -
      new Date(a.frontmatter.pubDate).getTime()
    )
    .map((post) => ({
      title: post.frontmatter.title,
      pubDate: new Date(post.frontmatter.pubDate),
      description: post.frontmatter.description || '',
      link: post.url,
    }));

  return rss({
    title: 'Cosmos Research',
    description: 'A multi-generational science and ideas platform.',
    site: context.site,
    items,
  });
}
