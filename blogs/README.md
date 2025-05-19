# Blog System for simonxin.com

This directory contains the blog system for simonxin.com.

## How to Add a New Blog Post

1. Create a new directory in the `/blogs` directory with a URL-friendly name (e.g., `my-new-post`).
2. You can copy the entire `template` directory as a starting point: `cp -r blogs/template blogs/my-new-post`
3. Edit the `post.md` file in your new directory with your blog post content in Markdown format.
4. Update the title in both `post.md` and `index.html` files.
5. Update the `blog_list.json` file in the `/blogs` directory to include your new blog post.

### Table of Contents

Your blog posts will automatically generate a foldable table of contents based on the headings in your Markdown file (h1, h2, h3, h4). The outline will appear at the top of your blog post, making it easy for readers to navigate through longer articles. Readers can expand or collapse the table of contents by clicking on it.

To get the best results:
- Use a clear hierarchy of headings in your post
- Start with an h1 (`#`) for the title
- Use h2 (`##`) for main sections
- Use h3 (`###`) and h4 (`####`) for subsections

The table of contents is set to be open by default. If you prefer to have it closed by default, you can modify the `<details open>` tag in the HTML file by removing the `open` attribute.

### Markdown Format

Your `post.md` file should follow Markdown syntax. Here are some examples:

```markdown
# Title of the Post

*Published on Month Day, Year*

This is a paragraph.

## Subheading

1. Numbered list item
2. Another numbered list item

- Bullet point
- Another bullet point

[Link text](https://example.com)

![Image alt text](path/to/image.jpg)

**Bold text**

*Italic text*

`inline code`

```language
// Code block
function example() {
  return "Hello World";
}
```
```

### Adding to blog_list.json

Add a new entry to the `blog_list.json` file with the following format:

```json
{
  "title": "Your Blog Post Title",
  "date": "YYYY-MM-DD",
  "url": "./your-post-directory/index.html",
  "summary": "A brief summary of your blog post."
}
```

## Accessing Your Blog

Your blog posts will be accessible at:
- Blog Index: simonxin.com/blogs
- Individual Posts: simonxin.com/blogs/your-post-directory 