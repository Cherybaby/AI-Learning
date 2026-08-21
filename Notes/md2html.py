#!/usr/bin/env python3
"""Markdown → HTML 转换脚本
运行方式:
    python md2html.py                    # 转换脚本所在目录下所有 .md 文件
    python md2html.py input.md           # 转换单个文件
    python md2html.py input.md -o out.html
    python md2html.py *.md               # 批量转换指定的 .md 文件
"""

import os
import sys
import markdown
import glob as glob_module


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']],
    processEscapes: true,
  }}
}};
</script>
<style>
:root {{
  --bg: #ffffff;
  --fg: #1a1a1a;
  --border: #e0e0e0;
  --code-bg: #f5f5f5;
  --table-stripe: #fafafa;
  --link: #1a6fb5;
  --h1-border: #333;
  --h2-border: #888;
  --info-bg: #eef6fc;
  --info-border: #b8d8f0;
}}
body {{
  max-width: 860px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
  font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
  line-height: 1.8;
  color: var(--fg);
  background: var(--bg);
}}
h1 {{ font-size: 1.8rem; border-bottom: 2px solid var(--h1-border); padding-bottom: 0.3em; margin-top: 1.8em; }}
h2 {{ font-size: 1.4rem; border-bottom: 1px solid var(--h2-border); padding-bottom: 0.2em; margin-top: 1.6em; }}
h3 {{ font-size: 1.15rem; margin-top: 1.4em; }}
h4 {{ font-size: 1rem; margin-top: 1.2em; }}
a {{ color: var(--link); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{
  background: var(--code-bg);
  padding: 0.15em 0.4em;
  border-radius: 4px;
  font-size: 0.92em;
  font-family: "Fira Code", "Source Code Pro", Consolas, monospace;
}}
pre {{
  background: var(--code-bg);
  padding: 1em 1.2em;
  border-radius: 6px;
  overflow-x: auto;
  line-height: 1.5;
}}
pre code {{ background: none; padding: 0; }}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
}}
th, td {{
  border: 1px solid var(--border);
  padding: 0.55em 0.8em;
  text-align: left;
}}
th {{ background: var(--code-bg); font-weight: 600; }}
tr:nth-child(even) {{ background: var(--table-stripe); }}
blockquote {{
  border-left: 4px solid var(--info-border);
  background: var(--info-bg);
  margin: 1em 0;
  padding: 0.6em 1em;
  color: #555;
}}
hr {{ border: none; border-top: 1px solid var(--border); margin: 2em 0; }}
img {{ max-width: 100%; }}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #1e1e1e;
    --fg: #d4d4d4;
    --border: #444;
    --code-bg: #2d2d2d;
    --table-stripe: #252525;
    --link: #569cd6;
    --h1-border: #666;
    --h2-border: #555;
    --info-bg: #1a2a3a;
    --info-border: #2a4a6a;
  }}
  blockquote {{ color: #aaa; }}
}}
@media print {{
  body {{ max-width: 100%; padding: 0; font-size: 11pt; }}
  pre, blockquote {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
{body}
</body>
</html>"""


def convert_file(input_path, output_path=None):
    """将单个 .md 文件转换为 .html"""
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".html"

    with open(input_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=[
            "pymdownx.arithmatex",
            "markdown.extensions.tables",
            "markdown.extensions.fenced_code",
            "markdown.extensions.codehilite",
            "markdown.extensions.footnotes",
            "markdown.extensions.toc",
            "markdown.extensions.nl2br",
        ],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True,
            },
        },
    )

    title = os.path.splitext(os.path.basename(input_path))[0]
    full_html = HTML_TEMPLATE.format(title=title, body=html_body)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"  ✓ {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    return output_path


def main():
    args = sys.argv[1:]

    # 无参数 → 转换脚本所在目录下所有 .md 文件
    if not args:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        md_files = glob_module.glob(os.path.join(script_dir, "*.md"))
        if not md_files:
            print(f"脚本所在目录 ({script_dir}) 下没有找到 .md 文件。")
            return
        print(f"转换 {script_dir} 下的 {len(md_files)} 个 .md 文件：")
        for f in sorted(md_files):
            convert_file(f)
        print(f"\n全部完成，共 {len(md_files)} 个文件。")
        return

    # -o 模式：单个文件指定输出
    if len(args) == 3 and args[1] == "-o":
        input_path = args[0]
        output_path = args[2]
        if not os.path.isfile(input_path):
            print(f"错误：找不到文件 {input_path}")
            sys.exit(1)
        convert_file(input_path, output_path)
        return

    # 单文件默认输出
    if len(args) == 1 and not args[0].startswith("-"):
        input_path = args[0]
        if not os.path.isfile(input_path):
            print(f"错误：找不到文件 {input_path}")
            sys.exit(1)
        convert_file(input_path)
        return

    # 通配符批量转换
    md_files = []
    for arg in args:
        if arg.startswith("-"):
            continue
        matches = glob_module.glob(arg)
        md_files.extend([f for f in matches if f.lower().endswith(".md")])
    if md_files:
        print(f"转换 {len(md_files)} 个 .md 文件：")
        for f in sorted(md_files):
            convert_file(f)
        print(f"\n全部完成，共 {len(md_files)} 个文件。")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
