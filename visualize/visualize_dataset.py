from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import html
from pathlib import Path
import statistics
import string
from typing import Mapping, cast

from datasets import load_from_disk
from tokenizers import Tokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
VIS_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "data" / "opus_books_en_it"
TOKENIZER_EN = ROOT_DIR / "data" / "tokenizer_en.json"
TOKENIZER_IT = ROOT_DIR / "data" / "tokenizer_it.json"
OUT_HTML = VIS_DIR / "dataset_overview.html"

SAMPLE_SIZE = 800
EXAMPLE_COUNT = 12


@dataclass(frozen=True)
class RowStats:
    index: int
    en: str
    it: str
    en_words: int
    it_words: int
    en_tokens: int
    it_tokens: int


def pick_even_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count <= 1:
        return [0]

    step = (total - 1) / (count - 1)
    return sorted({round(i * step) for i in range(count)})


def count_words(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def collect_words(text: str) -> list[str]:
    punctuation = string.punctuation
    words: list[str] = []
    for raw in text.lower().split():
        word = raw.strip(punctuation)
        if len(word) > 2:
            words.append(word)
    return words


def summarize(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def fmt_number(value: float) -> str:
    if isinstance(value, int) or value == int(value):
        return f"{int(value):,}"
    return f"{value:,.1f}"


def histogram_svg(title: str, values: list[int], bucket_size: int, color: str) -> str:
    width = 760
    height = 280
    margin_left = 54
    margin_right = 18
    margin_top = 34
    margin_bottom = 46
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    if not values:
        return ""

    bucket_count = max(values) // bucket_size + 1
    counts = [0] * bucket_count
    for value in values:
        counts[min(value // bucket_size, bucket_count - 1)] += 1

    max_count = max(counts) or 1
    bar_w = plot_w / bucket_count
    label_step = max(1, bucket_count // 8)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{margin_left}" y="22" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" class="axis" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis" />',
    ]

    for i in range(5):
        y = margin_top + plot_h - (plot_h * i / 4)
        count_label = round(max_count * i / 4)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid" />')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{count_label}</text>')

    for i, count in enumerate(counts):
        x = margin_left + i * bar_w + 1
        h = plot_h * count / max_count
        y = margin_top + plot_h - h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1, bar_w - 2):.1f}" '
            f'height="{h:.1f}" fill="{color}" opacity="0.82" />'
        )
        if i % label_step == 0:
            label = i * bucket_size
            parts.append(f'<text x="{x:.1f}" y="{height - 18}" class="tick" transform="rotate(45 {x:.1f},{height - 18})">{label}</text>')

    parts.append(f'<text x="{width / 2:.1f}" y="{height - 3}" class="axis-label" text-anchor="middle">Bucket start</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def scatter_svg(rows: list[RowStats]) -> str:
    width = 760
    height = 420
    margin_left = 58
    margin_right = 22
    margin_top = 36
    margin_bottom = 54
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_x = max((row.en_tokens for row in rows), default=1)
    max_y = max((row.it_tokens for row in rows), default=1)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Source and target token lengths">',
        f'<text x="{margin_left}" y="24" class="chart-title">Source vs target token length</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" class="axis" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis" />',
    ]

    for i in range(5):
        x = margin_left + plot_w * i / 4
        y = margin_top + plot_h - plot_h * i / 4
        x_label = round(max_x * i / 4)
        y_label = round(max_y * i / 4)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" class="grid" />')
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid" />')
        parts.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 20}" class="tick" text-anchor="middle">{x_label}</text>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{y_label}</text>')

    for row in rows:
        x = margin_left + row.en_tokens / max_x * plot_w
        y = margin_top + plot_h - row.it_tokens / max_y * plot_h
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" class="dot"><title>row {row.index}: en {row.en_tokens}, it {row.it_tokens}</title></circle>')

    parts.append(f'<text x="{width / 2:.1f}" y="{height - 10}" class="axis-label" text-anchor="middle">English tokens</text>')
    parts.append(f'<text x="16" y="{height / 2:.1f}" class="axis-label" transform="rotate(-90 16,{height / 2:.1f})" text-anchor="middle">Italian tokens</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def top_words_html(title: str, counts: Counter[str], color: str) -> str:
    top_items = counts.most_common(15)
    max_count = max((count for _, count in top_items), default=1)
    rows = [f"<h2>{html.escape(title)}</h2>", '<div class="word-bars">']
    for word, count in top_items:
        pct = count / max_count * 100
        rows.append(
            '<div class="word-row">'
            f'<span class="word">{html.escape(word)}</span>'
            f'<div class="bar"><span style="width:{pct:.1f}%; background:{color};"></span></div>'
            f'<span class="count">{count}</span>'
            "</div>"
        )
    rows.append("</div>")
    return "\n".join(rows)


def stat_card(label: str, value: str, hint: str = "") -> str:
    return (
        '<section class="stat-card">'
        f'<span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong>'
        f'<small>{html.escape(hint)}</small>'
        '</section>'
    )


def examples_html(rows: list[RowStats]) -> str:
    parts = [
        "<h2>Sample rows</h2>",
        "<table>",
        "<thead><tr><th>Index</th><th>English</th><th>Italian</th><th>Tokens</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{row.index}</td>"
            f"<td>{html.escape(row.en)}</td>"
            f"<td>{html.escape(row.it)}</td>"
            f"<td>{row.en_tokens} / {row.it_tokens}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def build_html(total_rows: int, sample_rows: list[RowStats]) -> str:
    en_word_stats = summarize([row.en_words for row in sample_rows])
    it_word_stats = summarize([row.it_words for row in sample_rows])
    en_token_stats = summarize([row.en_tokens for row in sample_rows])
    it_token_stats = summarize([row.it_tokens for row in sample_rows])

    en_words = Counter()
    it_words = Counter()
    for row in sample_rows:
        en_words.update(collect_words(row.en))
        it_words.update(collect_words(row.it))

    example_indices = pick_even_indices(len(sample_rows), EXAMPLE_COUNT)
    examples = [sample_rows[i] for i in example_indices]

    cards = "\n".join(
        [
            stat_card("Total rows", fmt_number(total_rows), "local dataset"),
            stat_card("Sample rows", fmt_number(len(sample_rows)), "evenly spaced"),
            stat_card("EN words avg", fmt_number(en_word_stats["mean"]), f'median {fmt_number(en_word_stats["median"])}'),
            stat_card("IT words avg", fmt_number(it_word_stats["mean"]), f'median {fmt_number(it_word_stats["median"])}'),
            stat_card("EN tokens avg", fmt_number(en_token_stats["mean"]), f'max {fmt_number(en_token_stats["max"])}'),
            stat_card("IT tokens avg", fmt_number(it_token_stats["mean"]), f'max {fmt_number(it_token_stats["max"])}'),
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OPUS Books en-it Dataset Overview</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #697386;
      --line: #d9e0eb;
      --blue: #2563eb;
      --green: #059669;
      --shadow: 0 1px 3px rgba(20, 30, 50, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto 40px;
    }}
    header {{
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .stat-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .stat-card {{
      min-height: 106px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .stat-card span, .stat-card small {{
      color: var(--muted);
      font-size: 13px;
    }}
    .stat-card strong {{
      font-size: 26px;
      line-height: 1.1;
    }}
    .grid-panels {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .panel {{
      padding: 16px;
      overflow: hidden;
    }}
    .panel.full {{
      grid-column: 1 / -1;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .chart-title {{
      font-size: 16px;
      font-weight: 700;
      fill: var(--text);
    }}
    .axis {{
      stroke: #8290a3;
      stroke-width: 1.2;
    }}
    .grid {{
      stroke: #e6ebf2;
      stroke-width: 1;
    }}
    .tick, .axis-label {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .dot {{
      fill: var(--blue);
      opacity: 0.34;
    }}
    .word-bars {{
      display: grid;
      gap: 8px;
    }}
    .word-row {{
      display: grid;
      grid-template-columns: minmax(86px, 120px) 1fr 46px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }}
    .word {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .bar {{
      height: 9px;
      background: #edf1f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
    }}
    .count {{
      color: var(--muted);
      text-align: right;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 9px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
    }}
    td:first-child, td:last-child {{
      white-space: nowrap;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .grid-panels {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 24px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>OPUS Books en-it Dataset Overview</h1>
      <p>Generated from local data in <code>data/opus_books_en_it</code>. Charts use an evenly spaced sample, not the full dataset.</p>
    </header>
    <section class="stats">
      {cards}
    </section>
    <section class="grid-panels">
      <section class="panel full">
        {scatter_svg(sample_rows)}
      </section>
      <section class="panel">
        {histogram_svg("English word length distribution", [row.en_words for row in sample_rows], 10, "var(--blue)")}
      </section>
      <section class="panel">
        {histogram_svg("Italian word length distribution", [row.it_words for row in sample_rows], 10, "var(--green)")}
      </section>
      <section class="panel">
        {top_words_html("Frequent English words", en_words, "var(--blue)")}
      </section>
      <section class="panel">
        {top_words_html("Frequent Italian words", it_words, "var(--green)")}
      </section>
      <section class="panel full">
        {examples_html(examples)}
      </section>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Missing dataset directory: {DATASET_DIR}")
    if not TOKENIZER_EN.exists():
        raise FileNotFoundError(f"Missing English tokenizer: {TOKENIZER_EN}")
    if not TOKENIZER_IT.exists():
        raise FileNotFoundError(f"Missing Italian tokenizer: {TOKENIZER_IT}")

    dataset = load_from_disk(str(DATASET_DIR))
    tokenizer_en = Tokenizer.from_file(str(TOKENIZER_EN))
    tokenizer_it = Tokenizer.from_file(str(TOKENIZER_IT))

    sample_indices = pick_even_indices(len(dataset), SAMPLE_SIZE)
    sample_rows: list[RowStats] = []
    for index in sample_indices:
        row = cast(Mapping[str, Mapping[str, str]], dataset[index])
        translation = row["translation"]
        en = translation["en"]
        it = translation["it"]
        sample_rows.append(
            RowStats(
                index=index,
                en=en,
                it=it,
                en_words=count_words(en),
                it_words=count_words(it),
                en_tokens=len(tokenizer_en.encode(en).ids),
                it_tokens=len(tokenizer_it.encode(it).ids),
            )
        )

    OUT_HTML.write_text(build_html(len(dataset), sample_rows), encoding="utf-8")
    print(f"Wrote {OUT_HTML.relative_to(ROOT_DIR)}")
    print(f"Rows: {len(dataset):,}; sampled: {len(sample_rows):,}")


if __name__ == "__main__":
    main()
