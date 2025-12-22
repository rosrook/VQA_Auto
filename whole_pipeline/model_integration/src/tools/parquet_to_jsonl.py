"""
从大型 Parquet 中提取前 N 条样本，输出 JSONL 与 Markdown 预览。

功能：
- 流式读取（低内存）：使用 pyarrow.ParquetFile.iter_batches
- 字段提取：question、answer、image/jpg/img（优先级依次降低）
- 如果存在 hint，追加在 question 后（换行）
- 输出：
  1) JSONL：前 N 条样本，每条包含 question、answer、image(base64)
  2) Markdown：前 3 条样本的可视化预览（内联 base64 图像）

依赖：
  pip install pyarrow

用法示例：
  python parquet_to_jsonl.py \
    --input data/huge.parquet \
    --output out/preview.jsonl \
    --md out/preview.md \
    --limit 100
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import pyarrow.parquet as pq


# 目标字段优先级
IMAGE_KEYS = ["image", "jpg", "img"]
QUESTION_KEY = "question"
ANSWER_KEY = "answer"
HINT_KEY = "hint"
# 可能的选项字段（按顺序追加）
OPTION_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def is_base64_image(s: Optional[str]) -> bool:
    if not s or not isinstance(s, str):
        return False
    s_strip = s.strip()
    return s_strip.startswith("data:image")


def normalize_image(b64: Optional[str]) -> Optional[str]:
    """确保返回 data:image/...;base64, 开头的字符串。"""
    if not b64:
        return None
    b64 = b64.strip()
    if b64.startswith("data:image"):
        return b64
    # 如果是裸的 base64，补齐前缀
    return f"data:image/jpeg;base64,{b64}"


def pick_image(row: Dict[str, Any]) -> Optional[str]:
    for key in IMAGE_KEYS:
        val = row.get(key)
        if val:
            return normalize_image(str(val))
    return None


def build_question(question: Optional[str], hint: Optional[str], options: Optional[List[str]]) -> Optional[str]:
    """构建最终 question，附加 hint 与选项"""
    if question is None:
        return None
    q = str(question)
    parts = [q]
    if hint:
        parts.append(str(hint))
    if options:
        parts.extend(options)
    return "\n".join(parts)


def iter_rows(pf: pq.ParquetFile, limit: int) -> List[Dict[str, str]]:
    # 只选择需要的列，避免无谓读取
    available_cols = set(pf.schema.names)
    cols = [c for c in [QUESTION_KEY, ANSWER_KEY, HINT_KEY, *IMAGE_KEYS, *OPTION_KEYS] if c in available_cols]

    results = []
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        data = batch.to_pydict()
        # 每列都是 list
        length = len(next(iter(data.values()))) if data else 0
        for i in range(length):
            q = data.get(QUESTION_KEY, [None] * length)[i]
            a = data.get(ANSWER_KEY, [None] * length)[i]
            h = data.get(HINT_KEY, [None] * length)[i]

            # 选项收集（按 A,B,C... 顺序，非空才收）
            opts = []
            for key in OPTION_KEYS:
                if key in data:
                    val = data[key][i]
                    if val is not None:
                        opts.append(f"{key}: {val}")

            img = None
            for k in IMAGE_KEYS:
                if k in data:
                    val = data[k][i]
                    if val:
                        img = normalize_image(str(val))
                        break

            q_final = build_question(q, h, opts if opts else None)
            a_final = None if a is None else str(a)

            if q_final is None or a_final is None or img is None:
                # 跳过缺失关键字段的样本
                continue

            results.append({
                "question": q_final,
                "answer": a_final,
                "image": img
            })

            if len(results) >= limit:
                return results
    return results


def write_jsonl(samples: List[Dict[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_markdown(samples: List[Dict[str, str]], path: Path, top_k: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    top_samples = samples[:top_k]
    lines = ["# Preview (Top 3)\n"]
    for idx, s in enumerate(top_samples, 1):
        lines.append(f"## Sample {idx}")
        lines.append(f"- **Question:** {s['question'].replace(chr(10), '<br>')}")
        lines.append(f"- **Answer:** {s['answer']}")
        # 仅使用 HTML img，避免某些渲染器对 Markdown 链接的 KaTeX 误解析
        lines.append("- **Image:**")
        lines.append("")
        lines.append(f'<p><img src="{s["image"]}" alt="sample image" style="max-width:320px;" /></p>')
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(samples: List[Dict[str, str]], path: Path, top_k: int = 3):
    """单独输出 HTML，最大化渲染兼容性"""
    path.parent.mkdir(parents=True, exist_ok=True)
    top_samples = samples[:top_k]
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Preview</title></head><body>",
        "<h1>Preview (Top 3)</h1>",
    ]
    for idx, s in enumerate(top_samples, 1):
        html_parts.append(f"<h2>Sample {idx}</h2>")
        html_parts.append(f"<p><strong>Question:</strong><br>{s['question'].replace(chr(10), '<br>')}</p>")
        html_parts.append(f"<p><strong>Answer:</strong> {s['answer']}</p>")
        html_parts.append(f"<p><strong>Image:</strong><br><img src=\"{s['image']}\" alt=\"sample image\" style=\"max-width:480px;\" /></p>")
    html_parts.append("</body></html>")
    path.write_text("\n".join(html_parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Extract top-N samples from a large Parquet to JSONL & Markdown.")
    parser.add_argument("--input", required=True, help="Input Parquet file path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--md", required=True, help="Output Markdown preview path")
    parser.add_argument("--html", required=False, help="Optional HTML preview path")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to extract (default: 100)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    md_path = Path(args.md)
    html_path = Path(args.html) if args.html else None

    if not input_path.exists():
        raise FileNotFoundError(f"Parquet文件不存在: {input_path}")

    # 流式读取，低内存
    pf = pq.ParquetFile(input_path)
    samples = iter_rows(pf, limit=args.limit)

    if not samples:
        raise ValueError("未能提取到有效样本，请检查数据字段（question/answer/image或jpg或img）。")

    write_jsonl(samples, output_path)
    write_markdown(samples, md_path, top_k=3)
    if html_path:
        write_html(samples, html_path, top_k=3)

    print(f"✓ 已提取 {len(samples)} 条样本 -> {output_path}")
    print(f"✓ Markdown预览 -> {md_path}")
    if html_path:
        print(f"✓ HTML预览 -> {html_path}")


if __name__ == "__main__":
    main()

