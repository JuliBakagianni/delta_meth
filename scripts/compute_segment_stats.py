#!/usr/bin/env python3
import json
import re
import statistics
from pathlib import Path


def count_sentences(text: str) -> int:
    if not text:
        return 0
    t = re.sub(r"\s+", " ", text.strip())
    parts = [x.strip() for x in re.split(r"[.!?]+(?=\s|$)", t) if x.strip()]
    return len(parts)


def quantile(xs, q):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    idx = (n - 1) * q
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def main():
    src = Path('data/processed/segments')
    files = sorted([p for p in src.glob('*.json') if p.is_file()])
    total = len(files)
    multi_count = 0
    sentences_per_segment = []

    for f in files:
        data = json.loads(f.read_text(encoding='utf-8'))
        segs = data.get('segments', [])
        if len(segs) > 1:
            multi_count += 1
            for s in segs:
                sentences_per_segment.append(count_sentences(s.get('text', '')))

    if total == 0:
        print('No segment JSON files found in', src)
        return

    pct_multi = multi_count / total * 100
    print(f'total_files = {total}')
    print(f'multi_segment_files = {multi_count} ({pct_multi:.2f}%)')

    if not sentences_per_segment:
        print('No multi-segment segments to compute statistics on.')
        return

    mean = statistics.mean(sentences_per_segment)
    std = statistics.stdev(sentences_per_segment) if len(sentences_per_segment) > 1 else 0.0
    mn = min(sentences_per_segment)
    mx = max(sentences_per_segment)
    p25 = quantile(sentences_per_segment, 0.25)
    p50 = quantile(sentences_per_segment, 0.5)
    p75 = quantile(sentences_per_segment, 0.75)

    print(f'segments_counted = {len(sentences_per_segment)}')
    print(f'mean = {mean:.3f}')
    print(f'std = {std:.3f}')
    print(f'min = {mn}')
    print(f'25% = {p25}')
    print(f'50% = {p50}')
    print(f'75% = {p75}')
    print(f'max = {mx}')


if __name__ == '__main__':
    main()
