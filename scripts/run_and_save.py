from pathlib import Path
import json

from src.pipeline.run_pipeline import run_pipeline


def main():
    a = Path('data/raw/dummy_note_20260212.txt').read_text(encoding='utf-8')
    b = Path('data/raw/dummy_note_20260213.txt').read_text(encoding='utf-8')
    contradiction, detailed = run_pipeline(note_a=a, note_b=b, config_path='configs/config.yaml', verbose=True)

    out = {
        'contradiction': contradiction,
        'detailed_pairs': detailed,
    }
    out_path = Path('results')
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / 'run_output.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
