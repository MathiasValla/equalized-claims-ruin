#!/usr/bin/env python3
"""Regenerate the six original examples and the independent main-case run."""
import json
from pathlib import Path

from verify_local_gap import importance_lomax

def main():
    folder = Path(__file__).resolve().parents[1] / 'data'
    cases = [
        (.001, 1, .1, 10000, .95, 10),
        (.001, 1, 1, 100000, .01, .01),
        (.01, 1, 1, 100000, .01, .1),
        (.001, 1, 1, 10000, 1, 10),
        (.1, 1, 1, 10000, .1, 10),
        (.001, 1, 1, 10000, 1.0011, 10),
    ]
    results = [importance_lomax(500000, 20261101 + i, *case) for i, case in enumerate(cases)]
    (folder / 'six_examples_importance.json').write_text(json.dumps(results, indent=2) + '\n')
    validation = importance_lomax(2000000, 20261110, *cases[0])
    (folder / 'main_example_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    for result in results:
        print(result['mean'], result['normal_95_lower'], result['normal_95_upper'])
    print('Independent main run:', validation['mean'])


if __name__ == '__main__':
    main()
