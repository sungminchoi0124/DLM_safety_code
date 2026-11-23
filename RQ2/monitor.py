"""Plain ASCII progress display; does not use terminal screen-control sequences."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent / 'outputs/full'


def show(conditions):
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    print(stamp, flush=True)
    for condition in conditions:
        path = ROOT / f'without_{condition}_seed0_n100/status.json'
        if not path.exists():
            print(f'Without {condition}: pending', flush=True)
            continue
        try:
            state = json.loads(path.read_text(encoding='utf-8-sig'))
            result = {'condition': condition, 'status': state.get('status'),
                      'n_done': state.get('n_done'), 'n_items': state.get('n_items')}
            print(json.dumps(result, ensure_ascii=True), flush=True)
        except (OSError, ValueError) as error:
            print(json.dumps({'condition': condition, 'read_error': str(error)}, ensure_ascii=True), flush=True)
    batches = ['without_0_A_batch_status.json', 'without_B_C_batch_status.json', 'without_D_batch_status.json', 'without_E_batch_status.json']
    for name in batches:
        path = ROOT / name
        if path.exists():
            try:
                state = json.loads(path.read_text(encoding='utf-8-sig'))
                print(json.dumps({'batch': name, 'status': state.get('status'),
                                  'current_condition': state.get('current_condition'), 'phase': state.get('phase')}, ensure_ascii=True), flush=True)
            except (OSError, ValueError) as error:
                print(json.dumps({'batch': name, 'read_error': str(error)}, ensure_ascii=True), flush=True)
    print('', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs='+', choices=['0', 'A', 'B', 'C', 'D', 'E'], default=['B', 'C'])
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            show(args.conditions)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
