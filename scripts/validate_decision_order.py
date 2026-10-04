#!/usr/bin/env python3
"""Audit real native candidate scoring; never replace or rescue a model choice."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, action='append', default=[])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-seq-max', type=int, default=8)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--max-thinking-tokens', type=int, default=512)
    parser.add_argument('--case', action='append')
    parser.add_argument('--rotations', type=int, default=4)
    args = parser.parse_args()
    directories = [os.add_dll_directory(str(p.resolve())) for p in args.library_dir] if os.name == 'nt' else []
    lib = C.CDLL(str(args.library.resolve()))
    lib.tj_engine_open_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
    lib.tj_engine_evaluate_json.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]
    lib.tj_engine_close.argtypes = [C.c_void_p]
    lib.tj_string_free.argtypes = [C.c_void_p]
    lib.tj_last_error_message.restype = C.c_char_p
    engine = C.c_void_p()
    config = {'model': str(args.model.resolve()), 'device': 'cpu', 'nGpuLayers': 0, 'nThreads': 4, 'nThreadsBatch': 4, 'nCtx': 4096, 'nSeqMax': args.n_seq_max, 'thinking': args.thinking, 'maxThinkingTokens': args.max_thinking_tokens, 'execution': {'orderRobustness': 'none'}}
    def check(status):
        if status:
            raise RuntimeError(lib.tj_last_error_message().decode())
    check(lib.tj_engine_open_json(json.dumps(config).encode(), C.byref(engine)))
    cases = [
        ('replacement', 'My package arrived broken. Please send a replacement.', 'Which team should handle this request?', {'dog': 'An animal', 'billing': 'Payments and invoices', 'shipping': 'Delivery, damage and replacement', 'technical': 'Software bugs'}, 'shipping'),
        ('payment', 'I was charged twice for my subscription. Please refund the duplicate payment.', 'Which team should handle this request?', {'dog': 'An animal', 'shipping': 'Delivery and replacement', 'technical': 'Software bugs', 'billing': 'Payments and refunds'}, 'billing'),
        ('software', 'The application crashes every time I open the settings screen.', 'Which team should handle this request?', {'billing': 'Payments and invoices', 'dog': 'An animal', 'shipping': 'Delivery and replacement', 'technical': 'Application bugs and crashes'}, 'technical'),
        ('color', 'The traffic signal is red.', 'What color is the traffic signal?', {'blue': 'Blue', 'green': 'Green', 'red': 'Red', 'yellow': 'Yellow'}, 'red'),
        ('direction', 'The food is directly to the right of the snake. The cell on the right is empty and inside the board.', 'Which direction should the snake move to reach the food?', {'up': 'Move up', 'down': 'Move down', 'left': 'Move left', 'right': 'Move right'}, 'right'),
        ('snake-wall', 'Board: 12 columns and 10 rows. Coordinates are (x, y); x increases to the right and y increases downward. Head: (6, 0). Body from neck to tail: [(6, 1)]. Food: (11, 0). Current direction: up.', "Choose the next direction for the snake to reach the food. Avoid the board walls and the snake's body; do not reverse directly. The tail moves away unless the snake eats. Decide this move only.", {'up': 'Move one cell up (y - 1).', 'down': 'Move one cell down (y + 1).', 'left': 'Move one cell left (x - 1).', 'right': 'Move one cell right (x + 1).'}, 'right'),
    ]
    records = []
    try:
        for name, state, instruction, criteria, expected in cases:
            if args.case and name not in args.case: continue
            items = list(criteria.items())
            for offset in range(min(args.rotations,len(items))):
                order = items[offset:] + items[:offset]
                request = {'state': state, 'questions': {'answer': {'type': 'choice', 'instructions': instruction, 'criteria': dict(order)}}}
                output = C.c_void_p()
                started = time.perf_counter()
                status = lib.tj_engine_evaluate_json(engine, json.dumps(request).encode(), C.byref(output))
                if status:
                    error = lib.tj_last_error_message().decode()
                    records.append({'case':name,'request':request,'expected':expected,'choice':None,'correct':False,'first':order[0][0],'seconds':time.perf_counter()-started,'error':error})
                    print(f'{name} first={order[0][0]} error={error}',flush=True)
                    continue
                try:
                    response = json.loads(C.string_at(output).decode())
                finally:
                    lib.tj_string_free(output)
                choice = response['answers']['answer']['choice']
                record = {'case': name, 'request': request, 'response': response, 'expected': expected, 'choice': choice, 'correct': choice == expected, 'first': order[0][0], 'seconds': time.perf_counter() - started}
                records.append(record)
                print(f'{name} first={record["first"]} choice={choice} correct={record["correct"]}', flush=True)
    finally:
        check(lib.tj_engine_close(engine))
        # Retain the Windows DLL search handles until all native calls finish.
        del directories
    with args.model.open('rb') as model_file:
        checksum = hashlib.file_digest(model_file, 'sha256').hexdigest()
    report = {'model': str(args.model.resolve()), 'sha256': checksum, 'config': config, 'library': str(args.library.resolve()), 'correct': sum(r['correct'] for r in records), 'total': len(records), 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{report["correct"]}/{report["total"]}; report={args.output}')


if __name__ == '__main__':
    main()
