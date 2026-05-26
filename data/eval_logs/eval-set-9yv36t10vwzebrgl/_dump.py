import json, sys
def dump(name, fn):
    p = f'data/.eval-cache/eval-set-9yv36t10vwzebrgl/{fn}.json'
    with open(p) as f: s = json.load(f)
    print('=====', name, '=====')
    for i, m in enumerate(s.get('messages', [])):
        role = m.get('role')
        content = m.get('content')
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get('type')
                    if t == 'text':
                        parts.append('[text] ' + (c.get('text') or '')[:4000])
                    elif t == 'reasoning':
                        parts.append('[reasoning] ' + (c.get('reasoning') or '')[:4000])
                    else:
                        parts.append(f'[{t}] ' + str(c)[:1000])
                else:
                    parts.append(str(c)[:1000])
            content = '\n'.join(parts)
        else:
            content = str(content)[:5000] if content else ''
        tcs = m.get('tool_calls')
        print(f'--- [{i}] role={role} ---')
        if content.strip():
            print(content[:5000])
        if tcs:
            for tc in tcs:
                fn_ = tc.get('function')
                args = tc.get('arguments')
                print(f'  TOOL: {fn_} args={json.dumps(args)[:3000] if args else ""}')
        if m.get('function'):
            print('  function:', m['function'])
        print()

targets = {
    'packedaway': 'jSGusgayNx5bZ5sKwXpEtx',
    'eval_me': 'Veh3hYRHWDeWrb9ojNAZnE',
    'it_has_begun': 'oDD5Yef7r5SURsKNpR4HTQ',
    'primary_knowledge': '4LS2uqKn42G7sB64Nn4oJQ',
    'urgent': 'Jv4za5YN8q28TYo5kbAoVf',
}
which = sys.argv[1]
dump(which, targets[which])
