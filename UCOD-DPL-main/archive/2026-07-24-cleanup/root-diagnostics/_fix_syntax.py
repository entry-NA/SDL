with open('scripts/run_vloose_refine.py', 'r', encoding='utf-8') as f:
    c = f.read()
c = c.replace(
    '    print(f\'\\nDone! {stats[\"total\"]} refined, {stats[\"no_points\"]} skipped, {stats[\"errors\"]} errors\')',
    '    t = stats[\"total\"]; n = stats[\"no_points\"]; e = stats[\"errors\"]\n    print(\'\\nDone!\', t, \'refined,\', n, \'skipped,\', e, \'errors\')')
with open('scripts/run_vloose_refine.py', 'w', encoding='utf-8') as f:
    f.write(c)
print('Fixed')
