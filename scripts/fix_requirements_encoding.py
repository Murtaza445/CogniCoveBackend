from pathlib import Path
p = Path('requirements.txt')
if not p.exists():
    print('requirements.txt not found')
    raise SystemExit(1)
b = p.read_bytes()
for enc in ('utf-8','utf-16','utf-16-le','utf-16-be','latin-1'):
    try:
        s = b.decode(enc)
        p.write_text(s, encoding='utf-8')
        print('rewritten using', enc)
        break
    except Exception:
        pass
else:
    print('no suitable encoding found')
    raise SystemExit(2)
