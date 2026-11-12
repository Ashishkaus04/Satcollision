"""Keep the newest element set per NORAD number from a Space-Track gp_history 3LE file."""
import sys
src, dst = sys.argv[1], sys.argv[2]
L = [l.rstrip("\r\n") for l in open(src) if l.strip()]
best = {}
i = 0
while i + 2 < len(L) + 1 and i < len(L):
    if L[i].startswith("0 ") and i + 2 < len(L) and L[i + 1].startswith("1 ") and L[i + 2].startswith("2 "):
        name, l1, l2 = L[i][2:].strip(), L[i + 1], L[i + 2]
        key, ep = l1[2:7], float(l1[18:32])
        if key not in best or ep > best[key][0]:
            best[key] = (ep, name, l1, l2)
        i += 3
    else:
        i += 1
with open(dst, "w") as f:
    for _, name, l1, l2 in sorted(best.values(), key=lambda t: t[2][2:7]):
        f.write(f"{name}\n{l1}\n{l2}\n")
print(dst, len(best))
