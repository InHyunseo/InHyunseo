# linearize_wp.py
import json
import argparse

def pick_keys(d):
    if "X" in d and "Y" in d:
        return "X", "Y"
    if "x" in d and "y" in d:
        return "x", "y"
    raise KeyError("waypoint keys not found (expected X/Y or x/y)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--a0", type=int, required=True, help="start anchor index (kept)")
    ap.add_argument("--a1", type=int, required=True, help="end anchor index (kept)")
    args = ap.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        d = json.load(f)

    kx, ky = pick_keys(d)
    X = list(d[kx])
    Y = list(d[ky])

    n = len(X)
    a0, a1 = args.a0, args.a1
    if not (0 <= a0 < a1 < n):
        raise ValueError(f"bad indices: a0={a0}, a1={a1}, len={n}")

    x0, y0 = X[a0], Y[a0]
    x1, y1 = X[a1], Y[a1]
    denom = (a1 - a0)

    # replace interior points (a0+1 .. a1-1) by linear interpolation
    for i in range(a0 + 1, a1):
        t = (i - a0) / denom
        X[i] = x0 + t * (x1 - x0)
        Y[i] = y0 + t * (y1 - y0)

    d[kx] = X
    d[ky] = Y

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    print(f"saved: {args.outfile}  (linearized {a0+1}..{a1-1}, anchors kept {a0},{a1})")

if __name__ == "__main__":
    main()
