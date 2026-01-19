import pandas as pd, numpy as np, math

in_path  = "smyd/hv_roundabout_v1.csv"
out_path = "smyd/hv_roundabout_v1_dense.csv"
step = 0.01  # 목표 간격 (0.01~0.02 추천)

df = pd.read_csv(in_path).dropna().reset_index(drop=True)
X, Y = df['X'].to_numpy(), df['Y'].to_numpy()

Xo, Yo = [X[0]], [Y[0]]
for i in range(len(X)-1):
    x1,y1 = X[i],Y[i]
    x2,y2 = X[i+1],Y[i+1]
    dist = math.hypot(x2-x1, y2-y1)
    n = int(dist/step)
    if n <= 1:
        Xo.append(x2); Yo.append(y2)
    else:
        for k in range(1, n+1):
            t = k/n
            Xo.append(x1 + t*(x2-x1))
            Yo.append(y1 + t*(y2-y1))

pd.DataFrame({"X":Xo, "Y":Yo}).to_csv(out_path, index=False)
print("saved:", out_path, "N:", len(Xo))
