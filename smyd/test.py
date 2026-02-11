import json
import numpy as np
import matplotlib.pyplot as plt

# 같은 폴더에 두 파일 두고 실행
orig_file = "3_cav2.json"
new_file  = "3_cav2_st.json"

def load_xy(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["X"], float), np.array(d["Y"], float)

xo, yo = load_xy(orig_file)
xn, yn = load_xy(new_file)

plt.figure()
plt.plot(xo, yo, label="orig")
plt.plot(xn, yn, label="linearized")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
