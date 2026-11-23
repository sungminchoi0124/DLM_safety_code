"""Within-prompt test: for items whose outcome differs across seeds, does the probe rank harmful above refusal
trajectories of the SAME prompt? Leave-one-item-out training on all other R items. Prints numbers only."""
import csv, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
T="/root/rq3-outputs/trace"; runs=("R_s0","R_s1000","R_s2000")
region=sys.argv[1]; key={"all":"hidden","infill":"hidden_infill","tail":"hidden_tail"}[region]
fracs=(0.0,0.05,0.1,0.25,0.5,0.75,1.0)
H={}; Y={}
for r in runs:
    lab={int(x["item_index"]):x["label"] for x in csv.DictReader(open(f"{T}/{r}/labels.csv"))}
    for i,l in lab.items():
        if l not in ("harmful","refusal"): continue
        with np.load(f"{T}/{r}/hidden/item{i:03d}.npz") as z:
            names=[str(n) for n in z["layer_names"]]; h=z[key][:, names.index("b23"), :].astype(np.float32)
        H[(r,i)]=h; Y[(r,i)]=int(l=="harmful")
items=sorted({i for _,i in H}); mixed=[i for i in items if len({Y[k] for k in H if k[1]==i})==2]
print(region, "trajectories",len(H),"mixed items",len(mixed))
def feat(h,f): return h[min(int(round(f*(len(h)-1))),len(h)-1)]
res={f:[] for f in fracs}
for i in mixed:
    tr=[k for k in H if k[1]!=i]; te=[k for k in H if k[1]==i]
    Xtr=np.stack([H[k].mean(0) for k in tr]); ytr=np.array([Y[k] for k in tr])   # traj_mean training like fit
    sc=StandardScaler().fit(Xtr); clf=LogisticRegression(C=0.01,max_iter=2000).fit(sc.transform(Xtr),ytr)
    for f in fracs:
        m={k:float(clf.decision_function(sc.transform(feat(H[k],f)[None]))[0]) for k in te}
        pos=[m[k] for k in te if Y[k]==1]; neg=[m[k] for k in te if Y[k]==0]
        res[f]+= [1.0 if p>n else 0.5 if p==n else 0.0 for p in pos for n in neg]
print("within-prompt concordance (0.5 = chance):", " ".join(f"{f}:{np.mean(v):.2f}(n={len(v)})" for f,v in res.items()))
