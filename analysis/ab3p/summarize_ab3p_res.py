import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


ab3p_fn = "../processed_data/preprocess/ab3p/summarize_ab3p/ab3p_freq.csv"
out_dir = "../processed_data/analysis/ab3p/"

os.makedirs(out_dir, exist_ok=True)

ab3p = pd.read_table(ab3p_fn)


plt.close()
fig, ax = plt.subplots()
ab3p.score.hist(bins=100, ax=ax)
fig.savefig(f"{out_dir}/score_hist.png")


plt.close()
fig, ax = plt.subplots()
ab3p.freq.hist(bins=10000, ax=ax)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Freq of SF-LF pairs")
ax.set_ylabel("Frequency")
fig.savefig(f"{out_dir}/freq_hist.png")


plt.close()
fig, ax = plt.subplots()
ax.plot(ab3p["freq"], ab3p["pmids"])
ax.set_xlabel("SF-LF Frequency")
ax.set_ylabel("# PMIDs SF-LF appears in")
fig.savefig(f"{out_dir}/freq5pmid.png")

plt.close()
fig, ax = plt.subplots()
ax.scatter(ab3p["freq"], ab3p["score"])
ax.set_yscale("linear")
ax.set_xscale("log")

ax.set_xlabel("SF-LF Frequency")
ax.set_ylabel("# PMIDs SF-LF appears in")
fig.savefig(f"{out_dir}/freq5pmid.png")

plt.ion()
plt.show()