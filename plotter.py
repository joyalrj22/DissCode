import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


filepath = "/home2/fpnr38/emotion/Year-3-Project/AVEC2019/Baseline_systems/CES/predictions/Devel_DE_tf.Tensor(2, shape=(), dtype=int32).csv"
fp = "/home2/fpnr38/emotion/Year-3-Project/AVEC2019/Baseline_systems/CES/baseline_predictions/predictions_visual-xbow/Devel_DE_02.csv"
rawCSV = [i.strip().split("::")[0].split(';') for i in open(fp, 'r').readlines()][1:]
pred_df = pd.DataFrame(rawCSV, columns = ['Name', 'timestamp', 'Arousal', 'Valence', 'Liking'], dtype = int)

x = pred_df['timestamp'].to_numpy()
arousal = pred_df['Arousal'].to_numpy()
valence = pred_df['Valence'].to_numpy()
liking = pred_df['Liking'].to_numpy()

rawCSV1 = [i.strip().split("::")[0].split(';') for i in open(filepath, 'r').readlines()][1:]
pred_df1 = pd.DataFrame(rawCSV, columns = ['Name', 'timestamp', 'Arousal', 'Valence', 'Liking'], dtype = int)

x1 = pred_df1['timestamp'].to_numpy()
arousal1 = pred_df1['Arousal'].to_numpy()
valence1 = pred_df1['Valence'].to_numpy()
liking1 = pred_df1['Liking'].to_numpy()

plt.plot(moving_average(x), moving_average(arousal))
plt.plot(moving_average(x1), moving_average(arousal1))

plt.savefig("./testfig2.png")
