#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: plot_running_means.py
Author: naught101
Email: naught101@email.com
Github: https://github.com/naught101/
Description: plots moving window and exponential averages
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pals_utils.df import get_met_df


df = get_met_df(['Tumba'], ['Rainf'])

fig, axes = plt.subplots(nrows=4)

df.plot(ax=axes[0])
plt.text(0.05, 0.9, 'raw df', transform=axes[0].transAxes)

df.rolling(3 * 48).mean().rename(columns={'Rainf': 'MW'}).plot(ax=axes[1])
df.ewm(3 * 48).mean().rename(columns={'Rainf': 'EWMA adj'}).plot(ax=axes[1])
df.ewm(3 * 48, adjust=False).mean().rename(columns={'Rainf': 'EWMA'}).plot(ax=axes[1])
plt.text(0.05, 0.9, '3 day window/COM', transform=axes[1].transAxes)

df.rolling(30 * 40).mean().rename(columns={'Rainf': 'MW'}).plot(ax=axes[2])
df.ewm(30 * 40).mean().rename(columns={'Rainf': 'EWMA adj'}).plot(ax=axes[2])
df.ewm(30 * 40, adjust=False).mean().rename(columns={'Rainf': 'EWMA'}).plot(ax=axes[2])
plt.text(0.05, 0.9, '30 day window/COM', transform=axes[2].transAxes)

df.rolling(300 * 48).mean().rename(columns={'Rainf': 'MW'}).plot(ax=axes[3])
df.ewm(300 * 48).mean().rename(columns={'Rainf': 'EWMA adj'}).plot(ax=axes[3])
df.ewm(300 * 48, adjust=False).mean().rename(columns={'Rainf': 'EWMA'}).plot(ax=axes[3])
plt.text(0.1, 0.1, '300 day window/COM', transform=axes[3].transAxes)

initialised = np.full_like(df, np.nan)
initialised[0] = df.mean()
com = 300 * 48
alpha = 1 / (1 + com)
for i in range(1, df.shape[0]):
    initialised[i] = alpha * df['Rainf'][i] + (1 - alpha) * initialised[i - 1]

pd.DataFrame(initialised, columns=['initialised'], index=df.index).plot(ax=axes[3])

s = np.full([70129, 1], np.nan)
s[0] = df.mean()
s[1:] = df
s = pd.DataFrame(s, columns=['Prefixed'])
s.ewm(300 * 48, adjust=False).mean()[1:].set_index(df.index).plot(ax=axes[3])


plt.savefig('plots/running_mean_comparison.png')
