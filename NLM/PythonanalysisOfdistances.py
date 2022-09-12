import numpy as np
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# cada 196 medidas, es un parche distinto
text_file = open("Distances.txt", "r")
lines = text_file.read()
vector=lines.split(' ')
vector=vector[1:]
print(len(vector))

j=0
df=pd.DataFrame()
for i in range(0,len(vector),196):
  df[f'parche {j}'] = vector[i:i+196]
  j=j+1
df['parche 0']

sns.set(rc={'figure.figsize':(16,10)})
sns.set_style("ticks")
sns.set_context("poster", font_scale = 1)
sns.distplot(df[['parche 1']], rug=True, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
sns.distplot(df[['parche 2']], rug=True,hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
sns.distplot(df[['parche 3']], rug=True,hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
sns.distplot(df[['parche 4']], rug=True,hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
plt.legend(labels=['parche 1', 'parche 2', 'parche 3','parche 4'])

plt.show()
plt.clf()
sns.set(rc={'figure.figsize':(16,10)})
sns.set_style("ticks")
sns.set_context("poster", font_scale = 1)
sns.distplot(np.array(vector).astype(np.float), rug=True, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
plt.show()
plt.clf()

count, bins_count = np.histogram(np.array(vector).astype(np.float), bins=200)
  
# finding the PDF of the histogram using count values
pdf = count / sum(count)
  
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
  
# plotting PDF and CDF
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()
plt.show()

print(bins_count[1:],cdf)