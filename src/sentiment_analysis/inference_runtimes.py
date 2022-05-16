from google.colab import drive
drive.mount('/content/drive')

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
path = '/content/drive/MyDrive/SpecTopSci Project/runtime_csv(polarity)'
csv_files = glob.glob(os.path.join(path, "*.csv"))

dfs = []
i = 0

for f in csv_files:  
    # read the csv file
    dfs.append(pd.read_csv(f,index_col=0))
    # dfs[i].index = 

    # print the location and filename
    # print('Location:', f)
    # print('File Name:', f.split("\\")[-1])
      
    # print the content
    # print('Content:')
    # display(dfs)
    
    print()

filenames = []
for f in csv_files:
  filenames.append(f[84:-5])

runtimes = pd.concat(dfs)

barWidth = 0.33
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
IT = runtimes.runtime_og
ECE = runtimes.runtime_gen
# CSE = [29, 3, 24, 25, 17]
 
# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, IT, color ='r', width = barWidth,
        edgecolor ='grey', label ='Original')
plt.bar(br2, ECE, color ='g', width = barWidth,
        edgecolor ='grey', label ='Generated')
# plt.bar(br3, CSE, color ='b', width = barWidth,
        # edgecolor ='grey', label ='CSE')
 
# Adding Xticks
plt.xlabel('GPU', fontweight ='bold', fontsize = 15)
plt.ylabel('Runtimes', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth - 0.16 for r in range(len(IT))],
        runtimes.filenames)
plt.title('Inference Time')

plt.legend(loc='upper left')

plt.show()
