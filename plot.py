import pandas as pd
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()

plots = ['val_loss','ID_loss','GAN_loss','tot_loss','D_loss','cyc_loss']

fig, axs = plt.subplots(2,3)

for i in range(2):
    for k in range(3):
        df = pd.read_csv('run11/losses/%s.csv' % plots[i*3+k]) 
        # print(df.head())
        df.plot(x=0, y=1, ax=axs[i,k], figsize = (10,3), title = plots[i*3+k], xlim = (1), legend = False, sharex = True)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.5)

plt.savefig('graphs.jpeg', bbox_inches='tight')

end = datetime.datetime.now() - start
print(end)
