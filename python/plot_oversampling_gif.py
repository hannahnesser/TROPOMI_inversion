import imageio
from os import listdir
from os.path import join

##########################
# Set default file paths
PLOT_DIR = '../plots/'
##########################

files = [f'northamerica_2019{m:02d}.png' for m in range(1, 13)]
files.sort()
print(files)

images = []
for f in files:
    images.append(imageio.imread(join(PLOT_DIR, f)))
imageio.mimsave(join(PLOT_DIR, 'northamerica.gif'), images,
                **{'duration' : 1})
