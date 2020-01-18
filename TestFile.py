from PIL import Image
import numpy as np

from ImagePatch import ImagePatch
from RandomPatchSelector import RandomPatchSelector

patches = RandomPatchSelector()
ones = np.ones((20,20,3))
zeds = np.zeros((20,20,3))
zeds[0][10] = (1,1,1)
zeds[5][13] = (1,1,1)
zeds[17][3] = (1,1,1)


print(list(patches.dynamicBlendVertical(ones,zeds)))