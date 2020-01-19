from PIL import Image
import numpy as np

from ImagePatch import ImagePatch
from RandomPatchSelector import RandomPatchSelector


def main():
    img = Image.open('honeycomb.jpg')
    ary = np.array(img)
    #ary[y-coordinate][x-coordinate]
    patchsize = 32
    frontierSize = patchsize // 2
    step = 8
    patches = RandomPatchSelector()
    for i in range(0, ary.shape[0] + 1 - patchsize - 2 * frontierSize, step):
        for j in range(0, ary.shape[1] + 1 - patchsize - 2 * frontierSize, step):
            #print(str(i + patchsize - 1), str(j + patchsize - 1))
            patch = ImagePatch((i,j), (i + patchsize + 2 * frontierSize - 1, j + patchsize + 2 * frontierSize - 1), ary, frontierSize)
            #print((i,j),(i + patchsize + 2 * frontierSize - 1, j + patchsize + 2 * frontierSize - 1))
            patches.addPatch(patch)
    product = patches.buildSimpleQuilt(256,256)
    im = Image.fromarray(product.astype(np.uint8))
    im.save('honeycombAfter.jpg')
    # Split the three channels
    #r, g, b = np.split(ary, 3, axis=2)

    '''
    r = r.reshape(-1)
    g = r.reshape(-1)
    b = r.reshape(-1)

    # Standard RGB to grayscale
    bitmap = list(map(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2],
                      zip(r, g, b)))
    bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float), 255)
    im = Image.fromarray(bitmap.astype(np.uint8))
    im.save('pastaAfter.bmp')
    '''

if __name__ == '__main__':
    main()