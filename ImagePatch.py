from PIL import Image
import numpy as np

class ImagePatch:

    def __init__(self, topleftcorner: tuple, botrightcorner: tuple, image, frontierSize: int):
        width = botrightcorner[0] - topleftcorner[0] + 1 - 2 * frontierSize
        height = botrightcorner[1] - topleftcorner[1] + 1 - 2 * frontierSize
        self.frontierSize = frontierSize
        self.interiorSize = width
        self.width = width
        self.imagePatch = np.zeros((width, height, 3))
        self.leftUpFront = np.zeros((frontierSize, frontierSize, 3))
        self.rightUpFront = np.zeros((frontierSize, frontierSize, 3))
        self.leftDownFront = np.zeros((frontierSize, frontierSize, 3))
        self.rightDownFront = np.zeros((frontierSize, frontierSize, 3))
        self.leftFront = np.zeros((frontierSize, height, 3))
        self.rightFront = np.zeros((frontierSize, height, 3))
        self.upFront = np.zeros((width, frontierSize, 3))
        self.downFront = np.zeros((width, frontierSize, 3))

        for i in range(0, width + 2 * frontierSize):
            for j in range(0, height + 2 * frontierSize):
                if i < frontierSize and j < frontierSize:
                    self.leftUpFront[i][j] = image[i + topleftcorner[0]][j + topleftcorner[0]]
                elif i < frontierSize and j < frontierSize + height:
                    self.leftFront[i][j - frontierSize] = image[i + topleftcorner[0]][j + topleftcorner[0]]
                elif i < frontierSize:
                    self.leftDownFront[i][j - frontierSize - height] = image[i + topleftcorner[0]][j + topleftcorner[0]]
                elif i < frontierSize + width and j < frontierSize:
                    self.upFront[i - frontierSize][j] = image[i + topleftcorner[0]][j + topleftcorner[0]]
                elif i < frontierSize + width and j < frontierSize + height:
                    self.imagePatch[i - frontierSize][j - frontierSize] = image[i + topleftcorner[0]][j + topleftcorner[0]]
                elif i < frontierSize + width:
                    self.downFront[i - frontierSize][j - frontierSize - height] = image[i + topleftcorner[0]][j + topleftcorner[0]]
                elif j < frontierSize:
                    self.rightUpFront[i - frontierSize - width][j] = image[i + topleftcorner[0]][j + topleftcorner[0]]
                elif j < frontierSize + height:
                    self.rightFront[i - frontierSize - width][j - frontierSize] = image[i + topleftcorner[0]][j + topleftcorner[0]]
                else:
                    self.rightDownFront[i - frontierSize - width][j - frontierSize - height] = image[i + topleftcorner[0]][j + topleftcorner[0]]


    def __str__(self):
        return str(self.imagePatch)

    def getPatchArray(self):
        return self.imagePatch


    def getLeftPart(self):
        top = np.concatenate((self.leftUpFront, self.upFront), axis=0)
        mid = np.concatenate((self.leftFront, self.imagePatch), axis=0)
        bot = np.concatenate((self.leftDownFront, self.downFront), axis=0)
        whole = np.concatenate((top, mid, bot), axis=1)
        print(top.shape, mid.shape, bot.shape)
        return whole

    def getFarthestLeftPart(self):
        top = self.leftUpFront
        mid = self.leftFront
        bot = self.leftDownFront
        print(top.shape,mid.shape,bot.shape)
        whole = np.concatenate((top, mid, bot), axis=0)
        print(whole.shape)
        return whole

    def getRightPart(self):
        top = np.concatenate((self.upFront, self.rightUpFront), axis=0)
        mid = np.concatenate((self.imagePatch, self.rightFront), axis=0)
        bot = np.concatenate((self.downFront, self.rightDownFront), axis=0)
        whole = np.concatenate((top, mid, bot), axis=1)
        return whole

    def getFarthestRightPart(self):
        top = np.concatenate((self.rightUpFront), axis=0)
        mid = np.concatenate((self.rightFront), axis=0)
        bot = np.concatenate((self.rightDownFront), axis=0)
        whole = np.concatenate((top, mid, bot), axis=1)
        return whole


    def getBotPart(self):
        mid = np.concatenate((self.leftFront, self.imagePatch, self.rightFront), axis=0)
        bot = np.concatenate((self.leftDownFront, self.downFront, self.rightDownFront), axis=0)
        whole = np.concatenate((mid, bot), axis=1)
        return whole


    def getLeftBotPart(self):
        mid = np.concatenate((self.leftFront, self.imagePatch), axis=0)
        bot = np.concatenate((self.leftDownFront, self.downFront), axis=0)
        whole = np.concatenate((mid, bot), axis=1)
        return whole

    def getRightBotPart(self):
        mid = np.concatenate((self.imagePatch, self.rightFront), axis=0)
        bot = np.concatenate((self.downFront, self.rightDownFront), axis=0)
        whole = np.concatenate((mid, bot), axis=1)
        return whole

    def getEntire(self):
        top = np.concatenate((self.leftUpFront, self.upFront, self.rightUpFront), axis=0)
        mid = np.concatenate((self.leftFront, self.imagePatch, self.rightFront), axis=0)
        bot = np.concatenate((self.leftDownFront, self.downFront, self.rightDownFront), axis=0)
        whole = np.concatenate((top,mid,bot), axis=1)
        return whole