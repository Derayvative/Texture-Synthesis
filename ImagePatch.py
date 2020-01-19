from PIL import Image
import numpy as np

class ImagePatch:

    def __init__(self, topleftcorner: tuple, botrightcorner: tuple, image, frontierSize: int):
        width = botrightcorner[1] - topleftcorner[1] + 1 - 2 * frontierSize
        height = botrightcorner[0] - topleftcorner[0] + 1 - 2 * frontierSize
        self.frontierSize = frontierSize
        self.interiorSize = width
        self.width = width
        self.height = height
        self.topCorn = topleftcorner
        self.imagePatch = np.zeros((height, width, 3))
        self.leftUpFront = np.zeros((frontierSize, frontierSize, 3))
        self.rightUpFront = np.zeros((frontierSize, frontierSize, 3))
        self.leftDownFront = np.zeros((frontierSize, frontierSize, 3))
        self.rightDownFront = np.zeros((frontierSize, frontierSize, 3))
        self.leftFront = np.zeros((height, frontierSize, 3))
        self.rightFront = np.zeros((height, frontierSize, 3))
        self.upFront = np.zeros((frontierSize, width, 3))
        self.downFront = np.zeros((frontierSize, width, 3))
        for i in range(0, height + 2 * frontierSize):
            for j in range(0, width + 2 * frontierSize):
                if i < frontierSize and j < frontierSize:
                    self.leftUpFront[i][j] = image[i + topleftcorner[0]][j + topleftcorner[1]]
                elif i < frontierSize and j < frontierSize + height:
                    self.upFront[i][j - frontierSize] = image[i + topleftcorner[0]][j + topleftcorner[1]]
                elif i < frontierSize:
                    self.rightUpFront[i][j - frontierSize - height] = image[i + topleftcorner[0]][j + topleftcorner[1]]
                elif i < frontierSize + width and j < frontierSize:
                    self.leftFront[i - frontierSize][j] = image[i + topleftcorner[0]][j + topleftcorner[1]]
                elif i < frontierSize + width and j < frontierSize + height:
                    self.imagePatch[i - frontierSize][j - frontierSize] = image[i + topleftcorner[0]][j + topleftcorner[1]]
                elif i < frontierSize + width:
                    self.rightFront[i - frontierSize][j - frontierSize - height] = image[i + topleftcorner[0]][j + topleftcorner[1]]
                elif j < frontierSize:
                    self.leftDownFront[i - frontierSize - width][j] = image[i + topleftcorner[0]][j + topleftcorner[1]]
                elif j < frontierSize + height:
                    self.downFront[i - frontierSize - width][j - frontierSize] = image[i + topleftcorner[0]][j + topleftcorner[1]]
                else:
                    self.rightDownFront[i - frontierSize - width][j - frontierSize - height] = image[i + topleftcorner[0]][j + topleftcorner[1]]


    def __str__(self):
        return str(self.imagePatch)

    def getPatchArray(self):
        return self.imagePatch


    def getLeftPart(self):
        top = np.concatenate((self.leftUpFront, self.upFront), axis=1)
        mid = np.concatenate((self.leftFront, self.imagePatch), axis=1)
        bot = np.concatenate((self.leftDownFront, self.downFront), axis=1)
        whole = np.concatenate((top, mid, bot), axis=0)
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
        top = np.concatenate((self.upFront, self.rightUpFront), axis=1)
        mid = np.concatenate((self.imagePatch, self.rightFront), axis=1)
        bot = np.concatenate((self.downFront, self.rightDownFront), axis=1)
        whole = np.concatenate((top, mid, bot), axis=0)
        return whole

    def getRight(self):
        top = self.rightUpFront
        mid = self.rightFront
        bot = self.rightDownFront
        whole = np.concatenate((top, mid, bot), axis=0)
        return whole

    def getDown(self):
        left = self.leftDownFront
        mid = self.downFront
        right = self.rightDownFront
        whole = np.concatenate((left,mid,right), axis=1)
        return whole

    def getUp(self):
        left = self.leftUpFront
        mid = self.upFront
        right = self.rightUpFront
        whole = np.concatenate((left,mid,right), axis=1)
        return whole

    def getLeft(self):
        top = self.leftUpFront
        mid = self.leftFront
        bot = self.leftDownFront
        whole = np.concatenate((top, mid, bot), axis=0)
        return whole

    def setRight(self, rightArray):
        self.rightUpFront = rightArray[0:self.frontierSize]
        self.rightFront = rightArray[self.frontierSize:self.frontierSize + self.height]
        self.rightDownFront = rightArray[self.frontierSize + self.height:self.frontierSize * 2 + self.height]

    def setDown(self, downArray):
        self.leftDownFront = downArray[:,0:self.frontierSize]
        self.downFront = downArray[:,self.frontierSize:self.frontierSize + self.width]
        self.rightDownFront = downArray[:,self.frontierSize + self.width:self.frontierSize * 2 + self.width]
        print(self.leftDownFront.shape, self.downFront.shape, self.rightDownFront.shape)

    def getFarthestRightPart(self):
        top = np.concatenate((self.rightUpFront), axis=1)
        mid = np.concatenate((self.rightFront), axis=1)
        bot = np.concatenate((self.rightDownFront), axis=1)
        whole = np.concatenate((top, mid, bot), axis=0)
        return whole


    def getBotPart(self):
        mid = np.concatenate((self.leftFront, self.imagePatch, self.rightFront), axis=1)
        bot = np.concatenate((self.leftDownFront, self.downFront, self.rightDownFront), axis=1)
        whole = np.concatenate((mid, bot), axis=0)
        return whole


    def getLeftBotPart(self):
        mid = np.concatenate((self.leftFront, self.imagePatch), axis=1)
        bot = np.concatenate((self.leftDownFront, self.downFront), axis=1)
        whole = np.concatenate((mid, bot), axis=0)
        return whole

    def getRightBotPart(self):
        mid = np.concatenate((self.imagePatch, self.rightFront), axis=1)
        bot = np.concatenate((self.downFront, self.rightDownFront), axis=1)
        whole = np.concatenate((mid, bot), axis=0)
        return whole

    def getEntire(self):
        top = np.concatenate((self.leftUpFront, self.upFront, self.rightUpFront), axis=1)
        mid = np.concatenate((self.leftFront, self.imagePatch, self.rightFront), axis=1)
        bot = np.concatenate((self.leftDownFront, self.downFront, self.rightDownFront), axis=1)
        whole = np.concatenate((top,mid,bot), axis=0)
        return whole

    def __str__(self):
        return str(self.topCorn)