from PIL import Image
import numpy as np
import random
import math
import copy


from ImagePatch import ImagePatch


class RandomPatchSelector:

    def __init__(self):
        self.totalWeight = 0
        self.patches = list()
        self.frontierSize = -1
        self.interiorSize = -1

    def addPatch(self, patch:ImagePatch, weight=1):
        if weight <= 0:
            return
        self.patches.append((patch, weight))
        self.totalWeight += weight
        if self.frontierSize == -1 or self.interiorSize == -1:
            self.frontierSize = patch.frontierSize
            self.interiorSize = patch.width
        elif self.frontierSize != patch.frontierSize or self.interiorSize != patch.width:
            raise ValueError
        self.patchSize = patch.getPatchArray().shape[0]
        self.patchFrontierSize = self.patchSize // 6

    def getRandomPatch(self):
        if len(self.patches) == 0:
            return;
        randIndex = random.randint(0, self.totalWeight)
        for i in self.patches:
            randIndex -= i[1]
            if randIndex <= 0:
                return i[0]
        return None

    def buildSimpleImage(self, length, height):
        pic = np.full((length,height,3),  -1)
        def imprintPattern(patch: ImagePatch, startCorner):
            for i in range(patch.getPatchArray().shape[0]):
                for j in range(patch.getPatchArray().shape[1]):
                    if i + startCorner[0] < 0 or i + startCorner[0] >= pic.shape[0] or j + startCorner[1] < 0 or j + startCorner[1] >= pic.shape[1]:
                        continue
                    location = pic[i + startCorner[0]][j + startCorner[1]]
                    if location[0] == -1:
                        pic[i + startCorner[0]][j + startCorner[1]] = patch.getPatchArray()[i][j]
        for i in range(0, pic.shape[0], self.patchSize):
            for j in range(0, pic.shape[1], self.patchSize):
                print('yello' + str(i) + " " + str(j))
                imprintPattern(self.getRandomPatch(), (i,j))
        return pic

    def buildSimpleQuilt(self, length, height):
        print("TO " + str(self.frontierSize) + " " + str(self.interiorSize))
        numPatchX = math.ceil((length - self.frontierSize) / (self.interiorSize + self.frontierSize))
        numPatchY = math.ceil((height - self.frontierSize) / (self.interiorSize + self.frontierSize))
        patchworkQuilt = [[None for i in range(numPatchY)] for j in range(numPatchX)]
        for i in range(0, len(patchworkQuilt)):
            for j in range(0, len(patchworkQuilt[i])):
                print(str(i) + " " + str(j))
                if i == 0 and j == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getRandomPatch())
                    print(patchworkQuilt[i][j])
                elif j == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingLeftPatch(patchworkQuilt[i-1][j]))
                elif i == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingUpPatch(patchworkQuilt[i][j-1]))
                    res = self.dynamicBlendHorizontal(patchworkQuilt[i][j].upFront, patchworkQuilt[i][j - 1].downFront)
                    patchworkQuilt[i][j-1].downFront = res
                else:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingLeftUpPatch(patchworkQuilt[i - 1][j], patchworkQuilt[i][j-1]))
                    res = self.dynamicBlendHorizontal(patchworkQuilt[i][j].upFront, patchworkQuilt[i][j - 1].downFront)
                    patchworkQuilt[i][j - 1].downFront = res
        pic = np.full((length, height, 3), -1)

        def imprintPattern(patch: ImagePatch, startCorner):
            if startCorner[0] == 0 and startCorner[1] == 0:
                entire = patch.getEntire()
            elif startCorner[1] == 0:
                entire = patch.getRightPart()
            elif startCorner[0] == 0:
                entire = patch.getBotPart()
            else:
                entire = patch.getRightBotPart()
            print("start", startCorner, entire.shape)
            for i in range(entire.shape[0]):
                for j in range(entire.shape[1]):
                    if i + startCorner[0] < 0 or i + startCorner[0] >= pic.shape[0] or j + startCorner[1] < 0 or j + \
                            startCorner[1] >= pic.shape[1]:
                        continue
                    #location = pic[i + startCorner[0]][j + startCorner[1]]
                    #if location[0] == -1:
                    pic[i + startCorner[0]][j + startCorner[1]] = entire[i][j]

        quiltX = 0
        quiltY = 0
        j = 0
        while j < pic.shape[1]:
            i = 0
            while i < pic.shape[0]:
                imprintPattern(patchworkQuilt[quiltX][quiltY], (i, j))
                if i == 0:
                    i += self.frontierSize * 2 + self.interiorSize
                else:
                    i += self.frontierSize * 1 + self.interiorSize
                quiltX += 1
            if j == 0:
                j += self.frontierSize * 2 + self.interiorSize
            else:
                j += self.frontierSize * 1 + self.interiorSize
            quiltX = 0
            quiltY += 1
        return pic




    def __findPatchDifference(self, ary1, ary2) -> float:
        if ary1.shape != ary2.shape:
            return float("inf")
        else:
            difference = 0
            for i in range(ary1.shape[0]):
                for j in range(ary1.shape[1]):
                    difference = difference + math.sqrt((ary1[i][j][0] - ary2[i][j][0])**2 + (ary1[i][j][1] - ary2[i][j][1])**2 + (ary1[i][j][2] - ary2[i][j][2])**2)
            return difference

    def getMatchingLeftPatch(self, patch: ImagePatch) -> ImagePatch:
        bestDiff = float("inf")
        bestMatchingPatch = patch
        numChecks = 0
        random.shuffle(self.patches)
        for pTup in self.patches:
            #if numChecks > 150:
            #    return bestMatchingPatch
            p = pTup[0]
            diff = self.__findPatchDifference(patch.rightFront, p.leftFront) + self.__findPatchDifference(patch.rightDownFront, p.leftDownFront) \
                + self.__findPatchDifference(patch.rightUpFront, p.leftUpFront)
            if (diff < bestDiff):
                bestDiff = diff
                bestMatchingPatch = p
            numChecks += 1
        return bestMatchingPatch

    def getMatchingUpPatch(self, patch: ImagePatch) -> ImagePatch:
        bestDiff = float("inf")
        bestMatchingPatch = patch
        numChecks = 0
        random.shuffle(self.patches)
        for pTup in self.patches:
            if numChecks > 500:
                return bestMatchingPatch
            p = pTup[0]
            diff = self.__findPatchDifference(patch.downFront, p.upFront) + self.__findPatchDifference(patch.rightDownFront, p.rightUpFront) \
                + self.__findPatchDifference(patch.leftDownFront, p.leftUpFront)
            if (diff < bestDiff):
                bestDiff = diff
                bestMatchingPatch = p
            numChecks += 1
        return bestMatchingPatch

    def getMatchingLeftUpPatch(self, patch, patchU):
        bestDiff = float("inf")
        bestMatchingPatch = patch
        numChecks = 0
        random.shuffle(self.patches)
        for pTup in self.patches:
            #if numChecks > 500:
            #    return bestMatchingPatch
            p = pTup[0]
            diff = self.__findPatchDifference(patch.rightFront, p.leftFront) + self.__findPatchDifference(
                patch.rightDownFront, p.leftDownFront) \
                   + self.__findPatchDifference(patch.rightUpFront, p.leftUpFront) + self.__findPatchDifference(patchU.downFront, p.upFront) \
                + self.__findPatchDifference(patchU.rightDownFront, p.rightUpFront) + self.__findPatchDifference(patchU.leftDownFront, p.leftUpFront)
            if (diff < bestDiff):
                bestDiff = diff
                bestMatchingPatch = p
            numChecks += 1
        return bestMatchingPatch

    def getMatchingRightPatch(self, patch: ImagePatch) -> ImagePatch:
        bestDiff = float("inf")
        bestMatchingPatch = patch
        for p in self.patches:
            diff = self.__findPatchDifference(patch.leftFront, p.rightFront)
            if (diff < bestDiff):
                bestDiff = diff
                bestMatchingPatch = p
        return bestMatchingPatch

    def dynamicBlendHorizontal(self, rightPatch: ImagePatch, leftPatch: ImagePatch):
        left = leftPatch
        right = rightPatch
        print(left.shape,right.shape)
        differenceMatrix = np.zeros((left.shape[0], left.shape[1]))
        for i in range(left.shape[0]):
            for j in range(left.shape[1]):
                differenceMatrix[i][j] = math.sqrt((left[i][j][0]-right[i][j][0])**2 + (left[i][j][1]-right[i][j][1])**2 + (left[i][j][2]-right[i][j][2])**2)
        def getBestVerticalDivisionOfDifference():
            pathMatrix = np.zeros(differenceMatrix.shape)
            for j in range(pathMatrix.shape[1]):
                pathMatrix[0][j] = differenceMatrix[0][j]
            for i in range(1,pathMatrix.shape[0]):
                for j in range(0, pathMatrix.shape[1]):
                    possibleVals = list()
                    if j != 0:
                        possibleVals.append(pathMatrix[i-1][j-1])
                    if j != pathMatrix.shape[1] - 1:
                        possibleVals.append(pathMatrix[i - 1][j + 1])
                    possibleVals.append(pathMatrix[i-1][j])
                    pathMatrix[i][j] = differenceMatrix[i][j] + min(possibleVals)
            return pathMatrix
        pathMatrix = getBestVerticalDivisionOfDifference()
        path = list()
        minDiff = float("inf")
        minIndex = -1
        for j in range(pathMatrix.shape[1]):
            if pathMatrix[pathMatrix.shape[0] - 1][j] < minDiff:
                minDiff = pathMatrix[pathMatrix.shape[0] - 1][j]
                minIndex = j
        path.append(minIndex)
        for i in range(pathMatrix.shape[0] - 2, -1, -1):
            print(path)
            adj = list()
            end = path[len(path) - 1]
            if end != 0:
                adj.append(pathMatrix[i][end - 1])
            else:
                adj.append(float("inf"))
            adj.append(pathMatrix[i][end])
            if end != len(pathMatrix[i]) - 1:
                adj.append(pathMatrix[i][end + 1])
            else:
                adj.append(float("inf"))
            path.append(adj.index(min(adj)) - 1 + end)
        path.reverse()
        result = np.zeros((left.shape[0], left.shape[1],3))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if j <= path[i]:
                    result[i][j] = left[i][j]
                else:
                    result[i][j] = right[i][j]
        print("path", path)
        return result

