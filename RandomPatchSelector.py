from PIL import Image
import numpy as np
import random
import math
import copy


from ImagePatch import ImagePatch

#This stores all the patches we wish to consider from the images and can randomly select one if needed
class RandomPatchSelector:



    def __init__(self):
        self.totalWeight = 0
        self.patches = list()
        self.frontierSize = -1
        self.interiorSize = -1
        self.tolerance = 1.15
        self.alpha = 0.75

    #Weight is default 1, I may consider using differing weights in the future, but as of now there is no plans
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

    #Returns a random patch using the weights by recognizing what the total weight is
    def getRandomPatch(self):
        if len(self.patches) == 0:
            return;
        randIndex = random.randint(0, self.totalWeight)
        for i in self.patches:
            randIndex -= i[1]
            if randIndex <= 0:
                return i[0]
        return None

    #Gets an initial square for the application listed in 'PatternPrinter'
    def getMatchingInitialPatchForImprint(self, aryToMatch: np.ndarray):
        bestDiff = float("inf")
        numChecks = 0
        candidates = list()
        for pTup in self.patches:
            p = pTup[0]
            diff = self.__findPatchDifferenceOfDifferingSize(p.getEntire(), aryToMatch)
            if (diff < bestDiff):
                bestDiff = diff
            numChecks += 1
            candidates.append((p, diff))
        bestCands = list()
        for i in candidates:
            if i[1] <= self.tolerance * bestDiff:
                bestCands.append(i[0])
        rand = random.choice(bestCands)
        return rand

    #This is a simple way of generating an image that selects patches without consideration to its surrounding patches. Unlikely to generate
    #pleasing images
    def buildSimpleImage(self, length, height):
        pic = np.full((length,height,3),  -1)
        #imprintPattern simply copies a given patch onto the final bitmap array
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
                imprintPattern(self.getRandomPatch(), (i,j))
        return pic

    #This generates an image by patches, where each patch takes into account surrounding patches, leading to images that
    #visually make sense.
    def buildSimpleQuilt(self, length, height):
        numPatchX = math.ceil((length - self.frontierSize) / (self.interiorSize + self.frontierSize))
        numPatchY = math.ceil((height - self.frontierSize) / (self.interiorSize + self.frontierSize))
        patchworkQuilt = [[None for i in range(numPatchY)] for j in range(numPatchX)]
        for i in range(0, len(patchworkQuilt)):
            for j in range(0, len(patchworkQuilt[i])):
                #This is the initial patch. It is chosen randomly
                if i == 0 and j == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getRandomPatch())
                #Only has left neighboor
                elif i == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingLeftPatch(patchworkQuilt[i][j-1]))
                #Only up neighbor
                elif j == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingUpPatch(patchworkQuilt[i-1][j]))
                #Up and left neighbor
                else:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingLeftUpPatch(patchworkQuilt[i][j-1], patchworkQuilt[i-1][j]))
                #This applies a seam cut and simple blend to help the borders appear less jarring
                if j != 0:
                    patchworkQuilt[i][j-1].setRight(self.dynamicBlendHorizontal(patchworkQuilt[i][j-1], patchworkQuilt[i][j]))
                if i != 0:
                    patchworkQuilt[i-1][j].setDown(self.dynamicBlendVertical(patchworkQuilt[i][j],patchworkQuilt[i-1][j]))
        pic = np.full((length, height, 3), -1)

        # imprintPattern simply copies a given patch onto the final bitmap array
        def imprintPattern(patch: ImagePatch, startCorner):
            if startCorner[0] == 0 and startCorner[1] == 0:
                entire = patch.getEntire()
            elif startCorner[0] == 0:
                entire = patch.getRightPart()
            elif startCorner[1] == 0:
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

        #Builds the quilt by calling the imprintPattern method on all the patches selected earlier
        quiltX = 0
        quiltY = 0
        j = 0
        while j < pic.shape[0]:
            i = 0
            while i < pic.shape[1]:
                imprintPattern(patchworkQuilt[quiltY][quiltX], (j,i))
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


    #This is used in PatternPrinter. It is similar to the method above this, except when selecting what patch to add nexxt
    #it also takes into account how well the patch matches the desired image to be projected. The relative weight of this factor
    #is controlled by the alpha variable
    def buildImprintQuilt(self, length, height, printArray):
        numPatchX = math.ceil((length - self.frontierSize) / (self.interiorSize + self.frontierSize))
        numPatchY = math.ceil((height - self.frontierSize) / (self.interiorSize + self.frontierSize))
        patchworkQuilt = [[None for i in range(numPatchY)] for j in range(numPatchX)]
        y = 0
        for i in range(0, len(patchworkQuilt)):
            x = 0
            for j in range(0, len(patchworkQuilt[i])):
                print('yo', x,y)
                if i == 0 and j == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingInitialPatchForImprint(printArray[y:y + (2 * self.frontierSize + self.interiorSize), x:x + (2 * self.frontierSize + self.interiorSize)]))
                    x += (2 * self.frontierSize + self.interiorSize)
                elif i == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingLeftPatchWithImprint(patchworkQuilt[i][j-1], printArray[y:y + (2 * self.frontierSize + self.interiorSize), x:x + (1 * self.frontierSize + self.interiorSize)]))
                    x += (1 * self.frontierSize + self.interiorSize)
                elif j == 0:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingUpPatchWithImprint(patchworkQuilt[i-1][j], printArray[y:y+(1 * self.frontierSize + self.interiorSize), x:x + (2 * self.frontierSize + self.interiorSize)]))
                    x += (2 * self.frontierSize + self.interiorSize)
                else:
                    patchworkQuilt[i][j] = copy.deepcopy(self.getMatchingLeftUpPatchWithImprint(patchworkQuilt[i][j-1], patchworkQuilt[i-1][j], printArray[y:y + (1 * self.frontierSize + self.interiorSize), x:x + (1 * self.frontierSize + self.interiorSize)]))
                    x += (1 * self.frontierSize + self.interiorSize)
                if j != 0:
                    patchworkQuilt[i][j-1].setRight(self.dynamicBlendHorizontal(patchworkQuilt[i][j-1], patchworkQuilt[i][j]))
                if i != 0:
                    patchworkQuilt[i-1][j].setDown(self.dynamicBlendVertical(patchworkQuilt[i][j],patchworkQuilt[i-1][j]))
            if i == 0:
                y += (2 * self.frontierSize + self.interiorSize)
            else:
                y += 1 * self.frontierSize + self.interiorSize
        pic = np.full((length, height, 3), -1)

        def imprintPattern(patch: ImagePatch, startCorner):
            if startCorner[0] == 0 and startCorner[1] == 0:
                entire = patch.getEntire()
            elif startCorner[0] == 0:
                entire = patch.getRightPart()
            elif startCorner[1] == 0:
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
        while j < pic.shape[0]:
            i = 0
            while i < pic.shape[1]:
                imprintPattern(patchworkQuilt[quiltY][quiltX], (j,i))
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




    #If ary1 and ary2 (which are rgb info) has the same dimensions, this returns the L2 difference between them
    def __findPatchDifference(self, ary1, ary2) -> float:
        if ary1.shape != ary2.shape:
            print(self.frontierSize, self.interiorSize)
            print(ary1.shape, ary2.shape)
            raise Exception
            return float("inf")
        else:
            difference = 0
            for i in range(ary1.shape[0]):
                for j in range(ary1.shape[1]):
                    difference = difference + ((ary1[i][j][0] - ary2[i][j][0])**2 + (ary1[i][j][1] - ary2[i][j][1])**2 + (ary1[i][j][2] - ary2[i][j][2])**2)
            return difference / (3 * 256**2 * ary1.shape[0] * ary1.shape[1])

    #If we want to allow different size patches, this computes the difference by only taking into account pixels that overlap
    def __findPatchDifferenceOfDifferingSize(self, ary1, ary2) -> float:
        difference = 0
        if ary2.shape[0] == 0 or ary2.shape[1] == 0 or ary1.shape[0] == 0 or ary1.shape[1] == 0:
            return 0
        for i in range(min(ary2.shape[0],ary1.shape[0])):
            for j in range(min(ary2.shape[1],ary1.shape[1])):
                    difference = difference + ((ary1[i][j][0] - ary2[i][j][0])**2 + (ary1[i][j][1] - ary2[i][j][1])**2 + (ary1[i][j][2] - ary2[i][j][2])**2)
        return difference / (3 * 256**2 * (min(ary2.shape[0],ary1.shape[0])) * (min(ary2.shape[1],ary1.shape[1])))

    #Gets an image patch that matches by finding an image which has a similar left periphery to the previous images right periphery
    def getMatchingLeftPatch(self, patch: ImagePatch) -> ImagePatch:
        bestDiff = float("inf")
        numChecks = 0
        random.shuffle(self.patches)
        candidates = list()
        for pTup in self.patches:
            p = pTup[0]
            diff = self.__findPatchDifference(patch.rightFront, p.leftFront) + self.__findPatchDifference(patch.rightDownFront, p.leftDownFront) \
                + self.__findPatchDifference(patch.rightUpFront, p.leftUpFront)
            if (diff < bestDiff):
                bestDiff = diff
            numChecks += 1
            candidates.append((p, diff))
        bestCands = list()
        for i in candidates:
            if i[1] <= self.tolerance * bestDiff:
                bestCands.append(i[0])
        rand = random.choice(bestCands)
        return rand

    #Similar to the above method, but takes into account similarity to a specified area of a provided image
    def getMatchingLeftPatchWithImprint(self, patch: ImagePatch, aryToMatch: np.ndarray) -> ImagePatch:
        bestDiff = float("inf")
        numChecks = 0
        random.shuffle(self.patches)
        candidates = list()
        for pTup in self.patches:
            p = pTup[0]
            diff = self.alpha*(self.__findPatchDifference(patch.rightFront, p.leftFront) + self.__findPatchDifference(patch.rightDownFront, p.leftDownFront) \
                + self.__findPatchDifference(patch.rightUpFront, p.leftUpFront)) + (1-self.alpha) * 3 * self.__findPatchDifferenceOfDifferingSize(p.getRightPart(), aryToMatch)
            if (diff < bestDiff):
                bestDiff = diff
            numChecks += 1
            candidates.append((p, diff))
        bestCands = list()
        for i in candidates:
            if i[1] <= self.tolerance * bestDiff:
                bestCands.append(i[0])
        rand = random.choice(bestCands)
        return rand

    # Gets an image patch that matches by finding an image which has a similar up periphery to the previous images down periphery
    def getMatchingUpPatch(self, patch: ImagePatch) -> ImagePatch:
        bestDiff = float("inf")
        numChecks = 0
        random.shuffle(self.patches)
        candidates = list()
        for pTup in self.patches:
            p = pTup[0]
            diff = self.__findPatchDifference(patch.downFront, p.upFront) + self.__findPatchDifference(patch.rightDownFront, p.rightUpFront) \
                + self.__findPatchDifference(patch.leftDownFront, p.leftUpFront)
            if (diff < bestDiff):
                bestDiff = diff
            numChecks += 1
            candidates.append((p, diff))
        bestCands = list()
        for i in candidates:
            if i[1] <= self.tolerance * bestDiff:
                bestCands.append(i[0])
        return random.choice(bestCands)

    # Similar to the above method, but takes into account similarity to a specified area of a provided image
    def getMatchingUpPatchWithImprint(self, patch: ImagePatch, aryToMatch: np.ndarray) -> ImagePatch:
        bestDiff = float("inf")
        numChecks = 0
        random.shuffle(self.patches)
        candidates = list()
        for pTup in self.patches:
            p = pTup[0]
            diff = self.alpha * (self.__findPatchDifference(patch.downFront, p.upFront) + self.__findPatchDifference(
                patch.rightDownFront, p.rightUpFront) \
                   + self.__findPatchDifference(patch.leftDownFront, p.leftUpFront)) + (1 - self.alpha) * 3 * self.__findPatchDifferenceOfDifferingSize(p.getLowerPart(), aryToMatch)
            if (diff < bestDiff):
                bestDiff = diff
            numChecks += 1
            candidates.append((p, diff))
        bestCands = list()
        for i in candidates:
            if i[1] <= self.tolerance * bestDiff:
                bestCands.append(i[0])
        return random.choice(bestCands)

    # Gets an image patch that matches by finding an image which has a similar up periphery to the previous images down periphery and left periphery to right periphery
    def getMatchingLeftUpPatch(self, patch, patchU):
        bestDiff = float("inf")
        numChecks = 0
        random.shuffle(self.patches)
        candidates = list()
        for pTup in self.patches:
            p = pTup[0]
            diff = (self.__findPatchDifference(patch.rightFront, p.leftFront) + self.__findPatchDifference(
                patch.rightDownFront, p.leftDownFront) \
                   + self.__findPatchDifference(patch.rightUpFront, p.leftUpFront) + self.__findPatchDifference(patchU.downFront, p.upFront) \
                + self.__findPatchDifference(patchU.rightDownFront, p.rightUpFront)) + self.__findPatchDifference(patchU.leftDownFront, p.leftUpFront)
            if (diff < bestDiff):
                bestDiff = diff
            numChecks += 1
            candidates.append((p, diff))
        bestCands = list()
        for i in candidates:
            if i[1] <= self.tolerance * bestDiff:
                bestCands.append(i[0])
        print(len(bestCands))
        return random.choice(bestCands)

    # Similar to the above method, but takes into account similarity to a specified area of a provided image
    def getMatchingLeftUpPatchWithImprint(self, patch, patchU, aryToMatch: np.ndarray):
        bestDiff = float("inf")
        numChecks = 0
        random.shuffle(self.patches)
        candidates = list()
        for pTup in self.patches:
            p = pTup[0]
            diff = self.alpha * (self.__findPatchDifference(patch.rightFront, p.leftFront) + self.__findPatchDifference(
                patch.rightDownFront, p.leftDownFront) \
                   + self.__findPatchDifference(patch.rightUpFront, p.leftUpFront) + self.__findPatchDifference(
                patchU.downFront, p.upFront) \
                   + self.__findPatchDifference(patchU.rightDownFront, p.rightUpFront) + self.__findPatchDifference(
                patchU.leftDownFront, p.leftUpFront)) + (1 - self.alpha) * 5 * self.__findPatchDifferenceOfDifferingSize(p.getLowerPart(), aryToMatch)
            if (diff < bestDiff):
                bestDiff = diff
            numChecks += 1
            candidates.append((p, diff))
        bestCands = list()
        for i in candidates:
            if i[1] <= self.tolerance * bestDiff:
                bestCands.append(i[0])
        print(len(bestCands))
        return random.choice(bestCands)


    def getMatchingRightPatch(self, patch: ImagePatch) -> ImagePatch:
        bestDiff = float("inf")
        bestMatchingPatch = patch
        for p in self.patches:
            diff = self.__findPatchDifference(patch.leftFront, p.rightFront)
            if (diff < bestDiff):
                bestDiff = diff
                bestMatchingPatch = p
        return bestMatchingPatch

    #Finds the optimal division between 2 images to produce a least jarring border using dynamic programming to find a min path
    def dynamicBlendHorizontal(self, rightPatch: ImagePatch, leftPatch: ImagePatch):
        left = leftPatch.getLeft()
        right = rightPatch.getRight()
        differenceMatrix = np.zeros((left.shape[0], left.shape[1]))
        for i in range(left.shape[0]):
            for j in range(left.shape[1]):
                differenceMatrix[i][j] = ((left[i][j][0]-right[i][j][0])**2 + (left[i][j][1]-right[i][j][1])**2 + (left[i][j][2]-right[i][j][2])**2)
        def getBestVerticalDivisionOfDifference():
            pathMatrix = np.zeros(differenceMatrix.shape)
            for y in range(pathMatrix.shape[0]):
                pathMatrix[y][0] = differenceMatrix[y][0]
            for y in range(1,pathMatrix.shape[0]):
                for x in range(0, pathMatrix.shape[1]):
                    possibleVals = list()
                    if x != 0:
                        possibleVals.append(pathMatrix[y-1][x-1])
                    if x != pathMatrix.shape[1] - 1:
                        possibleVals.append(pathMatrix[y - 1][x + 1])
                    possibleVals.append(pathMatrix[y-1][x])
                    pathMatrix[y][x] = differenceMatrix[y][x] + min(possibleVals)
            return pathMatrix
        pathMatrix = getBestVerticalDivisionOfDifference()
        path = list()
        minDiff = float("inf")
        minIndex = -1
        for x in range(pathMatrix.shape[1]):
            if pathMatrix[pathMatrix.shape[0] - 1][x] < minDiff:
                minDiff = pathMatrix[pathMatrix.shape[0] - 1][x]
                minIndex = x
        path.append(minIndex)
        for y in range(pathMatrix.shape[0] - 2, -1, -1):
            adj = list()
            end = path[len(path) - 1]
            if end != 0:
                adj.append(pathMatrix[y][end - 1])
            else:
                adj.append(float("inf"))
            adj.append(pathMatrix[y][end])
            if end != len(pathMatrix[y]) - 1:
                adj.append(pathMatrix[y][end + 1])
            else:
                adj.append(float("inf"))
            path.append(adj.index(min(adj)) - 1 + end)
        path.reverse()
        result = np.zeros((left.shape[0], left.shape[1],3))
        for y in range(result.shape[0]):
            maxDist = max((path[y] - 0, result.shape[1] - path[y]))
            for x in range(result.shape[1]):
                coeff = abs(x - path[y]) / (2*maxDist) + 0.5
                if x <= path[y]:
                    result[y][x] = right[y][x] * coeff + left[y][x] * (1-coeff)
                else:
                    result[y][x] = left[y][x] * coeff + right[y][x] * (1-coeff)
        return result

    # Finds the optimal division between 2 images to produce a least jarring border using dynamic programming to find a min path
    def dynamicBlendVertical(self, topPatch: ImagePatch, botPatch: ImagePatch):
        top = topPatch.getUp()
        bot = botPatch.getDown()
        differenceMatrix = np.zeros((top.shape[0], top.shape[1]))
        for i in range(top.shape[0]):
            for j in range(top.shape[1]):
                differenceMatrix[i][j] = ((top[i][j][0]-bot[i][j][0])**2 + (top[i][j][1]-bot[i][j][1])**2 + (top[i][j][2]-bot[i][j][2])**2)
        def getBestHorizontalDivisionOfDifference():
            pathMatrix = np.zeros(differenceMatrix.shape)
            for x in range(pathMatrix.shape[1]):
                pathMatrix[0][x] = differenceMatrix[0][x]
            for x in range(1,pathMatrix.shape[1]):
                for y in range(0, pathMatrix.shape[0]):
                    possibleVals = list()
                    if y != 0:
                        possibleVals.append(pathMatrix[y-1][x-1])
                    if y != pathMatrix.shape[0] - 1:
                        possibleVals.append(pathMatrix[y + 1][x - 1])
                    possibleVals.append(pathMatrix[y][x-1])
                    pathMatrix[y][x] = differenceMatrix[y][x] + min(possibleVals)
            return pathMatrix
        pathMatrix = getBestHorizontalDivisionOfDifference()
        path = list()
        minDiff = float("inf")
        minIndex = -1
        for y in range(pathMatrix.shape[0]):
            if pathMatrix[y][pathMatrix.shape[1] - 1] < minDiff:
                minDiff = pathMatrix[pathMatrix.shape[0] - 1][y]
                minIndex = y
        path.append(minIndex)
        for x in range(pathMatrix.shape[1] - 2, -1, -1):
            print(path)
            adj = list()
            end = path[len(path) - 1]
            if end != 0:
                adj.append(pathMatrix[end - 1][x])
            else:
                adj.append(float("inf"))
            adj.append(pathMatrix[end][x])
            if end != len(pathMatrix) - 1:
                adj.append(pathMatrix[end + 1][x])
            else:
                adj.append(float("inf"))
            path.append(adj.index(min(adj)) - 1 + end)
        path.reverse()
        result = np.zeros((top.shape[0], top.shape[1],3))
        for x in range(result.shape[1]):
            maxDist = max((path[x] - 0, result.shape[0] - path[x]))
            for y in range(result.shape[0]):
                coeff = abs(y - path[x]) / (2 * maxDist) + 0.5
                if y <= path[x]:
                    result[y][x] = bot[y][x] * coeff + top[y][x] * (1 - coeff)
                else:
                    result[y][x] = top[y][x] * coeff + bot[y][x] * (1-coeff)
        print("path", path)
        return result

