#### IMPORTS ####

import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt

#### GLOBAL VARIABLES ####

SpectrumPath = "spectrums\\"    # Path of spectrum file(s) to read
SpectrumLength = 12566          # Expected size of all spectrums
SpectralChannelWidth = 20       # (nm) Width of each spectral channel
SmoothMatching = False          # Enables smooth matching of spectral channel power calculations (False = calculated as average over wavelength interval | True = calculated as middle point after Guassian smoothing)
BaseUnits = True                # Whether the read spectrums will be in the base units (X: cm^(-1) | Y: mW)
GenerateGraphs = True           # Whether plot PNG files are generated

VoltageStart = 0                # Starting voltage point for read spectrums
VoltageStep = 0.25              # Voltage steps for read spectrums
PolyDegree = 6                  # Degree of generated fitted lookup polynomial

FlattenReach = 0.001            # Reach percentage argument of Gaussian smoothing function
FlattenSigma = 30               # Sigma argument of Gaussian smoothing function
FlattenPhi = 0                  # Phi (phase) argument of Gaussian smoothing function
FlattenAmplitude = 1            # Amplitude argument of Gaussian smoothing function

#### MATH FUNCTIONS ####

def FitGaussiansToData(xData, yData, sigmas, phis, xStart, xEnd, iterations, amplitudes, xAxisStart, dataLabels, dataFlattening, dataSkips, dataFits, calculateSum, reportNum, reportSavePath, reportFileName):
    datasNum = len(xData)
    retMus = []
    idx = 0

    while (idx < datasNum):
        idx += 1

        if ((len(dataFits) > 0) and (dataFits[(idx - 1)] == 0)):
            continue

        currIdx = (idx - 1)

        currXData = xData[currIdx]
        currYData = yData[currIdx]
        currFlatteningData = None
        currDataSkip = None

        if (len(dataFlattening) > 0):
            currFlatteningData = dataFlattening[currIdx]
        if (len(dataSkips) > 0):
            currDataSkip = dataSkips[currIdx]

        mus = []

        gaussianData = GenerateGaussianData(sigmas, 
                                            phis, 
                                            xStart, 
                                            xEnd, 
                                            iterations, 
                                            amplitudes, 
                                            xAxisStart, 
                                            currXData, 
                                            currYData, 
                                            mus, 
                                            calculateSum)

        retMus.append(mus)

        gaussianXData = gaussianData[0]
        gaussianYData = gaussianData[1]

        for j in range(len(gaussianXData)):
            xData.insert(j, gaussianXData[j])
            yData.insert(j, gaussianYData[j])

            if (len(dataFlattening) > 0):
                dataFlattening.insert(j, [-1])
            if (len(dataSkips) > 0):
                dataSkips.insert(j, 1)

            if ((calculateSum == True) and (j == (len(gaussianXData) - 1))):
                dataLabels.insert(j, "Gaussian Sum")
            else:
                dataLabels.insert(j, "Gaussian #" + str(j) + " (a = " + str(round(amplitudes[j], 3)) + ")")

    return retMus

def GenerateGaussianData(sigmas, phis, xStart, xEnd, iterations, amplitudes, xAxisStart, xPoints, yFitPoints, muList, calculateSum):
    mus = [0]
    peaks = len(sigmas)

    for i in range(peaks - 1):
        currSigma = sigmas[i + 1]
        currPhi = phis[i + 1]
        currAmplitude = amplitudes[i + 1]

        halfY = GetGaussianValueY(0, 0, currSigma, currPhi, currAmplitude) / 2
        halfX = GetGaussianValueX(halfY, 0, currSigma, currPhi, currAmplitude)[0]

        mus.append((halfX * 2) + mus[i])

    muOffset = (xAxisStart - mus[0])

    for i in range(len(mus)):
        mus[i] += muOffset
        muList.append(mus[i])

    if (len(yFitPoints) > 0):
        for i in range(len(amplitudes)):
            amplitudes[i] = yFitPoints[FindInArray(xPoints, GetClosestNum(xPoints, mus[i]))[0]]

    xCurr = xStart
    xInc = None
    loopRuns = iterations

    if (len(xPoints) == 0):
        xInc = ((xEnd - xStart) / iterations)
    else:
        loopRuns = len(xPoints)

    xData = []
    yData = []

    for i in range(peaks):
        xData.append(np.empty(shape = loopRuns, dtype = float))
        yData.append(np.empty(shape = loopRuns, dtype = float))

    if (calculateSum == True):
        xData.append(np.empty(shape = loopRuns, dtype = float))
        yData.append(np.empty(shape = loopRuns, dtype = float))

    for i in range(loopRuns):
        if (xInc == None):
            xCurr = xPoints[i]
        else:
            xCurr += xInc
        
        yCurrSum = 0

        for j in range(peaks):
            yCurr = GetGaussianValueY(xCurr, mus[j], sigmas[j], phis[j], 1)
            yCurr *= amplitudes[j]

            xData[j][i] = xCurr
            yData[j][i] = yCurr

            yCurrSum += yCurr

        if (calculateSum == True):
            xData[peaks][i] = xCurr
            yData[peaks][i] = yCurrSum

    return [xData, yData]

def GaussianSmoothData(data, limits):
    limitIndices = [FindInArray(data[0], GetClosestNum(data[0], limits[0]))[0], FindInArray(data[0], GetClosestNum(data[0], limits[1]))[0]]

    if (limitIndices[1] - limitIndices[0] == 0):
        print("WARNING: Smoothing no data, check the flattening limits!")

        return data

    xData = data[0][limitIndices[0] : limitIndices[1]]
    yData = data[1][limitIndices[0] : limitIndices[1]]
    flattenedYData = np.copy(data[1])

    for i in range(len(xData)):
        gaussianCenter = xData[i]
        baseReachEdges = GetGaussianValueX((GetGaussianValueY(0, 0, FlattenSigma, FlattenPhi, FlattenAmplitude) * FlattenReach), 0, FlattenSigma, FlattenPhi, FlattenAmplitude)
        reachEdgeIndices = [FindInArray(xData, GetClosestNum(xData, (gaussianCenter + baseReachEdges[0])))[0], 
                            FindInArray(xData, GetClosestNum(xData, (gaussianCenter + baseReachEdges[1])))[0]]
        currDataScanNum = reachEdgeIndices[0] - reachEdgeIndices[1]

        currYPoints = yData[reachEdgeIndices[1] : reachEdgeIndices[1] + currDataScanNum]
        kernel = GetGaussianValueY(np.arange(currDataScanNum), (i - reachEdgeIndices[1]), FlattenSigma, FlattenPhi, FlattenAmplitude)

        flattenedYData[i + limitIndices[0]] = np.sum(currYPoints * (kernel / np.sum(kernel)))

    return [data[0], flattenedYData]

def GetGaussianValueX(y, mu, sigma, phi, amplitude):
    x = ((sigma * np.sqrt(-2 * np.log(y / (amplitude * np.cos(phi))))) + mu)

    return [x, (mu - (x - mu))]

def GetGaussianValueY(x, mu, sigma, phi, amplitude):
    y = ((amplitude * np.cos(phi)) * np.exp(-np.power(((x - mu) / sigma), 2) / 2))

    return y

def FitPoly(xData, yData, degree):
    return np.polyfit(xData, yData, degree)

def GetClosestNum(array, base):
    array = np.asarray(array)

    return array[(np.abs(array - base)).argmin()]

def DiffSpectrums(spectrumData):
    diffData = {}

    for channel in spectrumData:
        diffData[channel] = {}
        baseSpectrum = None

        if 0 not in spectrumData[channel]:
            print("WARNING: Unable to find 0V spectrum in channel " + str(channel) + "!")

            voltages = []

            for voltage in spectrumData[channel]:
                voltages.append(voltage)

            baseSpectrum = spectrumData[channel][min(voltages)]

        else:
            baseSpectrum = spectrumData[channel][0]

        for voltage in spectrumData[channel]:
            #if (voltage == 0):
            #    continue

            diffData[channel][voltage] = [spectrumData[channel][voltage][0], spectrumData[channel][voltage][1] - baseSpectrum[1]]

            print("Diffed data of channel " + str(channel) + " of voltage " + str(voltage) + ".")

    return diffData

#### LOOKUP FUNCTIONS ####

def GenerateLookups(diffData):
    for channel in diffData:
        peakFindDatasX = []
        peakFindDatasY = []
        mus = None

        for voltage in diffData[channel]:
            peakFindDatasX.append([diffData[channel][voltage][0]])
            peakFindDatasY.append([diffData[channel][voltage][1]])

            gaussianSigmas = []
            gaussianPhis = []
            gaussianAmplitudes = []

            for j in range(20):
                gaussianSigmas.append(8.494)
                gaussianPhis.append(0)
                gaussianAmplitudes.append(1)

            mus = FitGaussiansToData(peakFindDatasX[len(peakFindDatasX) - 1], 
                                     peakFindDatasY[len(peakFindDatasY) - 1], 
                                     gaussianSigmas, 
                                     gaussianPhis, 
                                     1300, 1900, 
                                     0, 
                                     gaussianAmplitudes, 
                                     1410, 
                                     [], 
                                     [], [], [], 
                                     False,
                                     0, 
                                     "", "")

            del peakFindDatasX[len(peakFindDatasX) - 1][0]

        mus = mus[0]
        amplitudeTitles = []
        peakFindDataLabels = []

        for j in range(len(mus)):
            amplitudeTitles.append("Gaussian Position #" + str(j) + " (" + str(round(mus[j])) + ")")
            peakFindDataLabels.append(["Gaussian #" + str(j) + " (mu = " + str(round(mus[j])) + ")"])

        amplitudeDataX = []
        amplitudeDataY = []
        numVoltages = len(peakFindDatasX)
        iterations = None

        for l in range(len(mus)):
            currAmplitudeDataX = []
            currAmplitudeDataY = []

            voltage = VoltageStart
            voltage += VoltageStep

            for j in range(numVoltages):
                for k in range(len(peakFindDatasX[j])):
                    if (k != l):
                        continue

                    currAmplitudeDataX.append(voltage)

                    if (SmoothMatching == False):
                        currAmplitudeDataY.append(peakFindDatasY[j][k][FindInArray(peakFindDatasX[j][k], GetClosestNum(peakFindDatasX[j][k], mus[l]))[0]])

                    else:
                        channelEdgesX = [GetClosestNum(peakFindDatasX[j][k], mus[l] - (SpectralChannelWidth / 2)), GetClosestNum(peakFindDatasX[j][k], mus[l] + (SpectralChannelWidth / 2))]
                        channelEdgesIndices = [FindInArray(peakFindDatasX[j][k], channelEdgesX[0])[0], FindInArray(peakFindDatasX[j][k], channelEdgesX[1])[0]]
                        channelData = peakFindDatasY[j][k][channelEdgesIndices[0] : channelEdgesIndices[1]]
                        currAmplitudeDataY.append(sum(channelData) / len(channelData))

                voltage += VoltageStep

            iterations = len(currAmplitudeDataX) / numVoltages

            if (iterations.is_integer() == False):
                raise ValueError("Number of iterations isn't an integer!")

            iterations = int(iterations)
            idx = 0

            while (idx < iterations):
                insAmplitudeDataX = []
                insAmplitudeDataY = []

                for j in range(numVoltages):
                    insAmplitudeDataX.append(currAmplitudeDataX[idx + (j * iterations)])
                    insAmplitudeDataY.append(currAmplitudeDataY[idx + (j * iterations)])

                amplitudeDataX.append(insAmplitudeDataX)
                amplitudeDataY.append(insAmplitudeDataY)

                idx += 1

        idx = 0
        graphIdx = 0

        lookupDataX = []
        lookupDataY = []
        lookupLabels = []

        while (idx < len(amplitudeDataX)):
            currAmplitudeGraphDataX = []
            currAmplitudeGraphDataY = []

            for j in range(iterations):
                currAmplitudeGraphDataX.append(amplitudeDataX[idx + j])
                currAmplitudeGraphDataY.append(amplitudeDataY[idx + j])

            lookupDataX.append(currAmplitudeGraphDataX[0])
            lookupDataY.append(currAmplitudeGraphDataY[0])
            lookupLabels.append(peakFindDataLabels[graphIdx][0])

            idx += iterations
            graphIdx += 1

        GenerateLookupTable(lookupDataX, lookupDataY, lookupLabels, str(channel), "lookups\\", "lookup_" + str(channel) + ".txt")

        print("Generated lookup for channel " + str(channel) + ".")

def GenerateLookupTable(xData, yData, dataLabels, channelName, filePath, fileName):
    polys = []
    data = "LOOKUP " + channelName + "\n\n"

    for i in range(len(xData)):
        label = dataLabels[i]
        nameIdx = label.find("mu = ")

        if (nameIdx != -1):
            nameIdx += 5
            endIdx = nameIdx

            while (IsStringNum(label[endIdx], 0) == True):
                endIdx += 1

            label = label[nameIdx : endIdx]

        data += "LABEL " + label + "\n"

        for j in range(len(xData[i])):
            data += "DATA (" + str(xData[i][j]) + ", " + str(yData[i][j]) + ")\n"

        data += "POLY ("

        polyCoeffs = FitPoly(xData[i], yData[i], PolyDegree)

        for j in range(len(polyCoeffs)):
            data += str(polyCoeffs[j])

            if (j != (len(polyCoeffs) - 1)):
                data += ", "

        data += ")\n\n"
        polys.append(polyCoeffs)
        
    if not os.path.exists(filePath):
        os.mkdir(filePath)

    file = open(filePath + fileName, "w")

    file.write(data)

    file.close()

    if (GenerateGraphs == True):
        plots = 0

        if not os.path.exists("lookups\\plots_lookup\\"):
            os.mkdir("lookups\\plots_lookup\\")
        if not os.path.exists("lookups\\plots_lookup\\" + channelName + "\\"):
            os.mkdir("lookups\\plots_lookup\\" + channelName + "\\")
        
        for j in range(len(xData)):
            polyData = np.empty(shape = len(xData[j]), dtype = float)

            for k in range(len(xData[j])):
                value = 0
                poly = polys[j]
                poly = np.flip(poly)

                for l in range(len(poly)):
                    value += poly[l] * pow(xData[j][k], l)

                polyData[k] = value

            plt.xlabel("Voltage (V)")
            plt.ylabel("Amplitude Difference (dB)")

            plt.plot(xData[j], yData[j], label = dataLabels[j])
            plt.plot(xData[j], polyData, label = "Fitted Polynomial")
            plt.legend()
            plt.savefig("lookups\\plots_lookup\\" + channelName + "\\plot_" + str(plots + 1) + ".png")
            plt.clf()
            plots += 1

#### MISC FUNCTIONS ####

def IsStringNum(string, category):
    if (category == 0):
        try:
            int(string)
            return True
        except ValueError:
            return False
    elif (category == 1):
        try:
            float(string)

            return True
        except ValueError:
            return False

def FindInArray(array, value):
    return np.where(array == GetClosestNum(array, value))[0]

def LoadSpectrums():
    spectrumData = {}
    spectrumNames = [f for f in listdir(SpectrumPath) if isfile(join(SpectrumPath, f))]

    for i in range(len(spectrumNames)):
        if (spectrumNames[i][len(spectrumNames[i]) - 4 : len(spectrumNames[i])] != ".txt"):
            continue
        if (spectrumNames[i].find("_all") != -1):
            continue

        channelNameIdx = spectrumNames[i].find("_Ch")

        if (channelNameIdx == -1):
            print("WARNING: Unable to identify which channel \"" + spectrumNames[i] + "\" belongs to!")
            continue

        channelSuffix = spectrumNames[i][channelNameIdx + 1 : len(spectrumNames[i]) - 4]
        channel = int(channelSuffix[2 : channelSuffix.find("_")])
        voltage = None

        try:
            voltage = float(channelSuffix[channelSuffix.find("_") + 1 : len(channelSuffix) - 1])
        except:
            print("WARNING: Unable to identify which voltage \"" + spectrumNames[i] + "\" represents!")
            continue

        data = [np.empty(shape = SpectrumLength, dtype = float), np.empty(shape = SpectrumLength, dtype = float)]
        insIdx = 0

        file = open(SpectrumPath + spectrumNames[i], "r")

        while (True):
            line = file.readline()

            if not (line):
                break
            elif (line.find("#") != -1) or (line.find("[") != -1):
                continue

            wordBuff = ""
            point = []
            inserted = 0

            for j in range(len(line)):
                currChar = line[j]

                if (currChar == " ") or (currChar == "\t"):
                    continue
                elif (currChar == ";") or (currChar == ",") or (currChar == '\n'):
                    num = float(wordBuff)

                    if (BaseUnits == True):
                        if (inserted == 0):
                            num = 10000000 / num
                        elif (inserted == 1):
                            num = 10 * np.log10(num)

                    point.append(num)
                    wordBuff = ""
                    inserted += 1

                    if (currChar == '\n'):
                        break

                    continue

                wordBuff += currChar

            if (len(point) >= 2):
                if (BaseUnits == True):
                    data[0][(SpectrumLength - 1) - insIdx] = point[0]
                    data[1][(SpectrumLength - 1) - insIdx] = point[1]
                else:
                    data[0][insIdx] = point[0]
                    data[1][insIdx] = point[1]

                insIdx += 1

        file.close()

        if not (channel in spectrumData):
            spectrumData[channel] = {}

        spectrumData[channel][voltage] = data

        print("Loaded spectrum \"" + spectrumNames[i] + "\" data.")

    return spectrumData

#### MAIN FUNCTION ####

def main():
    print("---- SCRIPT STARTED ----")

    # STEP 1: Load spectrums
    print("\n-- STEP 1: LOADING SPECTRUMS --")
    spectrumData = LoadSpectrums()

    for channel in spectrumData:
        voltages = []
        sortedVoltages = []
        powers = []

        for voltage in spectrumData[channel]:
            voltages.append(voltage)
            powers.append(spectrumData[channel][voltage])

        sortedVoltages = []

        for i in range(len(voltages)):
            sortedVoltages.append(voltages[i])

        sortedVoltages.sort()
        sortedPowers = []

        for voltage in sortedVoltages:
            sortedPowers.append(powers[FindInArray(voltages, voltage)[0]])

        sortedData = {}

        for i in range(len(sortedVoltages)):
            sortedData[sortedVoltages[i]] = sortedPowers[i]

        spectrumData[channel] = sortedData

    # STEP 2: Diff spectrums
    print("\n-- STEP 2: DIFFING SPECTRUMS --")
    diffData = DiffSpectrums(spectrumData)

    # STEP 3: Smooth spectrums
    if (SmoothMatching == True):
        print("\n-- STEP 3: SMOOTHING SPECTRUMS --")

        for channel in diffData:
            for voltage in diffData[channel]:
                diffData[channel][voltage] = GaussianSmoothData(diffData[channel][voltage], [1400, 1780])

                print("Smoothened data of channel " + str(channel) + " of voltage " + str(voltage) + ".")

    if not os.path.exists("lookups\\"):
        os.mkdir("lookups\\")

    if (GenerateGraphs == True):
        if not os.path.exists("lookups\\plots_data\\"):
            os.mkdir("lookups\\plots_data\\")

        for channel in spectrumData:
            if not os.path.exists("lookups\\plots_data\\" + str(channel) + "\\"):
                os.mkdir("lookups\\plots_data\\" + str(channel) + "\\")

            for voltage in diffData[channel]:
                plt.xlim(1400, 1800)
                plt.plot(spectrumData[channel][voltage][0], spectrumData[channel][voltage][1], label = "Spectrum Data")
                plt.plot(diffData[channel][voltage][0], diffData[channel][voltage][1], label = "Diff Data")
                plt.legend()
                plt.savefig("lookups\\plots_data\\" + str(channel) + "\\plot_" + str(channel) + "_" + str(voltage) + ".png")
                plt.clf()

    # STEP 4: Generate lookup tables
    print("\n-- STEP 4: GENERATING LOOKUPS --")
    GenerateLookups(diffData)

    print("\n---- SCRIPT FINISHED ----")

if (__name__ == "__main__"):
    main()
