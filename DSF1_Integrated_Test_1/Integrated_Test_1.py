#### IMPORTS ####

import sys
from os import listdir
from os.path import isfile, join
import glob
import serial
import time
import clr
import numpy as np
import matplotlib.pyplot as plt

clr.AddReference("C:\\Program Files\\Thorlabs\\ThorSpectra\\ThorlabsOSAWrapper.dll")

from ThorlabsOSAWrapper import *
from ThorlabsOSAWrapper import InstrumentModel
from ThorlabsOSAWrapper import AcquisitionUpdateFlag
from ThorlabsOSAWrapper import SpectrumStruct
from ThorlabsOSAWrapper import FileIOInterface
from ThorlabsOSAWrapper import XUnit
from ThorlabsOSAWrapper import YUnit
 
#### GLOBAL VARIABLES ####

Iterations = 40                                 # Number of main loop iterations
ThresholdLevels = [-0.5, 0.5]                   # Threshold levels between which the spectrum power is accepted
MovePercentage = 0.5                            # Percentage of voltage move steps that will be taken
MaxVoltage = 17                                 # Max voltage that the script will not exceed (inclusive)
LookupPath = "lookups\\"                        # Path for lookup tables
WaveguidePath = "reference_spectrum_-50.spf2"   # Reference spectrum filename
OffsetsPath = "reference_offsets_-50.txt"       # Reference offsets filename
SmoothMatching = False                          # Enables smooth matching of spectral channel power calculations (False = calculated as average over wavelength interval | True = calculated as middle point after Guassian smoothing)
UsePoly = False                                 # Whether the fitted polynomial will be used to acquire values from lookup tables, instead of raw data
GenerateGraphs = True                           # Whether plot PNG files are generated
TestMode = False                                # If enabled, only spectral channel 1510 will be affected

SpectralLengths = 12566                         # Expected size of all spectrums
SpectralChannels = [                            # Spectral channels that will be affected
                    1410, 
                    1430, 
                    1450, 
                    1470, 
                    1490, 
                    1510, 
                    1530, 
                    1550, 
                    1570, 
                    1590,
                    1610, 
                    1630, 
                    1650, 
                    1670, 
                    1690,
                    1710,
                    1730,
                    1750,
                    1770,
                    1790
                    ]
SpectralChannelWidth = 20                       # (nm) Width of each spectral channel
OSASensitivity = 0                              # Sensitivity setting of OSA
OSAResoltion = 0                                # Resolution setting of OSA
DiscardedSpectrumNum = 2                        # Number of discarded spectrums prior to start of main loop
BaseUnitsReference = True                       # Whether the reference spectrum will be in the base units (X: cm^(-1) | Y: mW)
BaseUnitsOSA = True                             # Whether the OSA outputs spectrums that will be in the base units (X: cm^(-1) | Y: mW)

CheckXPOWChannels = [                           # XPOW channels that will be sent commands
                    #117,   # 1410 - TOPM
                    #115,    # 1430 - TOPM
                    #114,    # 1450 - TOPM
                    #113,    # 1470 - TOPM
                    #112,    # 1490 - TOPM
                    #109,    # 1510 - TOPM
                    #108,    # 1530 - TOPM
                    #107,    # 1550 - TOPM
                    #105,    # 1570 - TOPM
                    #104,    # 1590 - TOPM
                    #103,    # 1610 - TOPM
                    #100,    # 1630 - TOPM
                    #99,     # 1650 - TOPM
                    #98,     # 1670 - TOPM
                    #97,     # 1690 - TOPM

                    37,     # 1410 - MZI
                    36,     # 1430 - MZI
                    33,     # 1450 - MZI
                    32,     # 1470 - MZI
                    31,     # 1490 - MZI
                    29,     # 1510 - MZI
                    28,     # 1530 - MZI
                    27,     # 1550 - MZI
                    24,     # 1570 - MZI
                    23,     # 1590 - MZI
                    22,     # 1610 - MZI
                    21,     # 1630 - MZI
                    18,     # 1650 - MZI
                    17,     # 1670 - MZI
                    16,     # 1690 - MZI
                    14,     # 1710 - MZI
                    13,     # 1730 - MZI
                    12,     # 1750 - MZI
                    9,      # 1770 - MZI
                    8       # 1790 - MZI
                    ]
XPOWCurrentMax = 50                             # (mA) Max current applied to all 120 XPOW channels
XPOWPorts = ["COM36", "COM38", "COM35"]         # (D [1 (1 - 40), 2 (41 - 80), 3 (81 - 120)]) USB ports that XPOW is connected to
XPOWBaudRate = 115200                           # Baud rate used by XPOW
XPOWResetInterval = 0                           # Reading interval between reset of XPOW COM ports
XPOWCommandTimeout = 15                         # (1/10 sec) Timeout after command sent to XPOW COM port
XOPWCommandDelay = 0.005                        # (sec) Delay after each command sent to XPOW

FlattenReach = 0.001                            # Reach percentage argument of Gaussian smoothing function
FlattenSigma = 50                               # Sigma argument of Gaussian smoothing function
FlattenPhi = 0                                  # Phi (phase) argument of Gaussian smoothing function
FlattenAmplitude = 1                            # Amplitude argument of Gaussian smoothing function

IterationDelay = 1                              # (sec) Delay after each iteration of main loop
VoltageDelay = 0.005                            # (sec) Delay after each XPOW voltage setting
SpectrumCheckDelay = 2                          # (sec) Interval between checks for an acquired OSA spectrum
ErrorDelay = 1                                  # (sec) Delay between re-tries of sending XPOW commands of error encountered

## DO NOT EDIT ##

LookupTable = {}
OffsetsTable = {}
CurrVoltages = {}
Waveguide = None
OSA = None
Locator = None
CurrSpectrum = None
SerialLines = []
FileInterface = FileIOInterface()
StartTime = 0
SaveSpectrum = True
WaitingSpectrum = False

if (XPOWCurrentMax > 300):
    XPOWCurrentMax = 300

if (TestMode == True):
    SpectralChannels = [1510]
    CheckXPOWChannels = [29]

#### OSA FUNCTIONS ####

def OnSingleAcquisition(sender, event_args):
    global CurrSpectrum
    global WaitingSpectrum

    if (event_args.CallbackMessage.LastDataTypeUpdateFlag == AcquisitionUpdateFlag.Spectrum):
        if (SaveSpectrum == True):
            delay = round(time.time() - StartTime, 2)

            CurrSpectrum = SpectrumStruct(sender.ChannelInterfaces[0].GetLastSpectrumLength(), True)
            
            CurrSpectrum.XAxisUnit = int(XUnit.nmAir)
            CurrSpectrum.YAxisUnit = int(YUnit.dB)
            
            sender.ChannelInterfaces[0].GetLastSpectrum(CurrSpectrum)

            print("Spectrum Acquired! (" + str(delay) + " sec)")

        WaitingSpectrum = False

def GetSpectralData(spectrum, baseUnits):
    dataX = np.empty(shape = SpectralLengths, dtype = float)
    dataY = np.empty(shape = SpectralLengths, dtype = float)

    for i in range(SpectralLengths):
        if (baseUnits == True):
            dataX[(SpectralLengths - 1) - i] = 10000000 / spectrum.GetXAtIndex(i)
            dataY[(SpectralLengths - 1) - i] = 10 * np.log10(spectrum.GetValueAtIndex(i))
        else:
            dataX[i] = spectrum.GetXAtIndex(i)
            dataY[i] = spectrum.GetValueAtIndex(i)

    return [dataX, dataY]

def GetOSASpectrum():
    global OSA
    global StartTime
    global WaitingSpectrum

    OSA.AcquireSingleSpectrum()
    StartTime = time.time()
    WaitingSpectrum = True

    while (WaitingSpectrum == True):
        time.sleep(SpectrumCheckDelay)

def InitOSA():
    global OSA
    global Locator

    Locator = DeviceLocator()
    osaNum = Locator.InitializeSpectrometers()
    
    print("OSAs found: " + str(osaNum))
    
    if (osaNum > 0):
        print("Opening first OSA...")

        OSA = LibDeviceInterface(0)

    else:
        print("No OSA Found! Quitting...")

        raise SystemExit

    OSA.SetSensitivityMode(OSASensitivity) 
    OSA.SetResolutionMode(OSAResoltion) 
    OSA.AcquisitionSettings.AutomaticGain = True
    OSA.OnSingleAcquisition += OnSingleAcquisition

#### XPOW FUNCTIONS ####

def GetAllSerialPorts():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def WaitSerialResponse(portName, serialData, maxIterations):
    iteration = 0

    while iteration < maxIterations:
        if (serialData.inWaiting() > 0):        
            myData = serialData.readline().decode('utf-8')[5:-3]
            msg = portName + "\tanswer = {" + myData + "}"

            return msg
        else:
            iteration += 1
            time.sleep(0.1)

    return portName + " is not answered"

def CheckXPOWkey(xpowKey, maxIterations):
    command = "*key?\n"

    for j in range(len(SerialLines)):
        SerialLines[j].write(command.encode())
        value = WaitSerialResponse(XPOWPorts[j], SerialLines[j], maxIterations)

        if (xpowKey in value):
            print(value + " | XPOW key MATCHED")
        else:
            print(value + " | XPOW key NOT MATCHED")

def SendXPOWCommandAll(cmd, maxIterations):
    print("Sent Command: \"" + cmd + "\" to all Ports...")
    command = cmd + "\n"

    for j in range(len(SerialLines)):
        SerialLines[j].write(command.encode())
        time.sleep(XOPWCommandDelay)

        print(WaitSerialResponse(XPOWPorts[j], SerialLines[j], maxIterations))

def SendXPOWCommandSingle(cmd, portIdx, maxIterations):
    if (portIdx >= len(SerialLines)):
        print("Chosen port index is too large!")

        return

    print("Sent Command: \"" + cmd + "\" to Port \"" + XPOWPorts[portIdx] + "\"...")
    command = cmd + "\n"

    SerialLines[portIdx].write(command.encode())
    time.sleep(XOPWCommandDelay)

    if (maxIterations > 0):
        print(WaitSerialResponse(XPOWPorts[portIdx], SerialLines[portIdx], maxIterations))

def CreateXPOWPorts():
    for i in range(len(XPOWPorts)):
        SerialLines.append(serial.Serial(XPOWPorts[i], baudrate = XPOWBaudRate, timeout = 3.0, writeTimeout = 0))
        time.sleep(XOPWCommandDelay)

    print("XPOW ports created!")

def ClearXPOWPorts():
    CloseXPOWPorts()

    SerialLines.clear()
    time.sleep(XOPWCommandDelay)

    print("XPOW ports cleared!")

def OpenXPOWPorts():
    for i in range(len(SerialLines)):
        SerialLines[i].open()
        time.sleep(XOPWCommandDelay)

    print("XPOW ports opened!")

def CloseXPOWPorts():
    for i in range(len(SerialLines)):
        SerialLines[i].close()
        time.sleep(XOPWCommandDelay)

    print("XPOW ports closed!")

def ResetXPOWPorts():
    CloseXPOWPorts()
    OpenXPOWPorts()

def ResetAllXPOWChannels():
    #channel = 1

    #while (channel <= 120):
    for channel in CheckXPOWChannels:
        portIdx = 0
        modChannel = channel

        if (channel >= 41) and (channel <= 80):
            portIdx = 1
            modChannel = channel - 40
        elif (channel >= 81) and (channel <= 120):
            portIdx = 2
            modChannel = channel - 80

        SendXPOWCommandSingle("CH:" + str(modChannel) + ":VOLT:0", portIdx, XPOWCommandTimeout)
        time.sleep(VoltageDelay)
        SendXPOWCommandSingle("CH:" + str(modChannel) + ":CUR:" + str(XPOWCurrentMax), portIdx, XPOWCommandTimeout)
        time.sleep(VoltageDelay)

        channel += 1

def InitXPOW():
    print("------------------------------------------\nChosen XPOW Ports: " + str(XPOWPorts) + "\n------------------------------------------\n")

    startError = True

    while (startError == True):
        try:
            CreateXPOWPorts()
            SendXPOWCommandAll("board?", XPOWCommandTimeout)
            ResetAllXPOWChannels()

            startError = False

        except:
            print("\n-- ERROR ENCOUNTERED, TRYING AGAIN... --")

            time.sleep(ErrorDelay)

#### MATH FUNCTIONS ####

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

def GetClosestNum(array, base):
    array = np.asarray(array)

    return array[(np.abs(array - base)).argmin()]

#### MISC FUNCTIONS ####

def GetSpectralPower(spectralChannel, spectrumData):
    if (SmoothMatching == True):
        return spectrumData[1][FindInArray(spectrumData[0], GetClosestNum(spectrumData[0], spectralChannel))[0]]

    else:
        channelEdgesX = [GetClosestNum(spectrumData[0], spectralChannel - (SpectralChannelWidth / 2)), GetClosestNum(spectrumData[0], spectralChannel + (SpectralChannelWidth / 2))]
        channelEdgesIndices = [FindInArray(spectrumData[0], channelEdgesX[0])[0], FindInArray(spectrumData[0], channelEdgesX[1])[0]]
        channelData = spectrumData[1][channelEdgesIndices[0] : channelEdgesIndices[1]]

        return sum(channelData) / len(channelData)

def FindInArray(array, value):
    return np.where(array == GetClosestNum(array, value))[0]

def GetLookupAmplitude(XPOWChannel, spectralChannel, voltage):
    if not (XPOWChannel in LookupTable):
        print("Unable to find XPOW channel " + str(XPOWChannel) + " in lookup!")

        ResetAllXPOWChannels()
        sys.exit()

    table = LookupTable[XPOWChannel]

    if not (spectralChannel in table):
        print("Unable to find spectral channel " + str(spectralChannel) + " in XPOW channel " + str(XPOWChannel) + " in lookup!")

        ResetAllXPOWChannels()
        sys.exit()

    if (UsePoly == True):
        poly = table[spectralChannel][1]
        value = 0
        
        for i in range(len(poly)):
            value += poly[i] * pow(voltage, (len(poly) - 1) - i)

        return round(value, 2)

    else:
        data = table[spectralChannel]
        voltages = []
        amplitudes = []

        for i in range(len(data[0])):
            voltages.append(data[0][i][0])
            amplitudes.append(data[0][i][1])

        return round(amplitudes[FindInArray(voltages, GetClosestNum(voltages, voltage))[0]], 2)

def GetLookupVoltage(XPOWChannel, spectralChannel, amplitude):
    if not (XPOWChannel in LookupTable):
        print("Unable to find XPOW channel " + str(XPOWChannel) + " in lookup!")

        ResetAllXPOWChannels()
        sys.exit()

    table = LookupTable[XPOWChannel]

    if not (spectralChannel in table):
        print("Unable to find spectral channel " + str(spectralChannel) + " in XPOW channel " + str(XPOWChannel) + " in lookup!")

        ResetAllXPOWChannels()
        sys.exit()

    data = table[spectralChannel]
    voltages = []
    amplitudes = []

    for i in range(len(data[0])):
        voltages.append(data[0][i][0])
        amplitudes.append(data[0][i][1])

    #minIdx = FindInArray(amplitudes, min(amplitudes))[0]
    #maxIdx = FindInArray(amplitudes, max(amplitudes))[0]

    if (UsePoly == False):
        #if (minIdx < maxIdx):
        #    voltages = voltages[minIdx : maxIdx]
        #    amplitudes = amplitudes[minIdx : maxIdx]
        #elif (minIdx > maxIdx):
        #    voltages = voltages[maxIdx : minIdx]
        #    amplitudes = amplitudes[maxIdx : minIdx]

        #print("Requested amplitude of " + str(round(amplitude, 2)) + " dB, finds amplitude of " + str(round(GetClosestNum(amplitudes, amplitude), 2)) + " db.")
        #print("Found amplitude gives voltage of " + str(round(voltages[FindInArray(amplitudes, GetClosestNum(amplitudes, amplitude))[0]], 2)) + "V.")

        foundAmplitudes = []
        prevAmplitude = 0

        for i in range(len(amplitudes)):
            if (amplitudes[i] == amplitude):
                foundAmplitudes.append([amplitudes[i], amplitudes[i]])
            elif (i > 0) and (amplitudes[i] > amplitude) and (prevAmplitude < amplitude):
                foundAmplitudes.append([prevAmplitude, amplitudes[i]])

            prevAmplitude = amplitudes[i]

        foundVoltages = []

        for amplitudePair in foundAmplitudes:
            foundVoltages.append(voltages[FindInArray(amplitudes, GetClosestNum(amplitudePair, amplitude))[0]])

        if (len(foundVoltages) == 0):
            foundVoltages = [voltages[FindInArray(amplitudes, GetClosestNum(amplitudes, amplitude))[0]]]

        #print(amplitude)
        #print(foundAmplitudes)
        #print(foundVoltages)

        if (len(foundVoltages) == 0):
            print("ERROR: Unable to find voltage for amplitude of " + str(amplitude) + " dB in channel " + str(XPOWChannel) + " of " + str(spectralChannel) + "!")

            ResetAllXPOWChannels()
            sys.exit()

        return foundVoltages

    else:
        #minVoltage = voltages[minIdx]
        #maxVoltage = voltages[maxIdx]

        poly = table[spectralChannel][1]
        p = np.poly1d(poly)

        foundVoltages = (p - amplitude).roots

        print(poly)
        print(amplitude)
        print(foundVoltages)

        retVoltages = []

        for i in range(len(foundVoltages)):
            insVoltage = None

            if (isinstance(foundVoltages[i], complex) == True):
                if (foundVoltages[i].imag == 0):
                    insVoltage = round(foundVoltages[i].real, 2)

            else:
                insVoltage = round(foundVoltages[i], 2)

            if (insVoltage != None) and ((insVoltage >= 0) and (insVoltage <= MaxVoltage)):
                retVoltages.append(insVoltage)

        if (len(retVoltages) == 0):
            print("WARNING: Unable to find exact voltage for amplitude of " + str(round(amplitude, 2)) + " dB in channel " + str(XPOWChannel) + " of " + str(spectralChannel) + "!")

            distances = []
            distVoltages = []
            voltageStep = 0.1
            currVoltage = 0
            prevDiff = 0
            prevAmplitude = 0

            while (currVoltage < MaxVoltage):
                currAmplitude = GetLookupAmplitude(XPOWChannel, spectralChannel, currVoltage)

                if (currVoltage > 0):
                    currDiff = currAmplitude - prevAmplitude

                    if (currVoltage > voltageStep) and (((amplitude < currAmplitude) and ((currDiff > 0) and (prevDiff < 0))) or ((amplitude > currAmplitude) and ((currDiff < 0) and (prevDiff > 0)))):
                        distances.append(amplitude - currAmplitude)
                        distVoltages.append(currVoltage)

                    prevDiff = currDiff

                prevAmplitude = currAmplitude
                currVoltage += voltageStep

            if (len(distances) != 0):
                retVoltages.append(GetClosestNum(distances, 0))

        if (len(retVoltages) == 0):
            print("ERROR: Unable to find voltage for amplitude of " + str(round(amplitude, 2)) + " dB in channel " + str(XPOWChannel) + " of " + str(spectralChannel) + "!")

            ResetAllXPOWChannels()
            sys.exit()

        return retVoltages

def LoadWaveguide():
    global Waveguide

    Waveguide = SpectrumStruct(SpectralLengths)
    FileInterface.ReadSpectrum(Waveguide, WaveguidePath, 0)

def LoadOffsetsTable():
    global OffsetsTable

    file = open(OffsetsPath, "r")

    while (True):
        line = file.readline()

        if not (line):
            break

        if (line == "\n") or (line.find("OFFSETS") != -1):
            continue
        
        wordBuff = ""
        currSpectralChannel = -1
        currData = []

        for i in range(len(line)):
            currChar = line[i]

            if (currChar == '\n') or (currChar == ","):
                currData.append(float(wordBuff))
                wordBuff = ""

                continue
            elif (currChar == ":"):
                currSpectralChannel = int(wordBuff)
                wordBuff = ""

                continue

            wordBuff += currChar

        if (currSpectralChannel != -1) and (len(currData) >= 2):
            OffsetsTable[currSpectralChannel] = currData

        else:
            if (currSpectralChannel != -1):
                print("WARNING: Unable to acquire an entry of spectral channel " + str(currSpectralChannel) + " from the offsets file!")
            else:
                print("WARNING: Unable to acquire an entry from the offsets file!")

    file.close()

def LoadLookupTable():
    global LookupTable

    lookupNames = [f for f in listdir(LookupPath) if isfile(join(LookupPath, f))]

    for i in range(len(lookupNames)):
        currChannel = -1
        currTable = {}
        currLabel = -1
        currData = []
        currPoly = []

        file = open(LookupPath + lookupNames[i], "r")

        linesRead = 0

        while (True):
            line = file.readline()

            if not (line):
                if (currChannel != -1):
                    if (currLabel != -1):
                        if (len(currData) == 0):
                            print("WARNING: Data missing for channel " + str(currChannel) + "!")
                        if (len(currPoly) == 0):
                            print("WARNING: Poly missing for channel " + str(currChannel) + "!")

                        if (currLabel in currTable):
                            print("WARNING: Label " + str(currLabel) + " already in table of channel " + str(currChannel) + "!")

                        currTable[currLabel] = [currData, currPoly]

                break

            wordBuff = ""

            for j in range(len(line)):
                currChar = line[j]

                if (currChar == ' ') or (currChar == '\t'):
                    continue
                
                wordBuff += currChar

                if (wordBuff == "LOOKUP"):
                    k = 1
                    num = ""

                    while ((j + k) < len(line)):
                        if (line[j + k] == '\n'):
                            break

                        num += line[j + k]
                        k += 1

                    currChannel = int(num)

                    break

                elif (wordBuff == "LABEL"):
                    if (currChannel != -1):
                        if (currLabel != -1):
                            if (len(currData) == 0):
                                print("WARNING: Data missing for channel " + str(currChannel) + "!")
                            if (len(currPoly) == 0):
                                print("WARNING: Poly missing for channel " + str(currChannel) + "!")

                            if (currLabel in currTable):
                                print("WARNING: Label " + str(currLabel) + " already in table of channel " + str(currChannel) + "!")

                            currTable[currLabel] = [currData, currPoly]

                            currLabel = -1
                            currData = []
                            currPoly = []

                    k = 1
                    name = ""

                    while ((j + k) < len(line)):
                        if (line[j + k] == '\n'):
                            break

                        name += line[j + k]
                        k += 1

                    currLabel = float(name)

                    break

                elif (wordBuff == "DATA"):
                    k = 0
                    point = ""
                    points = []
                    inserted = 0

                    while ((j + k) < len(line)):
                        if (len(line) - 1):
                            k += 1

                        if (line[j + k] == '(') or (line[j + k] == ' ') or (line[j + k] == '\t'):
                            continue
                        elif (line[j + k] == ',') or (line[j + k] == ')'):
                            points.append(float(point))
                            point = ""
                            inserted += 1

                            if (inserted >= 2):
                                break

                            continue

                        point += line[j + k]

                    currData.append(points)

                    break

                elif (wordBuff == "POLY"):
                    k = 0
                    coeff = ""
                    coeffs = []

                    while ((j + k) < len(line)):
                        if (len(line) - 1):
                            k += 1

                        if (line[j + k] == '(') or (line[j + k] == ' ') or (line[j + k] == '\t'):
                            continue
                        elif (line[j + k] == ',') or (line[j + k] == ')'):
                            coeffs.append(float(coeff))
                            coeff = ""

                            if (line[j + k] == ')'):
                                break

                            continue

                        coeff += line[j + k]

                    for l in range(len(coeffs)):
                        currPoly.append(coeffs[l])

                    break

            linesRead += 1

        if (currChannel != -1):
            if (currChannel in LookupTable):
                print("WARNING: Channel " + str(currChannel) + " already in lookup table!")

            LookupTable[currChannel] = currTable
        else:
            print("WARNING: Unable to identify a channel in \"" + lookupNames[i] + "\"!")

#### MAIN FUNCTION ####

def main():
    global OSA
    global CurrVoltages
    global StartTime
    global SaveSpectrum

    print('---- SCRIPT STARTED ----')

    if (TestMode == True):
        print("WARNING: Test mode is enabled.")

    # STEP 1: Load lookup table
    print("\n-- STEP 1: LOADING LOOKUP --")
    LoadLookupTable()

    # STEP 2: Load reference spectrum
    print("\n-- STEP 2: LOADING REFERENCE --")
    LoadWaveguide()
    waveguideData = GetSpectralData(Waveguide, BaseUnitsReference)
    waveguideScanData = np.copy(waveguideData)

    if (SmoothMatching == True):
        waveguideScanData = GaussianSmoothData(waveguideScanData, [1400, 1800])

    LoadOffsetsTable()

    # STEP 3: Connect to OSA
    print("\n-- STEP 3: CONNECTING OSA --")
    InitOSA()

    # STEP 4: Connect to XPOW
    print("\n-- STEP 4: CONNECTING XPOW --")
    InitXPOW()

    for i in range(len(CheckXPOWChannels)):
        channel = CheckXPOWChannels[i]

        if (channel >= 1) and (channel <= 40):
            continue
        elif (channel >= 41) and (channel <= 80):
            continue
        elif (channel >= 81) and (channel <= 120):
            continue
        else:
            print("Invalid XPOW channel ID (" + str(modChannel) + ")!")

            return

    # STEP 5: Discarding spectrums
    print("\n-- STEP 5: DISCARDING " + str(DiscardedSpectrumNum) + " SPECTRUMS --")

    SaveSpectrum = False

    for j in range(DiscardedSpectrumNum):
        GetOSASpectrum()
        print("Discarded spectrum " + str(j + 1) + "...")

    SaveSpectrum = True

    #SendXPOWCommandSingle("CH:29:VOLT:13", 0, XPOWCommandTimeout)
    #time.sleep(VoltageDelay)
    #SendXPOWCommandSingle("CH:29:VAL?", 0, XPOWCommandTimeout)
    #SendXPOWCommandSingle("CH:37:VOLT:13", 0, XPOWCommandTimeout)
    #time.sleep(VoltageDelay)
    #SendXPOWCommandSingle("CH:37:VAL?", 0, XPOWCommandTimeout)
    #SendXPOWCommandSingle("CH:16:VOLT:13", 0, XPOWCommandTimeout)
    #time.sleep(VoltageDelay)
    #SendXPOWCommandSingle("CH:16:VAL?", 0, XPOWCommandTimeout)

    #SendXPOWCommandSingle("CH:29:VOLT:0", 0, XPOWCommandTimeout)
    #time.sleep(VoltageDelay)
    #SendXPOWCommandSingle("CH:29:VAL?", 0, XPOWCommandTimeout)
    #SendXPOWCommandSingle("CH:37:VOLT:0", 0, XPOWCommandTimeout)
    #time.sleep(VoltageDelay)
    #SendXPOWCommandSingle("CH:37:VAL?", 0, XPOWCommandTimeout)
    #SendXPOWCommandSingle("CH:16:VOLT:0", 0, XPOWCommandTimeout)
    #time.sleep(VoltageDelay)
    #SendXPOWCommandSingle("CH:16:VAL?", 0, XPOWCommandTimeout)

    for spectralChannel in SpectralChannels:
        CurrVoltages[spectralChannel] = 0

    firstSpectrum = 0
    #firstDiff = 0
    plots = 0
    errorNum = 0

    # STEP 6: Closed Loop
    print("\n-- STEP 6: STARTING MAIN LOOP --")

    for iteration in range(Iterations):
        print("\n-- STARTING LOOP ITERATION " + str(iteration + 1) + " --\n")
        startTime = time.time()

        # STEP 6.1: Acquire current spectrum from OSA
        GetOSASpectrum()
        spectrumData = GetSpectralData(CurrSpectrum, BaseUnitsOSA)
        spectrumScanData = np.copy(spectrumData)

        # STEP 6.2: Diff current spectrum vs. reference waveguide spectrum
        diffData = [spectrumData[0], waveguideData[1] - spectrumData[1]]
        diffScanData = np.copy(diffData)

        # STEP 6.3: Smooth diff spectrum
        diffData = GaussianSmoothData(diffData, [1400, 1800])

        # STEP 6.4: Determine power of each spectral channel in diff spectrum
        spectralPowers = []
        diffPowers = []

        if (SmoothMatching == True):
            spectrumScanData = GaussianSmoothData(spectrumScanData, [1400, 1800])
            diffScanData = GaussianSmoothData(diffScanData, [1400, 1800])

        for i in range(len(SpectralChannels)):
            spectralPowers.append(GetSpectralPower(SpectralChannels[i], spectrumScanData))
            diffPowers.append(GetSpectralPower(SpectralChannels[i], diffScanData))

        if (iteration == 0):
            firstSpectrum = np.copy(spectrumData)
            #firstDiff = np.copy(diffData)

        if (GenerateGraphs == True):
            plt.xlim(1400, 1800)
            plt.ylim(-100, 0)
            plt.plot(waveguideData[0], waveguideData[1], label = "Reference Spectrum", color = "#ff7f0e")
            plt.plot(firstSpectrum[0], firstSpectrum[1], label = "First Spectrum", alpha = 0.25, color = "#2ca02c")
            plt.plot(spectrumData[0], spectrumData[1], label = "Current Spectrum", color = "#1f77b4")
            #plt.plot(diffData[0], diffData[1], label = "Diff Data")
            #plt.plot(firstDiff[0], firstDiff[1], alpha = 0.25)
            plt.legend()
            #plt.show()
            plt.savefig("plot" + str(plots + 1) + ".png")
            plt.close()
            plt.clf()

            plots += 1

        resetTracker = 0

        # STEPS 6.5 & 6.6: Compare power of each spectral chanel to threshold levels & adjust if neccessary
        for i in range(len(spectralPowers)):
            targetDiffPower = OffsetsTable[SpectralChannels[i]][0]

            print("\n- SPECTRAL CHANNEL " + str(SpectralChannels[i]) + " -")
            print("Current power: " + str(round(spectralPowers[i], 2)) + " dB.")
            print("Current difference power: " + str(round(diffPowers[i], 2)) + " dB.")
            print("Current target power: " + str(round(targetDiffPower, 2)) + " dB.")

            if (diffPowers[i] < targetDiffPower + ThresholdLevels[0]) or (diffPowers[i] > targetDiffPower + ThresholdLevels[1]):
                #possibleTargetPowers = [spectralPowers[i] - diffPowers[i], spectralPowers[i] + diffPowers[i]]
                #chosenTargetPower = GetClosestNum(possibleTargetPowers, FlatLine)
                #possibleTargetVoltages = CurrVoltages[SpectralChannels[i]] + GetLookupVoltage(CheckXPOWChannels[i], SpectralChannels[i], chosenTargetPower)
                #possibleDiffVoltages = []
                possibleTargetVoltages = GetLookupVoltage(CheckXPOWChannels[i], SpectralChannels[i], GetLookupAmplitude(CheckXPOWChannels[i], SpectralChannels[i], CurrVoltages[SpectralChannels[i]]) + diffPowers[i])
                chosenTargetVoltage = GetClosestNum(possibleTargetVoltages, CurrVoltages[SpectralChannels[i]])
                #chosenTargetVoltage = CurrVoltages[SpectralChannels[i]] + GetClosestNum(possibleTargetVoltages, CurrVoltages[SpectralChannels[i]])
                diffVoltage = (chosenTargetVoltage - CurrVoltages[SpectralChannels[i]]) * MovePercentage
                targetVoltage = CurrVoltages[SpectralChannels[i]] + diffVoltage

                #print(possibleTargetPowers)
                #print(chosenTargetPower)
                #print(possibleTargetVoltages)
                #print(chosenTargetVoltage)
                #print(chosenTargetVoltage)
                #print(diffVoltage)
                #print(targetVoltage)
                #targetVoltage = CurrVoltages[SpectralChannels[i]] + GetLookupVoltage(CheckXPOWChannels[i], SpectralChannels[i], targetPower)[0]

                #if (targetPower == possibleTargetPowers[0]):
                #    diffVoltage = -diffVoltage

                #targetVoltage += diffVoltage

                print("Current voltage: " + str(round(CurrVoltages[SpectralChannels[i]], 2)) + "V.")
                #print("Possible target voltages: " + str(possibleTargetVoltages) + ".")
                print("Chosen target voltage: " + str(round(chosenTargetVoltage, 2)) + "V.")
                print("Difference voltage: " + str(round(diffVoltage, 2)) + "V.")
                print("Final target voltage: " + str(round(targetVoltage, 2)) + "V.")

                if (targetVoltage < 0):
                    CurrVoltages[SpectralChannels[i]] = 0

                    print("WARNING: Attempt to apply " + str(round(targetVoltage, 2)) + "V to spectral channel " + str(SpectralChannels[i]) + "!")

                elif (targetVoltage > MaxVoltage):
                    CurrVoltages[SpectralChannels[i]] = MaxVoltage

                    print("WARNING: Attempt to apply " + str(round(targetVoltage, 2)) + "V to spectral channel " + str(SpectralChannels[i]) + "!")

                else:
                    CurrVoltages[SpectralChannels[i]] = targetVoltage

                print("Final applied voltage: " + str(round(CurrVoltages[SpectralChannels[i]], 2)) + "V.")

                error = True

                while (error == True):
                    try:
                        if (resetTracker >= XPOWResetInterval):
                            ResetXPOWPorts()
                            resetTracker = 0
                        else:
                            resetTracker += 1

                        portIdx = 0
                        modChannel = CheckXPOWChannels[i]

                        if (CheckXPOWChannels[i] >= 41) and (CheckXPOWChannels[i] <= 80):
                            portIdx = 1
                            modChannel = CheckXPOWChannels[i] - 40
                        elif (CheckXPOWChannels[i] >= 81) and (CheckXPOWChannels[i] <= 120):
                            portIdx = 2
                            modChannel = CheckXPOWChannels[i] - 80

                        SendXPOWCommandSingle("CH:" + str(modChannel) + ":VOLT:" + str(round(CurrVoltages[SpectralChannels[i]], 2)), portIdx, XPOWCommandTimeout)
                        time.sleep(VoltageDelay)
                        SendXPOWCommandSingle("CH:" + str(modChannel) + ":VAL?", portIdx, XPOWCommandTimeout)

                        error = False

                    except:
                        errorNum += 1

                        print("\n-- ERROR " + str(errorNum) + " ENCOUNTERED, TRYING AGAIN... --")

                        time.sleep(ErrorDelay)

            else:
                print("Spectral channel " + str(SpectralChannels[i]) + " doesn't need to move.")

        print("\n-- FINISHED LOOP ITERATION " + str(iteration + 1) + " (" + str(round(time.time() - startTime, 2)) + " sec) --")

        time.sleep(IterationDelay)

    # STEP 7: Close devices
    print("\n-- STEP 7: CLOSING DEVICES --")

    OSA.CloseSpectrometer
    ResetAllXPOWChannels()
    ClearXPOWPorts()

    print('\n---- SCRIPT FINISHED ----')

if (__name__ == '__main__'):
    main()
