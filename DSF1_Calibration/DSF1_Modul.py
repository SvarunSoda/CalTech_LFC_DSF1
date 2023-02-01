#### IMPORTS ####

import sys
import os
import glob
import serial
import time
import clr

clr.AddReference("C:\\Program Files\\Thorlabs\\ThorSpectra\\ThorlabsOSAWrapper.dll")
clr.AddReference("System.Collections")

from ThorlabsOSAWrapper import *
from ThorlabsOSAWrapper import InstrumentModel
from ThorlabsOSAWrapper import AcquisitionUpdateFlag
from ThorlabsOSAWrapper import SpectrumStruct
from ThorlabsOSAWrapper import FileIOInterface
from ThorlabsOSAWrapper import XUnit
from ThorlabsOSAWrapper import YUnit
from System.Collections.Generic import List

#### GLOBAL VARIABLES ####

SweepChannels = [                           # XPOW channels that will be modulated
                #117,    # 1410 - TOPM
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
ExcludedChannels = []                       # XPOW channels that will be excluded from the modulation
TestMode = False                            # If enabled, only spectral channel 1510 will be affected

SweepVoltageStart = 0.0                     # Starting point for voltage modulation
SweepVoltageMax = 17                        # Ending point for voltage modulation
SweepVoltageStep = 0.25                     # Voltage steps at which the modulation occurrs
SweepCurrentMax = 50                        # (mA) Max current applied to all 120 XPOW channels

OSASensitivity = 0                          # Sensitivity setting of OSA
OSAResoltion = 0                            # Resolution setting of OSA
#AverageNum = 5
DiscardedSpectrumNum = 2                    # Number of discarded spectrums prior to start of modulation
SaveIndSpectrums = True                     # Save a spectrum at each individual voltage increment
SaveAllSpectrums = True                     # Save all voltage steps in singular file(s) for each channel

XPOWPorts = ["COM36", "COM38", "COM35"]     # (D [1 (1 - 40), 2 (41 - 80), 3 (81 - 120)]) USB ports that XPOW is connected to
XPOWBaudRate = 115200                       # Baud rate used by XPOW
XPOWResetInterval = 0                       # Reading interval between reset of XPOW COM ports
XPOWCommandTimeout = 15                     # (1/10 sec) Timeout after command sent to XPOW COM port
XOPWCommandDelay = 0.1                      # (sec) Delay after each command sent to XPOW
SpectrumFilePrefix = "DSF1_LookupData_2"    # Prefix of spectrum filename(s) 
SpectrumFilePath = "DSF1_LookupData_2\\"    # Folder path of saved spectrum file(s)

VoltageDelay = 1                            # (sec) Delay after each XPOW voltage setting
SpectrumCheckDelay = 2                      # (sec) Interval between checks for an acquired OSA spectrum
ErrorDelay = 1                              # (sec) Delay between re-tries of sending XPOW commands of error encountered

## DO NOT EDIT ##

SerialLines = []
FileInterface = FileIOInterface()
CurrSpectra = List[SpectrumStruct]()
ChannelsScanned = 0
ChannelSpectrums = 0
CurrChannel = 0
CurrVoltage = 0
CurrSpectraNum = 0
SpectraFilesGenerated = 0
CurrAverageNum = 0
StartTime = 0
SaveSpectrum = False
WaitingSpectrum = False

if (SweepCurrentMax > 300):
    SweepCurrentMax = 300

if (TestMode == True):
    SweepChannels = [29]

#### OSA FUNCTIONS ####

def OnSingleAcquisition(sender, event_args):
    global CurrSpectra
    global CurrSpectraNum
    global SpectraFilesGenerated
    global CurrAverageNum
    global WaitingSpectrum

    if (event_args.CallbackMessage.LastDataTypeUpdateFlag == AcquisitionUpdateFlag.Spectrum) and (len(SpectrumFilePrefix) > 0):
        delay = round(time.time() - StartTime, 2)

        spectrum = SpectrumStruct(sender.ChannelInterfaces[0].GetLastSpectrumLength(), True)
        
        spectrum.XAxisUnit = int(XUnit.nmAir)
        spectrum.YAxisUnit = int(YUnit.dB)
        
        sender.ChannelInterfaces[0].GetLastSpectrum(spectrum)

        if (SaveSpectrum == True):
            print("Spectrum Acquired! (" + str(delay) + " sec)")

            spectrumFilePath = SpectrumFilePath

            if (SpectrumFilePath == ""):
                spectrumFilePath = os.getcwd() + "\\"

            if (SaveIndSpectrums == True):
                spectrumFileNum = "_Ch" + str(CurrChannel) + "_" + str(round(CurrVoltage, 2)) + "V"

                FileInterface.WriteSpectrum(spectrum, str(spectrumFilePath + SpectrumFilePrefix + spectrumFileNum + ".txt"), 1)
                FileInterface.WriteSpectrum(spectrum, str(spectrumFilePath + SpectrumFilePrefix + spectrumFileNum + ".spf2"), 2)

            if (SaveAllSpectrums == True):
                CurrSpectra.Add(spectrum)
                CurrSpectraNum += 1

                if (CurrSpectraNum == 26) or (CurrVoltage >= SweepVoltageMax):
                    allSpectrumFileNum = "_Ch" + str(CurrChannel) + "_all_" + str(SpectraFilesGenerated + 1)

                    FileInterface.WriteSpectra(CurrSpectra, str(spectrumFilePath + SpectrumFilePrefix + allSpectrumFileNum + ".spf2"), 2)

                    CurrSpectra.Clear()
                    CurrSpectraNum = 0
                    SpectraFilesGenerated += 1

                    print("Spectra Saved!")

        CurrAverageNum = 0
        WaitingSpectrum = False

    elif (event_args.CallbackMessage.LastDataTypeUpdateFlag == AcquisitionUpdateFlag.Average):
        if (SaveSpectrum == True):
            CurrAverageNum += 1
            print("Spectrum Averaged " + str(CurrAverageNum) + " times.")

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
    channel = 1

    while (channel <= 120):
        modChannel = channel

        if (channel in ExcludedChannels):
            channel += 1

            continue
        
        portIdx = 0

        if (channel >= 41) and (channel <= 80):
            modChannel = channel - 40
            portIdx = 1
        elif (channel >= 81) and (channel <= 120):
            modChannel = channel - 80
            portIdx = 2

        SendXPOWCommandSingle("CH:" + str(modChannel) + ":VOLT:0", portIdx, XPOWCommandTimeout)
        time.sleep(VoltageDelay)
        SendXPOWCommandSingle("CH:" + str(modChannel) + ":CUR:" + str(SweepCurrentMax), portIdx, XPOWCommandTimeout)

        channel += 1

#### MAIN FUNCTION ####

def main():
    print('---- SCRIPT STARTED ----')

    # STEP 1: Connect to OSA
    print("\n-- STEP 1: CONNECTING OSA --")

    osa = 0
    locator = DeviceLocator()
    osaNum = locator.InitializeSpectrometers()
    
    print("OSAs found: " + str(osaNum))
    
    if (osaNum > 0):
        print("Opening first OSA...")

        osa = LibDeviceInterface(0)

    else:
        print("No OSA Found!")

        return

    osa.SetSensitivityMode(OSASensitivity) 
    osa.SetResolutionMode(OSAResoltion) 
    osa.AcquisitionSettings.AutomaticGain = True
    
    #if (AverageNum > 0):
    #    osa.AcquisitionSettings.RollingAverage = True
    #    osa.AcquisitionSettings.AverageSpecNum = AverageNum

    osa.OnSingleAcquisition += OnSingleAcquisition

    # STEP 2: Connect to XPOW
    print("\n-- STEP2: CONNECTING XPOW --")

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

    global ChannelsScanned
    global ChannelSpectrums
    global CurrChannel
    global CurrVoltage
    global StartTime
    global SaveSpectrum
    global WaitingSpectrum
    global SpectraFilesGenerated

    # STEP 3: Discard OSA spectrums
    print("\n-- STEP 3: DISCARDING " + str(DiscardedSpectrumNum) + " SPECTRUMS --")

    SaveSpectrum = False

    for j in range(DiscardedSpectrumNum):
        osa.AcquireSingleSpectrum()
        WaitingSpectrum = True

        while (WaitingSpectrum == True):
            time.sleep(SpectrumCheckDelay)

        print("Discarded spectrum " + str(j + 1) + "...")

    SaveSpectrum = True

    # STEP 4: Modulation loop
    print("\n-- STEP 4: STARTING MODULATION LOOP --")

    for j in range(len(SweepChannels)):
        channel = SweepChannels[j]
        CurrChannel = channel
        ChannelSpectrums = 0

        if (channel in ExcludedChannels):
            continue

        print("\n-- MODULATING CHANNEL " + str(channel) + " --\n")

        modChannel = None
        portIdx = None

        if (channel >= 1) and (channel <= 40):
            modChannel = channel
            portIdx = 0
        elif (channel >= 41) and (channel <= 80):
            modChannel = channel - 40
            portIdx = 1
        elif (channel >= 81) and (channel <= 120):
            modChannel = channel - 80
            portIdx = 2
        else:
            print("WARNING: Invalid modulated channel ID (" + str(channel) + ")!")

            continue
            
        currVoltage = SweepVoltageStart
        resetTracker = 0
        errorNum = 0

        while (currVoltage <= SweepVoltageMax):
            try:
                if (resetTracker >= XPOWResetInterval):
                    ResetXPOWPorts()
                    resetTracker = 0
                else:
                    resetTracker += 1

                CurrVoltage = currVoltage

                SendXPOWCommandSingle("CH:" + str(modChannel) + ":VOLT:" + str(currVoltage), portIdx, XPOWCommandTimeout)
                time.sleep(VoltageDelay)
                SendXPOWCommandSingle("CH:" + str(modChannel) + ":VAL?", portIdx, XPOWCommandTimeout)

                print("Spectrum Requested!")
                osa.AcquireSingleSpectrum()
                StartTime = time.time()
                WaitingSpectrum = True

                while (WaitingSpectrum == True):
                    time.sleep(SpectrumCheckDelay)
                    
                currVoltage += SweepVoltageStep
                ChannelSpectrums += 1

            except:
                errorNum += 1

                print("\n-- ERROR " + str(errorNum) + " ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(ErrorDelay)

        print("\n-- RESETING MODULATED CHANNEL VOL TO 0 --")

        finalError = True

        while (finalError == True):
            try:
                ResetXPOWPorts()

                SendXPOWCommandSingle("CH:" + str(modChannel) + ":VOLT:0", portIdx, XPOWCommandTimeout)
                time.sleep(VoltageDelay)
                SendXPOWCommandSingle("CH:" + str(modChannel) + ":VAL?", portIdx, XPOWCommandTimeout)

                finalError = False

            except:
                print("\n-- ERROR ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(ErrorDelay)

        SpectraFilesGenerated = 0
        ChannelsScanned += 1

    # STEP 5: Closing devices
    print("\n-- STEP 5: CLOSING DEVICES --")

    osa.CloseSpectrometer
    ClearXPOWPorts()

    print('\n---- SCRIPT FINISHED ----')

if (__name__ == '__main__'):
    main()
