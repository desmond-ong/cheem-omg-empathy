"""Extracts audio features using the OpenSMILE binary.

Requires: SMILExtract binary, FFMPEG
"""

import subprocess
import csv
import os, shutil
import scipy
import numpy
import scipy.io.wavfile

OPENSMILE_DIRECTORY = "./"
OPENSMILE_CONFIG = "./emobase_mod.conf"

def calcFeatures(signal, sampleRate, outputFeatureFilename, winSize=1.0):
    tempWAVFilename = "./tmp/tempWAVFilename.wav"
    tempOutputFilename = "./tmp/tempFeaturesOutput.csv"

    windowWidth = (int)(winSize * sampleRate) 
    signalLength = len(signal)
    numWindows = (int)(signalLength/windowWidth)  # Take the floor
    
    for j in xrange(numWindows):
        print("audioProcessing: Processing Window Number " + str(j+1) +
              " out of " + str(numWindows))
        sample = signal[(j*windowWidth):((j+1)*windowWidth)]
        scipy.io.wavfile.write(filename=tempWAVFilename,
                               rate=sampleRate, data=sample)

        command = "{}SMILExtract -C {} -I {} -O {}".\
                  format(OPENSMILE_DIRECTORY, OPENSMILE_CONFIG,
                         tempWAVFilename, tempOutputFilename)
        subprocess.call(command, shell=True)
        with open(tempOutputFilename, 'rb') as inputFile:
            thisReader = csv.reader(inputFile, delimiter = ",")
            rowNum = 0
            for thisRow in thisReader:
                if j==0 and rowNum==0:
                    with open(outputFeatureFilename,'w') as outputReader:
                        outputReader.write(', '.join(thisRow))
                        outputReader.write("\n")
                if rowNum > 0:
                    # Increment frameIndex field by j
                    thisRow[0] = str(int(thisRow[0]) + j)
                    # Increment frameTime field by j*winSize
                    thisRow[1] = str(float(thisRow[1]) + j*winSize) 
                    with open(outputFeatureFilename,'a') as outputReader:
                        # Write feature row to outputFeatureFilename
                        outputReader.write(', '.join(thisRow)) 
                        outputReader.write("\n")
                rowNum+=1
    return

def readInWAVFile(filename, convertToMono=True):
    '''
    processes an audio file in .WAV format to a numpy matrix

    @input:
        filename    : (string) path of input .wav file.

    @return:
        sampleRate    : (int)    .wav file sampling rate, in Hz
        signal        : the raw amplitude data
    '''
    try:
        [sampleRate, signal] = scipy.io.wavfile.read(filename)
    except IOError:
        print "Error: file not found or other I/O error."
        return (-1, -1)
    if convertToMono:
        if signal.ndim == 2:
            # if signal is stereo, average the two channels
            signal = (signal[:, 0] + signal[:, 1]) / 2.0
    return (signal, sampleRate)


def convertMP4(inputFilename, outputFilename=None,
               loglevel="quiet", toFormat=".wav"):
    '''
    strips the audio file from MP4, converts to WAV,
    using ffmpeg (and a subprocess.call)

    @input:
        inputFilename    : (string) path of input .mp4 file.
        outputFilename    : (string; optional) path of output .wav file.
        loglevel         : (string; optional) how verbose do you want
                           the ffmpeg call to be? Default: quiet

    @return:
        None
    '''
    if toFormat not in ["wav", "mp4", "mp3"]:
        print "Error. toFormat not recognized, defaulting to .wav"
        toFormat = "wav"
    if outputFilename is None:
        outputFilename = inputFilename[:-4] + "." + toFormat
    if loglevel not in ["quiet", "fatal", "error",
                        "warning", "info", "verbose", "debug"]:
        print "Error. loglevel not recognized, defaulting to quiet."
        loglevel = "quiet"
    # Note, without this flag, info is the default
    loglevelFlag = "-loglevel " + loglevel + " "  
    try:
        # Use -ac 1 : only 1 audio channel -vn : no video
        command = ("ffmpeg -i " + inputFilename + " -ac 1 -vn "
                   + loglevelFlag + outputFilename)
        print("Starting audio extraction operation on " + inputFilename)
        subprocess.call(command, shell=True)
        print("Wrote output .wav file to " + outputFilename)
    except IOError:
        print "Error: file not found or other I/O error."
    return

def extractFeatures(input_dir, output_dir, win_size=1.0):
    winSize = win_size

    for d in ['./tmp', output_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    for thisFilename in os.listdir(input_dir):
        if thisFilename.endswith(".mp4"):
            tempWAVFilename2 = "./tmp/temp_" + thisFilename + ".wav"
            convertMP4(
                inputFilename=os.path.join(input_dir, thisFilename),
                outputFilename=tempWAVFilename2)
            print("Processing " + thisFilename)
            outputFeatureFilename = os.path.join(
                output_dir,
                thisFilename[:-4] + "_acousticFeatures.csv")
            signal, sampleRate = readInWAVFile(tempWAVFilename2)
            audioFeatureVectors = calcFeatures(
                signal=signal, sampleRate=sampleRate,
                winSize = winSize,
                outputFeatureFilename = outputFeatureFilename)

    # Clear temporary files and folders once done
    shutil.rmtree('./tmp')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help='input directory',
                        default='../data/Testing/Videos')
    parser.add_argument('--out_dir', type=str, help='output directory',
                        default='../data/Testing/OpenSMILE')
    parser.add_argument('--win_size', type=float, default=1.0,
                        help='window size in seconds (default: 1.0)')
    args = parser.parse_args()
    
    extractFeatures(args.in_dir, args.out_dir, args.win_size)
