
import speech_recognition as sr
import ffmpeg
from ffmpeg import video
import os
import time
from src import utils


audio_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/Training/Audios_shorts_10/"
transcript_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/Training/Transcripts_10/"


def audio_to_text_for_file(audio_file, output_file, r):
    audiofile = sr.AudioFile(audio_file)
    with audiofile as source:
        audio = r.record(source)
    writer = open(output_file, 'w')
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        transcript = r.recognize_google(audio)

    except sr.UnknownValueError:
        print("!!!Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("!!! Could not request results from Google Speech Recognition service; {0}".format(e))
    writer.write(transcript)
    writer.flush()
    writer.close()
    # transcript = r.recognize_sphinx(audio)
    # print(transcript)



def audio_to_text(audio_dir, transcript_dir, r):
    for filename in os.listdir(audio_dir):
        if not filename.endswith(".wav"):
            continue
        if not filename.startswith("Subject_1_Story_2"):
            continue
        print("Processing for ", filename)
        name = filename[:-4]
        input_file = os.path.join(audio_dir, filename)
        output_file = os.path.join(transcript_dir, name + ".txt")
        if os.path.isfile(output_file):
            continue
        start_local = time.time()
        audio_to_text_for_file(input_file, output_file, r)
        print("Time:", utils.time_since(start_local))
        # time.sleep(30)


r = sr.Recognizer()
start = time.time()
audio_to_text(audio_dir, transcript_dir, r)

# input_file = "/Users/sonnguyen/Downloads/harvard.wav"
# output_file = "/Users/sonnguyen/Downloads/harvard.txt"
# for i in range(1000):
#     print(i)
#     audio_to_text_for_file(input_file, output_file, r)

print("Total time: ", utils.time_since(start))


# audio_file = "/Users/sonnguyen/Downloads/harvard.wav"
# harvard = sr.AudioFile(audio_file)
#
# with harvard as source:
#     audio = r.record(source)
#
# s = r.recognize_google(audio)
# print(type(audio))
# print(s)