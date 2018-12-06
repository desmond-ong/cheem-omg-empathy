from pydub import AudioSegment
import wave


import subprocess
import re
import os



def get_duration(audio_file):
    # valid for any audio file accepted by ffprobe
    args = ("ffprobe", "-show_entries", "format=duration", "-i", audio_file)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = popen.communicate()
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(output))
    return float(match.group())


def cut_audio_start_end(audio_file, start, end, output_file):
    command = "ffmpeg -ss {} -to {} -i {} {}".format(start, end, audio_file, output_file)
    # print(command)
    os.system(command)


def cut_audio(audio_file, audio_name, output_dir, duration=20):
    start = 0
    end = get_duration(audio_file)

    count = 0

    current_end = start + duration
    while True:
        if current_end > end:
            current_end = end
        cut_audio_start_end(audio_file, start, current_end, os.path.join(output_dir, "{}_{}.wav".format(audio_name, count)))

        count += 1
        start = current_end
        current_end = start + duration

        if start >= end:
            break


def cut_audio_for_dir(audio_dir, output_dir, duration=20):
    for file in os.listdir(audio_dir):
        if not file.endswith(".wav"):
            continue

        print("Processing file ", file)
        name = file[:-4]
        cut_audio(os.path.join(audio_dir, file), name, output_dir, duration)


input_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/Training/Audios/"
output = "/Users/sonnguyen/Research/OMG_Empathy/data/Training/Audios_shorts_10/"

cut_audio_for_dir(input_dir, output, 10)