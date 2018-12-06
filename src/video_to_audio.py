
import os
from ffmpeg import video
import subprocess
import tempfile



def getLength(input_video):
    result = subprocess.Popen('ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(input_video),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.communicate()
    return output[0]


def get_duration(input_video):
    tmpf = tempfile.NamedTemporaryFile()
    os.system("ffmpeg -i \"%s\" 2> %s" % (input_video, tmpf.name))
    lines = tmpf.readlines()
    tmpf.close()
    for line in lines:
        l = "{}".format(line.strip())
        print(l)
        if l.startswith("b'Duration:"):
            start = l.index(":")
            t = l[start + 1: l.index(",")]
            return t.strip()


def duration_to_seconds(duration):
    parts = duration.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return s + m * 60 + h * 60 * 60


def main():
    input_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/Training/Videos/"
    audio_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/Training/Audios/"

    for filename in os.listdir(input_dir):
        if not filename.endswith(".mp4"):
            continue
        name = filename[:-4]
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(audio_dir, name + ".wav")

        command = "ffmpeg  -i {} {}".format(input_file, output_file)
        os.system(command)

# input_video = "/Users/sonnguyen/Research/OMG_Empathy/data/Training/Videos/Subject_1_Story_2.mp4"
# # len = getLength(input_video)
#
# t = get_duration(input_video)
# print(t)
# print(duration_to_seconds(t))
