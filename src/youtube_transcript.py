
from src import utils
import os


def duration_to_seconds(duration):
    parts = duration.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2].replace(",", "."))
    return s + m * 60 + h * 60 * 60


def write_subtitles(subtitles, output_file):
    size = len(subtitles)
    writer = open(output_file, 'w')
    for i in range(size):
        current_sub = subtitles[i]
        next_sub = None
        if i + 1 < size:
            next_sub = subtitles[i+1]
        ending_time = current_sub[2]
        if next_sub is not None:
            if duration_to_seconds(current_sub[2]) > duration_to_seconds(next_sub[1]):
                ending_time = next_sub[1]

        writer.write("{}\n{} --> {}\n{}\n\n".format(current_sub[0], current_sub[1], ending_time, current_sub[3]))
    writer.flush()
    writer.close()




def fix_ending_time_one_file(input_file, output_file):
    index_tmp = 0
    starting_time = ""
    ending_time = ""
    text = None

    subtitles = []
    flag = 0  # 0: start new, 1: time, 2: text
    with open(input_file, 'r') as reader:
        for line in reader.readlines():
            line = line.strip()
            if line == "":
                # empty line, write the old one
                subtitles.append((index_tmp, starting_time, ending_time, text))
                flag = 0
            elif flag == 0:
                # new sub, this line is index
                index_tmp = line
                flag = 1
            elif flag == 1:
                # timing line
                ps = line.split("-->")
                starting_time = ps[0].strip()
                ending_time = ps[1].strip()
                flag = 2
            elif flag == 2:
                text = line
        if flag == 2:
            # need to add the last one
            subtitles.append((index_tmp, starting_time, ending_time, text))
    write_subtitles(subtitles, output_file)


def fix_ending_time(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        name = filename.replace(" ", "_")
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, name)
        print("Processing for ", filename)
        fix_ending_time_one_file(input_file, output_file)


source = "Testing"
input_dir = "data/{}/transcripts_youtube_raw/".format(source)
output_dir = "data/{}/Transcripts_Youtube/".format(source)
utils.mkdir(output_dir)

fix_ending_time(input_dir, output_dir)
