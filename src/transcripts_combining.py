import os
import collections

input_dir = "data/Training/Transcripts_10/"
output_dir = "data/Training/Transcripts_10_combined/"

transcripts = {}  # {video_id: {id: content}}

def get_sorted_keys(parts):
    keys = []
    for key in sorted(parts.iterkeys()):
        keys.append(key)
    return keys


def combine_content(sorted_tuples):
    content = ""
    for key, value in sorted_tuples.items():
        content = "{}\n({}) {}".format(content, key, value)
    return content.strip()


for filename in os.listdir(input_dir):
    if not filename.endswith(".txt"):
        continue
    index = filename.rfind("_")
    name = filename[:index]
    part_id = int(filename[index+1:-4])

    with open(os.path.join(input_dir, filename), 'r') as reader:
        content = reader.read()

    if name in transcripts:
        transcripts[name][part_id] = content
    else:
        transcripts[name] = {part_id: content}

for key, value in transcripts.items():
    output_file = os.path.join(output_dir, "{}.txt".format(key))
    od = collections.OrderedDict(sorted(value.items()))
    # print(od)
    with open(output_file, 'w') as writer:
        writer.write(combine_content(od))

