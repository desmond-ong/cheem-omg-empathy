

class Utterance():
    def __init__(self, id, start_time, end_time, start_frame, end_frame, text, scores, average):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.text = text
        self.scores = scores
        self.average = average


class UtteranceDiff():
    def __init__(self, id, start_time, end_time, start_frame, end_frame, text, average):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.text = text
        self.average = average  # this is indeed difference, keep same name --> no need to change utils functions
