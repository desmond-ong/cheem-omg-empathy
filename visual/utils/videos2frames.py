## Script for extracting frames from videos ##
## Change base_dir to your own file path ##

import cv2
import time
import os
base_dir = './data/Testing'   ##change this directory for train, val and test
input = os.path.join(base_dir,'Videos')
output = os.path.join(base_dir,'Frames')
if not os.path.isdir(output):
	os.mkdir(output)
videos = os.listdir(input)
num_videos = len(videos)  

def video_to_frames(input_loc, output_loc):

    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print "Number of frames: ", video_length
    count = 0
    print "Converting video..\n"
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print "Done extracting frames.\n%d frames extracted" % count
            print "It took %d seconds forconversion." % (time_end-time_start)
            break
def main():
	for i in range(num_videos):
	    vid_inp = videos[i]
	    vid_id = vid_inp[0:len(vid_inp) - 4]
	    video_loc = os.path.join(input,vid_inp)
	    frame_loc = os.path.join(output,vid_id)
	    if not os.path.exists(frame_loc):
		os.mkdir(frame_loc)
	    video_to_frames(video_loc, frame_loc)

if __name__ == '__main__':
    main()

