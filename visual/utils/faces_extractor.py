## Extract Subject and Actor faces locations and landmarks from frames ##
## Crop the faces separately using the face locations ##
## sudo pip install face_recognition ##

import face_recognition
import os
import time
import math
from face_obj import FaceLoc
import json
import matplotlib.pyplot as plt
from PIL import Image
import cv2

base_dir = './data/Testing'   ##change this directory for train, val and test

def mkdir(input_dir):
    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)

def time_since(start):
    s = time.time() - start
    m = math.floor(s / 60)
    s = s - m * 60
    return "{}m {:.0f}s".format(m, s)

def get_loc_landmarks(image_path, frame_name, face_locs, sub_face_dir, actor_face_dir):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model='cnn')

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for i in range(len(face_locations)):
        loc = face_locations[i]
        landmarks = face_landmarks_list[i]
        face_encoding = face_encodings[i]
	# Face boundary coordinates for cropping 
        top = loc[0]    
        right = loc[1]
        bottom = loc[2]
        left = loc[3]
      	## Subject face location
        if (loc[1] > 1080):
            subject = image[top:bottom, left:right]
            cv2.imwrite((os.path.join(sub_face_dir,frame_name+'.jpg')),cv2.cvtColor(subject, cv2.COLOR_RGB2BGR))
	## Actor face location
        else:
            actor = image[top:bottom, left:right]
            cv2.imwrite((os.path.join(actor_face_dir, frame_name + '.jpg')),cv2.cvtColor(actor, cv2.COLOR_RGB2BGR))
    
        face_loc = FaceLoc(frame_name, i + 1, loc, landmarks)
        face_locs.append(face_loc)

input_root_dir = os.path.join(base_dir,"Frames/")
faces_output_root_dir = os.path.join(base_dir,"Faces_Loc/" )
extract_faces_output_root_dir = os.path.join(base_dir,"Extracted_Faces/")
mkdir(faces_output_root_dir)
mkdir(extract_faces_output_root_dir)

def main():
	start = time.time()
	print("Time: {}".format(time_since(start)))

	start = time.time()

	for data_source in os.listdir(input_root_dir):
	    input_dir = os.path.join(input_root_dir, data_source)
	    if not os.path.isdir(input_dir):
		continue
	    print("Processing for {}\n".format(input_dir))
	    faces_output_dir = os.path.join(faces_output_root_dir, data_source)
	    mkdir(faces_output_dir)
	    extracted_faces_dir = os.path.join(extract_faces_output_root_dir , data_source)
	    mkdir(extracted_faces_dir)
	    sub_face_dir = os.path.join(extracted_faces_dir, 'Subject')
	    mkdir(sub_face_dir)
	    actor_face_dir = os.path.join(extracted_faces_dir, 'Actor')
	    mkdir(actor_face_dir)
	    for frame in os.listdir(input_dir):

		start_frame = time.time()
		print("Processing for frames {}".format(frame))
		frame_input= os.path.join(input_dir, frame)
		output_file = os.path.join(faces_output_dir, frame[0:len(frame)-4] + ".json")
		face_locs = []

		print("\tAnalyzing frame {}".format(frame_input))
		get_loc_landmarks(frame_input, frame[0:len(frame)-4], face_locs, sub_face_dir, actor_face_dir)

		print("Writing details info to", output_file)
		with open(output_file, 'w') as writer:
		    json.dump([face_loc.__dict__ for face_loc in face_locs], writer, indent=3)
		print("Time {}".format(time_since(start_frame)))


	print("Total time: ", time_since(start))



if __name__ == '__main__':
    main()

