# Visual Model for Empathy Prediction

Train a VGG-LSTM for predicting empathy on the OMG-Empathy Prediction Dataset

## Frames and Faces from OMG-Video dataset

```bash
pip install face_recognition
```
1. Run utils/videos2frames.py to extract frames from videos (25 frames per second).
2. Run utils/faces_extractor.py to extract subject and actor faces from extracted frames. 

## VGG-Face Features Extraction

We use the pre-trained VGG face model to extract 4096-d vector for the subject and actor faces.

1. Download the pre-trained [VGG-Face caffe model](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) and convert it into PyTorch model using this [tool](https://github.com/fanq15/caffe_to_torch_to_pytorch). 
2. Run feature_extraction/extract_face_features.py to extract fc7 features.
3. Average features over 1 second (25 frames) by running the script feature_extraction/avgvisual_rawfeats_25fps.py.
4. Match the average feature dimensions to total number of frames by feature_extraction/sanity_check_visualfeats.py.

## Training a VGG-LSTM Model on the OMG-Empathy dataset

We use train.py to train our VGG-LSTM model using the 1 second average subject-features (extracted in 4. above) and test.py to calculate the final CCC and generate predictions.

Please make sure to update paths to directories as appropriate.
