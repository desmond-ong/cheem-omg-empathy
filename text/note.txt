preprocessing.py: preprocessing data

train.py: train the model using training set
python3 train.py --chunk 10 --step 5 --train data/Training/transcripts_with_scores_difference/ --embeddings data/glove.840B.300d.filtered.word2vec --groundtruth data/Training/Annotations/

test.py: evaluate the trained model on validation set
python3 test.py --model model_1543994032_best.pt --data data/Training/transcripts_with_scores/ --output data/Training/model_predictions/ --groundtruth data/Training/Annotations/ --embeddings data/glove.840B.300d.filtered.word2vec

model_prediction.py: perform predictions for the testing set
python3 src/text/model_prediction.py -m data/Training/models/model_1544087790_Story_1_4850.pt --data data/Testing/transcripts_with_scores/ --output data/Testing/tmp_predictions/

