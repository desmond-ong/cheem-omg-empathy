# Text LSTM model for Empathy Prediction

## Preprocessing
1. Convert Glove embeddings file into word2vec format: 
```bash
python preprocessing.py 0 --glove <glove_file> --word2vec_output <output_file>
```
2. Video to text: we used Youtube to get transcripts for the videos. The transcripts (text + timing) are then mapped with the annotated empathy scores (for training + validation data): 
```bash
python preprocessing.py  1 --transcript <youtube_transcript_dirname> --annotation <groundtruth_annotation> --output_score <mapped_output_dirname>
```
3. Since for Text LSTM model, we trained on the changes of valence, we need to compute the average changes of valence for each two consecutive utterances: 
```bash
python preprocessing.py 2 --output_score <mapped_actualValue_dirname> output_difference <output_dirname>
```
4. Preprocessing for testing data:
```bash
 python preprocessing.py 3 --transcript <youtube_transcript_dirname> --output_score <mapped_output_dirname>
```



## Train the model 
```bash
python3 train.py --chunk <chunk_size> --step <chunk_step> --train <train_mapped_transcrip with difference-scores> --embeddings <glove word2vec-format> --groundtruth <groundtruth annotations>
```

## Test the model on the validation set:

```bash
python3 test.py --model <trained model file> --data <testing data> --output <predictions output dir> --groundtruth <groundtruth annotations dir> --embeddings <glove word2vec-format>
```

## Generate predictions for new dataset (testing):
```bash
python3 model_prediction.py -m <saved model> --data <transcript with scores data> --output <predictions output>
```

