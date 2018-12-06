# from src.sample import calculateCCC
#
# # data_source = "Training"
# data_source = "Validation"
# model_id = "1542296294_999"
# # pred_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/{}/model_predictions/{}".format(data_source, model_id)
# # gt_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/{}/Annotations".format(data_source)
# pred_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/Validation/model_predictions/1542638847_99"
# gt_dir = "/Users/sonnguyen/Research/OMG_Empathy/data/Validation/model_predictions/1542638847_99_seq"
#
# calculateCCC.calculateCCC(gt_dir, pred_dir)

# import argparse
#
# arg = argparse.ArgumentParser("Run trained model on train/val (validation)")
# arg.add_argument("-m", "--model", help="model's id, e.g., model_1542296294_999", required=True)
# arg.add_argument('-s', '--source', help="data source: train/val", default="val")
#
# args = vars(arg.parse_args())
# print(args)
# print(type(args))
# model_name = args['model']
# model_id = model_name[model_name.index("_") + 1:]
# print(model_id)

# text = '''dropout=0.2
# learning_rate=0.001
# epoch_num=100
# lstm_dim=256
# ln1_output_dim=128
# batch_size=5
# sequence_size=5
# '''
# print(text.replace('\n', '; '))

