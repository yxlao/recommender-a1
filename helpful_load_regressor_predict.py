from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy as sp
import cPickle as pickle
import time
from pprint import pprint

from util_feature import predict_helpful

class LoadRegressorPredictWriter():
    def __init__(self, model_path, output_path='predictions_Helpful.txt'):
        self.model_path = model_path
        self.output_path = output_path

    def run(self):
        # load regressor
        opt_regressor = pickle.load(open(self.model_path, 'rb'))

        # load helpful_data.json
        test_data = pickle.load(open('helpful_data.pickle', 'rb'))

        # on test set
        test_helpfuls_predict = [
            predict_helpful(d, opt_regressor.predict) for d in test_data]

        # load 'pairs_Helpful.txt'
        # get header_str and user_item_outofs
        with open('pairs_Helpful.txt') as f:
            # read and strip lines
            lines = [l.strip() for l in f.readlines()]
            # stirip out the headers
            header_str = lines.pop(0)
            # get a list of user_item_ids
            user_item_outofs = [l.split('-') for l in lines]
            user_item_outofs = [[d[0], d[1], float(d[2])] for d in user_item_outofs]

        # make sure `data.json` and `pairs_Helpful.txt` the same order
        for (user_id, item_id, outof), d in zip(user_item_outofs, test_data):
            assert d['reviewerID'] == user_id
            assert d['itemID'] == item_id
            assert d['helpful']['outOf'] == outof

        # write to output file
        f = open(self.output_path, 'w')
        print(header_str, file=f)
        for (user_id, item_id, outof), helpful_predict in zip(user_item_outofs,
                                                              test_helpfuls_predict):
            if helpful_predict > 1800:
                helpful_predict = round(helpful_predict * 1.02)
            print('%s-%s-%s,%s' %
                  (user_id, item_id, int(outof), round(helpful_predict)), file=f)
        f.close()

        print('output:', self.output_path)


if __name__ == "__main__":
    model_path = ""
    output_path='predictions_Helpful.txt'

    predictor = LoadRegressorPredictWriter(model_path, output_path)
    predictor.run()
