import pandas as pd
import numpy as np
import os
import gc
import argparse

import luigi

import pickle
from datetime import datetime

import catboost
from catboost import CatBoostClassifier


gc.enable()
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--test', required=False, help='path to test data', default='data_test.csv')
parser.add_argument('-f', '--features', required=False, help='path features', default='features.pkl')
parser.add_argument('-m', '--model', required=False, help='model path and name', default='cat_red')
parser.add_argument('-t', '--threshold', required=False, help='prediction threshold', default='0.6113188028649806')
args = vars(parser.parse_args())
test_path = args['test']
features = args['features']
model_path = args['model']
threshold = float(args['threshold'])


def reduce_memory_df(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)[:4] != 'uint' and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif str(col_type)[:4] != 'uint':
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


class Transformer(luigi.Task):
    path_data_train = luigi.Parameter()
    path_features = luigi.Parameter()
    use_features = ['week_no', 'timedelta', 'num_proposals', 'day', 'vas_id', '193', '3', '207', '167', '59',
                    'buy_time']

    def output(self):
        return luigi.LocalTarget('prepared_data_test.csv')

    def _prepare(self, data, features):
        data = data.copy()
        data['buy_time'] = data['buy_time'].astype(np.uint32)
        data['id'] = data['id'].astype(np.uint64)
        features = features[features['id'].isin(data['id'].unique())]

        data = data.sort_index()
        data = data.sort_values(by='buy_time')
        features = features.sort_values(by='buy_time')

        data['date'] = pd.to_datetime(data['buy_time'], unit='s')
        features['buy_time_cp'] = features['buy_time'].copy()

        data_test = pd.merge_asof(data, features, on='buy_time', by='id', direction='nearest')

        data_test['day'] = data_test['date'].dt.day

        data_test['num_proposals'] = data_test.groupby('id')['vas_id'].transform('count')
        with open('prop_dict.pkl', 'rb') as infile:
            prop_dict = pickle.load(infile)

        data_test['num_proposals'] = data_test.apply(lambda row: row['num_proposals'] + prop_dict.get(row['id'], 0),
                                                     axis=1)

        data_test['week_no'] = data_test['day'].apply(lambda x: x // 7 + 1)

        data_test.set_index('id', inplace=True)

        data_test['buy_time_d'] = [datetime.fromtimestamp(x) for x in data_test['buy_time']]
        data_test['buy_time_cp'] = [datetime.fromtimestamp(x) for x in data_test['buy_time_cp']]
        data_test['timedelta'] = (data_test['buy_time_d'] - data_test['buy_time_cp']).dt.days

        data_test.drop(['date', 'buy_time_cp', 'buy_time_d'], axis=1, inplace=True)

        data_test = data_test[self.use_features]
        data_test = reduce_memory_df(data_test)

        return data_test

    def run(self):
        df = pd.read_csv(self.path_data_train).drop('Unnamed: 0', axis=1)
        features = pd.read_pickle(self.path_features).reset_index()
        df = self._prepare(df, features)
        del features

        with self.output().open('w') as f:
            f.write(df.to_csv(index=True, encoding='utf-8', float_format='%.10f'))
        gc.collect()


class Forecaster(luigi.Task):
    path_data_train = luigi.Parameter()
    path_features = luigi.Parameter()
    model_name = luigi.Parameter()
    threshold = luigi.Parameter(default=0.6113188028649806)

    def output(self):
        return luigi.LocalTarget('luigi_predictions.csv')

    def requires(self):
        return Transformer(self.path_data_train, self.path_features)

    def run(self):
        df = pd.read_csv(self.input().open('r'), index_col='id')
        model = CatBoostClassifier().load_model(self.model_name)

        probas = catboost.CatBoost.predict(model, df.drop('buy_time', axis=1), prediction_type='Probability')[:, 1]
        preds = (probas >= self.threshold).astype(int)
        df['target'] = preds
        df.reset_index(inplace=True)

        with self.output().open('w') as f:
            f.write(df[['id', 'vas_id', 'buy_time', 'target']].to_csv(index=False, encoding='utf8'))
        os.remove('prepared_data_test.csv')


if __name__ == '__main__':
    luigi.build([Forecaster(test_path, features, model_path, threshold)])
