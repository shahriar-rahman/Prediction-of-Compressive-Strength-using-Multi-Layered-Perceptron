import os
import sys
import pandas as pd
import matplotlib
from sklearn import preprocessing
sys.path.append(os.path.abspath('../visualization'))
import visualize
matplotlib.style.use('fivethirtyeight')
path = '../../data/processed/compressive_strength.csv'


class FeatureTransformation:
    def __init__(self):
        # Object Instantiation
        self.df_concrete = pd.read_csv(path, index_col=False)
        self.visualize = visualize.Visualize()

    @staticmethod
    def display_dataframe(name, df, contents):
        table = df.head(contents)
        print('\n')
        print("=" * 150)
        print("◘ ", name, " Dataframe:")
        print(table.to_string())
        print("=" * 150)

    @staticmethod
    def debug_text(title, task):
        print('\n')
        print("=" * 150)
        print('◘ ', title)

        try:
            print(task)

        except Exception as exc:
            print("! ", exc)

        finally:
            print("=" * 150)

    def feature_transformation(self):
        df_columns = ['compressive_strength', 'cement', 'blast_furnace_slag', 'water', 'super_plasticisers',
                      'coarse_aggregate', 'fine_aggregate', 'age', 'combined_aggregate']

        super_title = "Original Scaling Distribution"
        self.visualize.plot_multi_kde(self.df_concrete, df_columns, super_title)

        super_title = "Robust, Standard, and Min-Max Scaler distribution"
        # Robust Scaler
        scaler = preprocessing.RobustScaler()
        df_robust = scaler.fit_transform(self.df_concrete)
        df_robust = pd.DataFrame(df_robust, columns=df_columns)

        # Standard Scaler
        scaler = preprocessing.StandardScaler()
        df_standard = scaler.fit_transform(self.df_concrete)
        df_standard = pd.DataFrame(df_standard, columns=df_columns)

        # Min-Max Scaler
        scaler = preprocessing.MinMaxScaler()
        df_minmax = scaler.fit_transform(self.df_concrete)
        df_minmax = pd.DataFrame(df_minmax, columns=df_columns)

        self.visualize.plot_compare_kde(df_robust, df_standard, df_minmax, df_columns, super_title)
        self.display_dataframe("Min-Max Scaler", df_minmax, 20)
        self.data_storage(df_minmax)

    def data_storage(self, df):
        # Save the transformed-intermediate data to storage
        try:
            df.to_csv('../../data/interim/compressive_strength.csv', sep=',', index=False)

        except Exception as exc:
            self.debug_text("! Exception encountered", exc)

        else:
            text = "Dataframe successfully saved"
            self.debug_text(text, '...')


if __name__ == "__main__":
    main = FeatureTransformation()
    main.feature_transformation()
