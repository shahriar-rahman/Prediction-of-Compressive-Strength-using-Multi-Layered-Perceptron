import pandas as pd
import missingno as msn
import matplotlib.pyplot as plt

path = '../../data/raw/compressive_strength.xlsx'


class ConstructFeatures:
    def __init__(self):
        self.df_concrete = pd.read_excel(path, index_col=False)

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

    def construct_features(self):
        # Inquire structural integrity
        self.debug_text("• Dataset Statistical Information:", self.df_concrete.describe())
        self.debug_text("• Dataset General Information:", '')
        self.df_concrete.info()

        # Enhance data accessibility
        self.debug_text("• Raw Dataset Columns:", self.df_concrete.columns)

        self.df_concrete.rename(columns={'Cement': 'cement', 'BlastFurnaceSlag': 'blast_furnace_slag', 'Water': 'water',
                                         'Superplasticizer': 'super_plasticisers',
                                         'CoarseAggregate': 'coarse_aggregate',
                                         'FineAggregate': 'fine_aggregate', 'Age': 'age'}, inplace=True)

        self.debug_text("• Modified Dataset Columns:", self.df_concrete.columns)

        # Analyze any Missing values
        null_checker = self.df_concrete.isnull().sum()
        self.debug_text("• Dataframe Null values:", null_checker)

        msn.matrix(self.df_concrete, color=(0.72, 0.32, 0.17), figsize=[17, 20], fontsize=12)
        plt.title("Missingno Matrix for the raw data", fontsize=15, fontweight='bold')
        plt.show()

        # Clean from the data and fix structural errors (i.e. NaN values)
        self.df_concrete.dropna(axis=0, how='any')

        # Find duplicates and perform the De-duplication process
        duplicated_cells = 0
        check_duplicate = self.df_concrete.duplicated()

        for row in check_duplicate:
            if row:
                duplicated_cells += 1

        duplicated_prc = (duplicated_cells / len(check_duplicate)) * 100
        self.debug_text("• Total Cells:", len(check_duplicate))
        self.debug_text("• Duplicated Cells:", duplicated_cells)
        self.debug_text("• Duplicate %:", duplicated_prc)

        self.df_concrete = self.df_concrete.drop_duplicates(subset=None, keep='first', inplace=False,
                                                            ignore_index=False)

        # Validate de-duplication
        duplicated_cells = 0
        check_duplicate = self.df_concrete.duplicated()

        for row in check_duplicate:
            if row:
                duplicated_cells += 1

        self.debug_text("• Number of duplicated cells after de-duplication:", duplicated_cells)

        # Combine similar features
        self.df_concrete['combined_aggregate'] = self.df_concrete['coarse_aggregate']\
                                                 + self.df_concrete['fine_aggregate']

        # Dataframe Information
        self.display_dataframe("• Concrete dataset", self.df_concrete, 20)
        self.data_storage(self.df_concrete)

    def data_storage(self, df):
        # Save modified data to storage
        try:
            df.to_csv('../../data/processed/compressive_strength.csv', sep=',', index=False)

        except Exception as exc:
            self.debug_text("! Exception encountered", exc)

        else:
            text = "• Dataframe successfully saved"
            self.debug_text(text, '...')


if __name__ == "__main__":
    main = ConstructFeatures()
    main.construct_features()
