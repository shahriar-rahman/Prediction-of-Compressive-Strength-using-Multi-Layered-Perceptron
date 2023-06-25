import os
import sys
import pandas as pd
sys.path.append(os.path.abspath('../visualization'))
import visualize
path = '../../data/processed/compressive_strength.csv'


class FeatureAnalysis:
    def __init__(self):
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

    def feature_analysis(self):
        text = "• Processed data:"
        self.debug_text(text, self.df_concrete['compressive_strength'].describe())

        # Histogram for inspecting Compressive Strength
        bins = 20
        title = "Distribution of Compressive Strength"
        x_label = "Strength (MPa)"
        y_label = 'Frequency'

        self.visualize.plot_histogram(self.df_concrete['compressive_strength'], bins, title, x_label, y_label)

        # Histogram for the inspection of features
        bins = 15
        d1 = self.df_concrete['cement']
        d2 = self.df_concrete['blast_furnace_slag']
        d3 = self.df_concrete['water']
        d4 = self.df_concrete['super_plasticisers']
        d5 = self.df_concrete['coarse_aggregate']
        d6 = self.df_concrete['fine_aggregate']
        d7 = self.df_concrete['age']
        self.visualize.plot_multi_histogram(d1, d2, d3, d4, d5, d6, d7, bins)

        # Quantity of cement where strength is greater than mean value
        high_strength = self.df_concrete[self.df_concrete['compressive_strength'] >
                                         self.df_concrete['compressive_strength'].mean()]
        df = high_strength[['cement', 'compressive_strength']].sort_values(by='compressive_strength', ascending=False)

        # Bar Plot and Covariance Matrix
        title = "Cement vs High Compressive Strength"
        x_label = "Compressive Strength (MPa)"
        y_label = "Cement (kg in a m^3 mixture)"

        self.display_dataframe('df', df, 25)
        self.visualize.plot_bar(df['compressive_strength'].iloc[:150], df['cement'].iloc[:150], title, x_label, y_label)
        self.visualize.covariance_matrix(df, "Covariance Matrix: Cement vs High Compressive Strength")

        # Blast furnace effect on Compressive strength (Combined)
        df = self.df_concrete[['blast_furnace_slag', 'compressive_strength']].sort_values(by='compressive_strength',
                                                                                          ascending=False)

        # Scatter Plot and Covariance Matrix
        title = "Blast Furnace Slag vs High Compressive Strength"
        x_label = "Compressive Strength (MPa)"
        y_label = "Blast Furnace Slag (kg in a m^3 mixture)"
        self.display_dataframe('df', df, 25)
        self.visualize.plot_scatter(df['compressive_strength'].iloc[:150], df['blast_furnace_slag'].iloc[:150],
                                    title, x_label, y_label)

        df = high_strength[['blast_furnace_slag', 'compressive_strength']].sort_values(by='compressive_strength',
                                                                                       ascending=False)
        self.visualize.covariance_matrix(df, "Covariance Matrix: Blast furnace vs Compressive strength")

        # How much water is required to acquire adequate Strength
        df = high_strength[['water', 'compressive_strength']].sort_values(by='compressive_strength',
                                                                          ascending=False)
        # Bar Plot and Covariance Matrix
        title = "Water vs High Compressive Strength"
        x_label = "Compressive Strength (MPa)"
        y_label = "Water (kg in a m^3 mixture)"

        self.display_dataframe('df', df, 25)
        self.visualize.plot_bar(df['compressive_strength'].iloc[:150], df['water'].iloc[:150], title, x_label, y_label)
        self.visualize.covariance_matrix(df, "Covariance Matrix -Water vs High Compressive Strength")

        # How Super Plasticisers affect Compressive strength (combined)
        df = self.df_concrete[['super_plasticisers', 'compressive_strength']].sort_values(by='compressive_strength',
                                                                                          ascending=False)

        # Scatter Plot and Covariance Matrix
        title = "Super Plasticisers vs Compressive Strength"
        x_label = "Compressive Strength (MPa)"
        y_label = "Super Plasticisers (kg in a m^3 mixture)"

        self.display_dataframe('df', df, 25)
        self.visualize.plot_scatter(df['compressive_strength'].iloc[:150], df['super_plasticisers'].iloc[:150],
                                    title, x_label, y_label)
        self.visualize.covariance_matrix(df, "Covariance Matrix -Super Plasticisers vs Compressive strength")

        # Cases involving Coarse and Fine Aggregate vs Compressive strength (combined)
        text = "Displaying columns after modification:"
        self.debug_text(text, self.df_concrete.columns)

        df = self.df_concrete[['coarse_aggregate', 'compressive_strength']].sort_values(by='compressive_strength',
                                                                                        ascending=False)
        self.visualize.covariance_matrix(df, "Covariance Matrix: Coarse Aggregate vs Compressive Strength")

        df = self.df_concrete[['fine_aggregate', 'compressive_strength']].sort_values(by='compressive_strength',
                                                                                      ascending=False)
        self.visualize.covariance_matrix(df, "Covariance Matrix: Fine Aggregate vs Compressive Strength")

        df = self.df_concrete[['combined_aggregate', 'compressive_strength']].sort_values(by='compressive_strength',
                                                                                          ascending=False)
        self.visualize.covariance_matrix(df, "Covariance Matrix: Combined Aggregate vs Compressive Strength")

        # Whether age affected the strength in the historical data
        df = self.df_concrete[['age', 'compressive_strength']].sort_values(by='compressive_strength',
                                                                           ascending=False)

        # Scatter Plot and Covariance Matrix
        title = "Age vs High Compressive Strength"
        x_label = "Compressive Strength (MPa)"
        y_label = "Age (Years)"
        self.display_dataframe('df', df, 25)
        self.visualize.plot_bar(df['compressive_strength'].iloc[:150], df['age'].iloc[:150],
                                title, x_label, y_label)
        self.visualize.covariance_matrix(df, "Covariance Matrix: Age vs Compressive strength")

        # Pair Plot
        title = "Pair Plot: Concrete Dataset"
        self.visualize.plot_pair_plot(self.df_concrete, title)

        # Pearson Correlation
        title = "Pearson Correlation Heatmap"
        self.visualize.pearson_correlation(self.df_concrete, title)


if __name__ == "__main__":
    main = FeatureAnalysis()
    main.feature_analysis()
