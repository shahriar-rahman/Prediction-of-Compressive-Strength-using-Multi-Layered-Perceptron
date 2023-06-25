import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
sys.path.append(os.path.abspath('../features'))
sys.path.append(os.path.abspath('../visualization'))
import visualize
import construct_features
import feature_analysis
import feature_transformation
path = '../../data/interim/compressive_strength.csv'


class TrainModel:
    def __init__(self):
        # Object Instantiation
        self.df_concrete = pd.read_csv(path, index_col=False)
        self.construct = construct_features.ConstructFeatures()
        self.analysis = feature_analysis.FeatureAnalysis()
        self.transformation = feature_transformation.FeatureTransformation()
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

    def feature_exploration(self):
        self.construct.construct_features()
        self.analysis.feature_analysis()
        self.transformation.feature_transformation()

    def model_train(self):
        # Train-Test Split
        df_x = self.df_concrete[['cement', 'blast_furnace_slag', 'water', 'super_plasticisers',
                                'coarse_aggregate', 'fine_aggregate', 'age', 'combined_aggregate']].copy()
        df_y = self.df_concrete['compressive_strength']

        self.display_dataframe("Features", df_x, 10)
        self.display_dataframe("Label", df_y, 10)

        train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.20, random_state=48)
        self.debug_text("Shape of Training set:", train_x.shape)
        self.debug_text("Shape of Testing set:", test_x.shape)

        # Grid CV Search
        activation_functions = ["relu", "tanh", "logistic"]
        lr_starts = [0.15, 0.02, 0.05]
        hidden_layers = [(40, 30, 20), (30, 15, 5), (40, 20, 10)]

        mlp_params = {"hidden_layer_sizes": hidden_layers, "activation": activation_functions,
                      "learning_rate_init": lr_starts}

        mlp = MLPRegressor(random_state=48, solver='adam', early_stopping=True, max_iter=1000)

        mlp_grid = GridSearchCV(mlp, mlp_params, scoring='neg_mean_squared_error', cv=5)
        grid_search = mlp_grid
        grid_search.fit(train_x, train_y)

        self.debug_text("Ideal parameters: ", grid_search.best_params_)
        self.debug_text("Ideal Score: ", grid_search.best_score_)

        # MLP Regression
        mlp = MLPRegressor(random_state=48, solver='adam', early_stopping=True, max_iter=2000,
                           activation='relu', hidden_layer_sizes=(30, 15, 5), learning_rate_init=0.02)
        mlp.fit(train_x, train_y)
        pred_y = mlp.predict(train_x)

        # Loss function
        mse = MSE(train_y, pred_y, squared=False)
        r2 = r2_score(train_y, pred_y)

        self.debug_text("MSE for predicting the training set:", mse)
        self.debug_text("R-Squared Error for predicting the training set:", r2)

        # Model diagnostics
        title = "Loss curve after first 15 epochs"
        x_label = "Epochs"
        y_label = "Mean Squared Error"
        self.visualize.plot_loss_curve(mlp, title, x_label, y_label)

        title = ["Scatter plot for Training samples", "Residual plot for Training samples"]
        x_label = ["Actual Values", "Fitted Values"]
        y_label = ["Predicted Values", "Residuals"]

        self.visualize.plot_scatter(train_y, pred_y, title[0], x_label[0], y_label[0])
        self.visualize.plot_residual(train_y, pred_y, title[1], x_label[1], y_label[1])

        error_list = pred_y - train_y.to_numpy().ravel()
        bins = 10
        x_label = 'Errors'
        y_label = 'Frequency'
        title = "Raw Prediction errors for the Training set"
        self.visualize.plot_dist(error_list, bins, title, x_label, y_label)

        # Store Model
        try:
            pickle.dump(mlp, open('../../models/mlp.pkl', 'wb'))

        except Exception as exc:
            self.debug_text("! Exception encountered", exc)

        else:
            self.debug_text("Model saved successfully", '')

        # Store Test partitioned data
        try:
            self.data_storage(test_x, 'test_x')
            self.data_storage(test_y, 'test_y')

        except Exception as exc:
            self.debug_text("! Exceptions encountered:", exc)

        else:
            self.debug_text("Partitioned data saved successfully.", "")

    def data_storage(self, df, name):
        # Save partitioned data to storage
        try:
            df.to_csv(f'../../data/processed/compressive_strength_{str(name)}.csv', sep=',', index=False)

        except Exception as exc:
            self.debug_text("! Exception encountered", exc)

        else:
            text = "Dataframe successfully saved"
            self.debug_text(text, '...')


if __name__ == "__main__":
    main = TrainModel()
    main.feature_exploration()
    main.model_train()
