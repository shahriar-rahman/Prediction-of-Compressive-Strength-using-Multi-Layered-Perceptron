import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
sys.path.append(os.path.abspath('../features'))
sys.path.append(os.path.abspath('../visualization'))
import visualize
path = ['../../data/processed/compressive_strength_test_x.csv', '../../data/processed/compressive_strength_test_y.csv',
        '../../models/mlp.pkl']


class TestModel:
    def __init__(self):
        # Object Instantiation
        self.visualize = visualize.Visualize()
        self.test_x = pd.read_csv(path[0], index_col=False)
        self.test_y = pd.read_csv(path[1], index_col=False)
        self.mlp = pickle.load(open(path[2], 'rb'))

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

    def test_model(self):
        pred_y = self.mlp.predict(self.test_x)

        # Loss function
        mse = MSE(self.test_y, pred_y, squared=False)
        r2 = r2_score(self.test_y, pred_y)

        self.debug_text("MSE for predicting the Test set:", mse)
        self.debug_text("R-Squared Error for predicting the Test set:", r2)

        # Model diagnostics
        title = "Loss curve after first 15 epochs"
        x_label = "Epochs"
        y_label = "Mean Squared Error"
        self.visualize.plot_loss_curve(self.mlp, title, x_label, y_label)

        title = ["Scatter plot for Test samples", "Residual plot for Test samples"]
        x_label = ["Actual Values", "Fitted Values"]
        y_label = ["Predicted Values", "Residuals"]

        self.visualize.plot_scatter(self.test_y, pred_y, title[0], x_label[0], y_label[0])
        self.visualize.plot_residual(self.test_y, pred_y, title[1], x_label[1], y_label[1])

        error_list = pred_y - self.test_y.to_numpy().ravel()
        bins = 10
        x_label = 'Errors'
        y_label = 'Frequency'
        title = "Raw Prediction errors for the Test set"
        self.visualize.plot_dist(error_list, bins, title, x_label, y_label)


if __name__ == "__main__":
    main = TestModel()
    main.test_model()
