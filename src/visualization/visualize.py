import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import math
from matplotlib.font_manager import FontProperties
import random as rnd


class Visualize:
    def __init__(self):
        pass

    @staticmethod
    def graph_settings():
        # Customizable Set-ups
        plt.figure(figsize=(13, 15))
        font = FontProperties()
        font.set_family('serif bold')
        font.set_style('oblique')
        font.set_weight('bold')
        ax = plt.axes()
        ax.set_facecolor("#e6eef1")

    def plot_dist(self, df, bins, title, x_label, y_label):
        self.graph_settings()

        sb.histplot(data=df, kde=True, bins=bins)
        plt.title(title, fontsize=17, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.show()

    def plot_histogram(self, data_values, bins, title, x_label, y_label):
        self.graph_settings()
        color = ['maroon', '#01575c', '#012f5c', '#73021b', '#5c270a']
        index = rnd.randint(0, 4)

        plt.hist(data_values, bins=bins, color=color[index], edgecolor='#0d0103', linewidth=1.2)
        plt.title(title, fontsize=17, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.show()

    def plot_multi_histogram(self, d1, d2, d3, d4, d5, d6, d7, bins):
        self.graph_settings()

        fig, axes = plt.subplots(2, 4)
        plt.rcParams["figure.figsize"] = [14, 16]
        plt.rcParams["figure.autolayout"] = True

        d1.hist(bins=bins, color='#754106', ax=axes[0, 0], edgecolor='#0d0103', linewidth=1.2)
        d2.hist(bins=bins, color='#754e06', ax=axes[0, 1], edgecolor='#0d0103', linewidth=1.2)
        d3.hist(bins=bins, color='#064575', ax=axes[0, 2], edgecolor='#0d0103', linewidth=1.2)
        d4.hist(bins=bins, color='#066175', ax=axes[0, 3], edgecolor='#0d0103', linewidth=1.2)

        d5.hist(bins=bins, color='#067543', ax=axes[1, 0], edgecolor='#0d0103', linewidth=1.2)
        d6.hist(bins=bins, color='#067519', ax=axes[1, 1], edgecolor='#0d0103', linewidth=1.2)
        d7.hist(bins=bins, color='maroon', ax=axes[1, 2], edgecolor='#0d0103', linewidth=1.2)
        fig.delaxes(axes[1, 3])

        axes[0, 0].set_title('Cement', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Blast Furnace Slag', fontsize=12, fontweight='bold')
        axes[0, 2].set_title('Water', fontsize=12, fontweight='bold')
        axes[0, 3].set_title('Super Plasticisers', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Coarse Aggregate', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Fine Aggregate', fontsize=12, fontweight='bold')
        axes[1, 2].set_title('Age', fontsize=12, fontweight='bold')
        plt.show()

    def plot_pair_plot(self, df, title):
        # Construct Pair-plot for the dataset
        self.graph_settings()
        print(title)
        sb.set(style="ticks", color_codes=True)
        sb.pairplot(df)

        plt.show()

    def plot_pie(self, data_values, explode, labels, title):
        self.graph_settings()

        plt.pie(data_values, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
        plt.title(title, fontsize=17, fontweight='bold')
        plt.show()

    def plot_bar(self, x, y, title, x_label, y_label):
        # Vertical Bar charts using matplotlib
        self.graph_settings()

        color = ['maroon', '#01575c', '#012f5c', '#73021b', '#5c270a']
        index = rnd.randint(0, 4)
        plt.bar(x, y, color=color[index], edgecolor='#0d0103', linewidth=1.2)

        plt.title(title, fontsize=17, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.xticks()
        plt.show()

    def plot_scatter(self, x, y, title, x_label, y_label):
        # Vertical Bar charts using matplotlib
        self.graph_settings()

        color = ['maroon', '#01575c', '#012f5c', '#73021b', '#5c270a']
        index = rnd.randint(0, 4)
        plt.scatter(x, y, color=color[index], edgecolor='#0d0103', linewidth=1.2)

        plt.title(title, fontsize=17, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.xticks()
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.show()

    def plot_residual(self, x, y, title, x_label, y_label):
        self.graph_settings()
        sb.residplot(x=x, y=y, lowess=True)

        plt.title(title, fontsize=17, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.xticks()
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.show()

    def plot_loss_curve(self, model, title, x_label, y_label):
        self.graph_settings()

        plt.plot(np.array(model.loss_curve_[15:]))
        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.show()

    @staticmethod
    def plot_multi_kde(df, columns, super_title):
        i = 0
        j = 0
        row = 3
        column = 0
        items = len(columns)

        if items % 3 == 0:
            column = int(items / 3)

        else:
            column = int(math.floor(items / 3))

        fig, ax = plt.subplots(row, column, figsize=(19, 23))
        for feature in columns:
            x_label = str(feature).replace('_', ' ').title()

            if i <= row-1 and j <= column-1:
                sb.kdeplot(df[feature], ax=ax[i, j], color='r')
                ax[i, j].set_xlabel(x_label, fontsize=10, fontweight='bold')
                ax[i, j].set_ylabel('Density', fontsize=10, fontweight='bold')

            j += 1
            if j > column-1:
                i += 1
                j = 0

        fig.suptitle(super_title, fontsize=18, fontweight='bold')
        plt.show()

    @staticmethod
    def plot_compare_kde(df1, df2, df3, columns, super_title):
        i = 0
        j = 0
        row = 3
        column = 0
        items = len(columns)

        if items % 3 == 0:
            column = int(items / 3)

        else:
            column = int(math.floor(items / 3))

        fig, ax = plt.subplots(row, column, figsize=(19, 23))
        for feature in columns:
            x_label = str(feature).replace('_', ' ').title()

            if i <= row - 1 and j <= column - 1:
                sb.kdeplot(df1[feature], ax=ax[i, j], color='r')
                sb.kdeplot(df2[feature], ax=ax[i, j], color='g')
                sb.kdeplot(df3[feature], ax=ax[i, j], color='b')
                ax[i, j].set_xlabel(x_label, fontsize=10, fontweight='bold')
                ax[i, j].set_ylabel('Density', fontsize=10, fontweight='bold')
                ax[i, j].legend(["Robust", "Standard", "MinMax"], fontsize="10", loc="upper right")

            j += 1
            if j > column - 1:
                i += 1
                j = 0

        fig.suptitle(super_title, fontsize=18, fontweight='bold')
        plt.show()

    @staticmethod
    def pearson_correlation(df, title):
        print("• Pearson Correlation Matrix:\n", df.corr())
        data_plot = sb.heatmap(df.corr(), cmap="YlGnBu", annot=True)
        plt.title(title, fontsize=17, fontweight='bold')
        plt.yticks(rotation='horizontal')
        plt.show()

    @staticmethod
    def covariance_matrix(df, title):
        print("• Covariance Matrix:\n", df.corr())
        index = rnd.randint(0, 6)
        color = ['Blues', 'Reds', 'PuBuGn', 'Purples', 'RdGy', 'YlGn', 'PuRd']
        corr = df.select_dtypes('number').corr()
        sb.heatmap(corr, cmap=color[index], annot=True)
        plt.title(title, fontsize=17, fontweight='bold')
        plt.show()


if __name__ == "__main__":
    main = Visualize()