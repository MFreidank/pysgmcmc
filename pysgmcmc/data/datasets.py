import pandas
from os.path import join as path_join, abspath


def split_dataframe(data_frame, label_column, test_split=0.2, seed=113):
    test_data = data_frame.sample(frac=test_split, random_state=seed)
    train_data = data_frame[~data_frame.index.isin(test_data.index)]

    feature_columns = tuple(
        column_name for column_name in data_frame.columns
        if column_name != label_column
    )

    x_train = train_data.as_matrix(feature_columns)
    y_train = train_data[label_column].as_matrix()

    x_test = test_data.as_matrix(feature_columns)
    y_test = test_data[label_column].as_matrix()

    return (x_train, y_train), (x_test, y_test)


class BostonHousing(object):
    @classmethod
    def load_data(cls, path=abspath(path_join(__file__, "../../../datasets/uci/boston_housing.csv")),
                  test_split=0.2, seed=113):
        data_frame = pandas.read_csv(path, delimiter=",")

        return split_dataframe(
            data_frame, label_column="MEDV", test_split=test_split, seed=seed
        )


class WineQualityRed(object):
    @classmethod
    def load_data(cls, path=abspath(path_join(__file__, "../../../datasets/uci/winequality-red.csv")),
                  test_split=0.2, seed=113):
        data_frame = pandas.read_csv(path, delimiter=";")

        return split_dataframe(
            data_frame, label_column="quality", test_split=test_split, seed=seed
        )


class YachtHydrodynamics(object):
    @classmethod
    def load_data(cls, path=abspath(path_join(__file__, "../../../datasets/uci/yacht-hydrodynamics.csv")),
                  test_split=0.2, seed=113):
        data_frame = pandas.read_csv(path, delimiter=";")

        return split_dataframe(
            data_frame, label_column="resistance", test_split=test_split, seed=seed
        )


class Concrete(object):
    @classmethod
    def load_data(cls, path=abspath(path_join(__file__, "../../../datasets/uci/concrete.xls")), test_split=0.2, seed=113):
        data_frame = pandas.read_excel(path)
        return split_dataframe(
            data_frame, label_column="Concrete compressive strength(MPa, megapascals) ",
            test_split=test_split, seed=seed
        )
