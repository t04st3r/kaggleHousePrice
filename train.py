import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler


def debug(df):
    print(df.info())
    print(df.describe())
    print(df.head())
    print('......................')
    print(df.tail())


columns_to_drop = ['Street', 'Alley', 'Utilities', 'Condition2', 'RoofMatl', 'MoSold', 'MiscFeature', 'Id']

nominal_columns = ['MSSubClass', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType',
                   'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                   'CentralAir', 'GarageType', 'SaleType', 'SaleCondition']

ordinal_columns = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                   'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical',
                   'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                   'PavedDrive',
                   'PoolQC', 'Fence']

skewed_attributes = ['MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch', 'LandSlope']

all_columns = ['LotFrontage', 'LotArea', 'LotShape', 'LandSlope', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscVal', 'YrSold', 'MSSubClass_60', 'MSSubClass_20', 'MSSubClass_70', 'MSSubClass_50', 'MSSubClass_190', 'MSSubClass_45', 'MSSubClass_90', 'MSSubClass_120', 'MSSubClass_30', 'MSSubClass_85', 'MSSubClass_80', 'MSSubClass_160', 'MSSubClass_75', 'MSSubClass_180', 'MSSubClass_40', 'MSSubClass_150', 'MSZoning_RL', 'MSZoning_RM', 'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'LandContour_Lvl', 'LandContour_Bnk', 'LandContour_Low', 'LandContour_HLS', 'LotConfig_Inside', 'LotConfig_FR2', 'LotConfig_Corner', 'LotConfig_CulDSac', 'LotConfig_FR3', 'Neighborhood_CollgCr', 'Neighborhood_Veenker', 'Neighborhood_Crawfor', 'Neighborhood_NoRidge', 'Neighborhood_Mitchel', 'Neighborhood_Somerst', 'Neighborhood_NWAmes', 'Neighborhood_OldTown', 'Neighborhood_BrkSide', 'Neighborhood_Sawyer', 'Neighborhood_NridgHt', 'Neighborhood_NAmes', 'Neighborhood_SawyerW', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Timber', 'Neighborhood_Gilbert', 'Neighborhood_StoneBr', 'Neighborhood_ClearCr', 'Neighborhood_Edwards', 'Neighborhood_NPkVill', 'Neighborhood_Blmngtn', 'Neighborhood_BrDale', 'Neighborhood_SWISU', 'Neighborhood_Blueste', 'Condition1_Norm', 'Condition1_Feedr', 'Condition1_PosN', 'Condition1_Artery', 'Condition1_RRAe', 'Condition1_RRNn', 'Condition1_RRAn', 'Condition1_PosA', 'Condition1_RRNe', 'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_TwnhsE', 'BldgType_Twnhs', 'HouseStyle_2Story', 'HouseStyle_1Story', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'HouseStyle_2.5Unf', 'HouseStyle_2.5Fin', 'RoofStyle_Gable', 'RoofStyle_Hip', 'RoofStyle_Gambrel', 'RoofStyle_Mansard', 'RoofStyle_Flat', 'RoofStyle_Shed', 'Exterior1st_VinylSd', 'Exterior1st_MetalSd', 'Exterior1st_Wd Sdng', 'Exterior1st_HdBoard', 'Exterior1st_BrkFace', 'Exterior1st_WdShing', 'Exterior1st_CemntBd', 'Exterior1st_Plywood', 'Exterior1st_Stucco', 'Exterior1st_AsbShng', 'Exterior1st_BrkComm', 'Exterior1st_Stone', 'Exterior1st_ImStucc', 'Exterior1st_CBlock', 'Exterior2nd_VinylSd', 'Exterior2nd_MetalSd', 'Exterior2nd_Wd Shng', 'Exterior2nd_HdBoard', 'Exterior2nd_Plywood', 'Exterior2nd_Wd Sdng', 'Exterior2nd_CmentBd', 'Exterior2nd_BrkFace', 'Exterior2nd_Stucco', 'Exterior2nd_AsbShng', 'Exterior2nd_Brk Cmn', 'Exterior2nd_ImStucc', 'Exterior2nd_AsphShn', 'Exterior2nd_Other', 'Exterior2nd_Stone', 'Exterior2nd_CBlock', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_BrkCmn', 'Foundation_PConc', 'Foundation_CBlock', 'Foundation_BrkTil', 'Foundation_Wood', 'Foundation_Slab', 'Foundation_Stone', 'Heating_GasA', 'Heating_GasW', 'Heating_Wall', 'Heating_Grav', 'Heating_OthW', 'Heating_Floor', 'CentralAir_Y', 'CentralAir_N', 'GarageType_Attchd', 'GarageType_Detchd', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Basment', 'GarageType_2Types', 'SaleType_WD', 'SaleType_New', 'SaleType_COD', 'SaleType_ConLI', 'SaleType_CWD', 'SaleType_ConLw', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_Oth', 'SaleCondition_Normal', 'SaleCondition_Abnorml', 'SaleCondition_Partial', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_AdjLand', 'FullBathSum', 'HalfBathSum']


column_to_add = [['BsmtFullBath', 'FullBath', 'FullBathSum'], ['BsmtHalfBath', 'HalfBath', 'HalfBathSum']]

ordinal_maps = [{'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}, {'Gtl': 2, 'Mod': 1, 'Sev': 0},
                {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}, {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
                {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
                {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1}, {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1},
                {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1},
                {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
                {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1},
                {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
                {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0},
                {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, {'Fin': 3, 'RFn': 2, 'Unf': 1},
                {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}, {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
                {'Y': 2, 'P': 1, 'N': 0}, {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1},
                {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1}]


class FeaturesDropper(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_to_drop):
        self.attributes_to_drop = attributes_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.drop(self.attributes_to_drop, axis=1, inplace=True)
        return X


class HandleOrdinalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_attributes, ordinal_map):
        self.ordinal_attributes = ordinal_attributes
        self.ordinal_map = ordinal_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column, mapping in zip(self.ordinal_attributes, self.ordinal_map):
            X.loc[:, column] = X[column].map(mapping)
            if X[column].isna().any():
                X[column].fillna(0, inplace=True)
        return X


class HandleNominalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, nominal_attributes, whole_dataset):
        self.nominal_attributes = nominal_attributes
        self.whole_dataset = whole_dataset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.nominal_attributes:
            X[column].fillna(X[column].mode(), inplace=True)
            X.loc[:, column] = X[column].astype(pd.api.types.CategoricalDtype(categories=self.whole_dataset[column].unique()))
            X = pd.concat([X, pd.get_dummies(X[column], prefix=column)], axis=1)
            X.drop([column], axis=1, inplace=True)
        return X


class HandleSkewness(BaseEstimator, TransformerMixin):
    def __init__(self, skewed_attributes):
        self.skewed_attributes = skewed_attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.skewed_attributes:
            X.loc[:, column] = np.log1p(X[column])
        return X


class ImputeMissingValues(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fill_nan = SimpleImputer(strategy="median")
        imputed_X = pd.DataFrame(fill_nan.fit_transform(X))
        imputed_X.columns = X.columns
        imputed_X.index = X.index
        return imputed_X


class SelectFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.features]


# create train ant test pandas DataFrame from csv
iowa_file_path = 'train.csv'
train_data = pd.read_csv(iowa_file_path)
y = np.log(train_data.SalePrice)
train_data.drop('SalePrice', axis=1, inplace=True)

test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)


# Create target object and call it y

nominal_merged = pd.concat([train_data[nominal_columns], test_data[nominal_columns]])
nominal_merged.dropna(inplace=True)


pipeline_gbr = Pipeline([
    ('dropper', FeaturesDropper(columns_to_drop)),
    ('ordinal', HandleOrdinalFeatures(ordinal_columns, ordinal_maps)),
    ('nominal', HandleNominalFeatures(nominal_columns, nominal_merged)),
    ('imputer', ImputeMissingValues()),
    ('skewness', HandleSkewness(skewed_attributes)),
    ('scaler', RobustScaler()),
    ('gbr', GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                 learning_rate=0.02, loss='ls', max_depth=4, max_features=0.1,
                                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, min_samples_leaf=3,
                                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                                 n_iter_no_change=None, presort='auto', n_estimators=3000,
                                 random_state=None, subsample=1.0, tol=0.0001,
                                 validation_fraction=0.1, verbose=0, warm_start=False))
])

mae = []

for i in range(0, 3):
    scores = cross_val_score(pipeline_gbr, train_data, y, scoring='neg_mean_absolute_error', n_jobs=-1, cv=8)
    mae.append((-1 * scores.mean()))


print('MAE %2f' % (sum(mae)/len(mae)))

