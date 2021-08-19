# deconvolution model for blood PBMC samples
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict
from core.cell_types import CellTypes
from core.mixer import Mixer
import timeit


# boosting parameters typing
boosting_parameters_dtypes = {
        'learning_rate': float,
        'max_depth': int,
        'min_data_in_leaf': int,
        'num_iterations': int,
        'n_estimators': int,
        'subsample': float,
        'bagging_fraction': float,
        'bagging_freq': int,
        'lambda_l1': float,
        'lambda_l2': float,
        'feature_fraction': float,
        'gamma': float,
        'reg_alpha': float,
        'reg_lambda': float,
        'colsample_bytree': float,
        'colsample_bylevel': float,
        'min_child_weight': int,
        'random_state': int,
        'n_jobs': int}

class DeconvolutionModel:
    """
    Base class for model training and prediction.
    """
    def __init__(self, cell_types: CellTypes,
                 boosting_params_first_step = 'configs/boosting_params/lgb_parameters_first_step.tsv',
                 boosting_params_second_step = 'configs/boosting_params/lgb_parameters_second_step.tsv',
                 genes_in_expression_path='configs/genes_in_expression.txt',
                 l1_models: Dict = None,    
                 l2_models: Dict = None,
                 random_seed=0):
        """
        :param cell_types: Object of class CellTypes
        :param boosting_params_first_step: path to boosting parameters for the first step
        :param boosting_params_second_step: path to boosting parameters for the second step
        :param random_seed: random seed
        """
        self.cell_types = cell_types
        self.random_seed = random_seed
        self.boosting_params_first_step = pd.read_csv(boosting_params_first_step, sep='\t',
                                                      index_col=0, dtype=boosting_parameters_dtypes)
        self.boosting_params_second_step = pd.read_csv(boosting_params_second_step, sep='\t',
                                                       index_col=0, dtype=boosting_parameters_dtypes)
        # assigned in fit or directly in __init__
        self.l1_models = l1_models or {}
        self.l2_models = l2_models or {}
        # assigned in fit from mixer.genes_in_expression
        self.genes_in_expression = []
        with open(genes_in_expression_path, "r") as f:
            for line in f:
                self.genes_in_expression.append(line.strip())
            
    def fit(self, mixer: Mixer):
        """
        Training pipeline for this model.
        :param mixer: object of Mixer/TumorMixer/... class
        """
        np.random.seed(self.random_seed)
        start = timeit.default_timer()
        print('============== L1 models ==============')
        for i, cell in enumerate(self.cell_types.models):
            print(f'Generating mixes for {cell} model')
            start1 = timeit.default_timer()
            expr, values = mixer.generate(cell, genes=self.cell_types[cell].genes, random_seed=i+1)
            print(f'Fitting {cell} model')
            self.l1_models[cell] = self.train_l1_model(expr, values, cell)
            end1 = timeit.default_timer()
            print(f'Trained in:  {round(end1-start1, 1)} sec.')
            print('\n')

        print('============== L2 models ==============')
        for i, cell in enumerate(self.cell_types.models):
            print(f'Generating mixes for {cell} model')
            start1 = timeit.default_timer()
            expr, values = mixer.generate(cell, genes=self.cell_types.genes, random_seed=i+1007)
            print(f'Fitting {cell} model')
            self.l2_models[cell] = self.train_l2_model(expr, values, cell)
            end1 = timeit.default_timer()
            print(f'Trained in:  {round(end1-start1, 1)} sec.')
            print('\n')
        
        end = timeit.default_timer()
        print(f'Deconv model fitting done in: {round(end-start, 1)} sec.')

    def train_l1_model(self, expr, values, cell):
        """
        Trains L1 model for one cell type.
        :param expr: pd df with samples in columns and genes in rows
        :param values: pd df with true RNA fractions
        :param cell: cell type for which model is trained
        :return: trained model for cell type
        """
        features = sorted(list(set(self.cell_types[cell].genes)))
        x = expr.T[features]
        x = x.sample(frac=1)
        y = values.loc[cell].loc[x.index]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        boosting_params = self.boosting_params_first_step.to_dict(orient='index')[cell]
        model = lgb.LGBMRegressor(**boosting_params,
                                random_state=0)
        model.fit(x, y)

        return model

    def train_l2_model(self, expr, values, cell):
        """
        Trains L2 model for one cell type. Uses L1 models as an input features.
        :param expr: pd df with samples in columns and genes in rows
        :param values: pd df with true RNA fractions
        :param cell: cell type for which model is trained
        :return: trained model for cell type
        """
        features = sorted(list(set(self.cell_types.genes)))
        x = expr.T[features]
        x = x.sample(frac=1)
        l1_preds = self.predict_l1(x.T)
        features = sorted(list(set(self.cell_types[cell].genes)))
        x = x[features]
        x = pd.concat([x, l1_preds], axis=1)
        y = values.loc[cell].loc[x.index]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        boosting_params = self.boosting_params_second_step.to_dict(orient='index')[cell]
        model = lgb.LGBMRegressor(**boosting_params,
                                random_state=0)

        model.fit(x, y)
        return model
    
    def predict(self, expr, use_l2=False, add_other=True, other_coeff=0.073468):
        """
        Prediction pipeline for the model.
        :param expr: pd df with samples in columns and genes in rows
        :param predict_cells: If RNA fractions to be recalculated to cells fractions.
        :return: pd df with predictions for cell types in rows and samples in columns.
        """
        self.check_expressions(expr)
        expr = self.renormalize_expr(expr)
        preds = self.predict_l2(expr)

        preds = self.adjust_rna_fractions(preds, other_coeff)
        preds = self.convert_rna_to_cells_fractions(preds, other_coeff)
        preds = preds.T
        return preds
    
    def check_expressions(self, expr):
        '''
        Checks if expressions have the right format.
        '''
        if not any(expr.max(axis=1) > np.log2(10**6)):
            raise ValueError("MODEL DOES NOT WORK WITH LOG NORMALIZED DATA. LINEARIZE YOUR EXPRESSION MATRXI.")
        diff = set(self.cell_types.genes).difference(set(expr.index))
        if diff:
            raise ValueError("EXPRESSION MATRIX HAS TO CONTAIN AT LEAST ALL THE GENES THAT ARE USED AS A FEATURES")
        diff = set(self.cell_types.genes).symmetric_difference(set(expr.index))
        if not diff:
            print(f'WARNING: YOU USING ONLY FEATURE GENES. MAKE SURE THAT NORMALIZATION IS CORRECT')
        else:
            print("Expressions OK")

    def renormalize_expr(self, expr):
        sym_diff = set(self.genes_in_expression).symmetric_difference(set(expr.index))
        if len(sym_diff) > 0:
            expr = expr.loc[self.genes_in_expression]
            expr = (expr / expr.sum()) * 10**6

        return expr

    def adjust_rna_fractions(self, preds, add_other):
        """
        Adjusts predicted fractions based on cell types tree structure. Lower subtypes recalculated to sum up to
        value of its parent type.
        :param preds: pd df with predictions for cell types in columns and samples in rows.
        :add_other: if not None adds Other fraction in case if sum of all general cell types predictors yeilds < 1
        :returns: adjusted preds
        """
        preds[preds < 0] = 0
        cell = self.cell_types.root
        general_types = [ct for ct in self.cell_types.get_direct_subtypes(cell) if ct in self.cell_types.models]
        # adding other 
        for sample in preds.index:
            s = preds.loc[sample, general_types].sum()
            if s < 1 and add_other:
                preds.loc[sample, 'Other'] = 1 - s
            else:
                preds.loc[sample, general_types] = preds.loc[sample, general_types] / s
                preds.loc[sample, 'Other'] = 0

            
        cells_with_unadjusted_subtypes = general_types

        while cells_with_unadjusted_subtypes:
            cell = cells_with_unadjusted_subtypes.pop()
            subtypes = [ct for ct in self.cell_types.get_direct_subtypes(cell) if ct in self.cell_types.models]
            preds[subtypes] = preds[subtypes].divide(preds[subtypes].sum(axis=1), axis=0)
            preds[subtypes] = preds[subtypes].multiply(preds[cell], axis=0)
            cells_with_unadjusted_subtypes = subtypes + cells_with_unadjusted_subtypes

        return preds

    def convert_rna_to_cells_fractions(self, rna_fractions, other_coeff):
        """
        Multiplies RNA fractions predictions for each cell on corresponded rna_per_cell coefficient from cell_config.yaml
        :param preds: pd df with RNA fractions predictions
        :return: pd df with adjusted predictions
        """
        rna_fractions = rna_fractions.T
        terminal_models = []
        for cell in self.cell_types.models:
            subtypes = self.cell_types.get_all_subtypes(cell)
            submodels = [c for c in subtypes if self.cell_types[c].model]
            if not submodels:
                terminal_models.append(cell)
        
        non_terminal_models = [cell for cell in self.cell_types.models if cell not in terminal_models]

        cells_fractions = rna_fractions.loc[['Other'] + terminal_models]
        coefs = pd.Series([other_coeff] + [self.cell_types[cell].rna_per_cell for cell in terminal_models])
        terminal_models = ['Other'] + terminal_models
        coefs.index = terminal_models
        cells_fractions = cells_fractions.mul(coefs, axis='rows')
        cells_fractions = cells_fractions / cells_fractions.sum()
        while non_terminal_models:
            m = non_terminal_models.pop()
            submodels = self.cell_types.get_direct_subtypes(m) # get all subtypes maybe??? 
            submodels = [cell for cell in submodels if cell in self.cell_types.models]
            # if its subtypes still unadjusted move it to the end of the queue
            skip = [cell for cell in submodels if cell in non_terminal_models]
            if skip:
                non_terminal_models = [m] + non_terminal_models
            else:
                cells_fractions.loc[m] = cells_fractions.loc[submodels].sum(axis=0)

        return cells_fractions.T

    def predict_l1(self, expr):
        """
        Predicts rna fractions by L1 models.
        :param expr: pd df with samples in columns and genes in rows.
        :return: L1 models predictions.
        """
        preds = {}
        for cell in sorted(self.l1_models.keys()):
            features = sorted(list(set(self.cell_types[cell].genes)))
            x = expr.T[features]
            preds[cell] = self.l1_models[cell].predict(x)
        preds =  pd.DataFrame(preds)
        preds.index = x.index
        return preds

    def predict_l2(self, expr):
        """
        Predicts rna fractions by L2 models using L1 models predictions as an input features.
        :param expr: pd df with samples in columns and genes in rows.
        :return: L2 models predictions.
        """
        preds = {}
        l1_preds = self.predict_l1(expr)
        for cell in sorted(self.l2_models.keys()):
            features = sorted(list(set(self.cell_types[cell].genes)))
            x = expr.T[features]
            x = pd.concat([x, l1_preds], axis=1)
            preds[cell] = self.l2_models[cell].predict(x)
        preds =  pd.DataFrame(preds)
        preds.index = x.index
        return preds
