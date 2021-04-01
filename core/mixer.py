import gc
from typing import Dict, Set, Union, List, Tuple
import pandas as pd
import numpy as np
from numpy import random
from core.cell_types import CellTypes, get_proportions_series


class Mixer:
    """
    Class for mix generation. Handles cells expression mixing and noise adding.
    """
    def __init__(self,
                 cell_types: CellTypes,
                 cells_expr: pd.DataFrame,
                 cells_annot: pd.DataFrame,
                 tumor_expr: pd.DataFrame, 
                 tumor_annot: pd.DataFrame,
                 tumor_mean=0.5,
                 tumor_sd=0.5,
                 hyperexpression_fraction=0.01,
                 max_hyperexpr_level=1000,
                 num_points: int = 1000,
                 rebalance_param: float = 0.3,
                 gene_length: str = 'configs/gene_length_values.tsv',
                 genes_in_expression_path='configs/genes_in_expression.txt',
                 num_av: int = 5,
                 all_genes: bool = False):
        """
        :param proportions: pandas Series with numbers for proportions for each type
        :param cell_types: Object of class CellTypes
        :param gene_length: path to table with gene lengths values
        :param rebalance_parameter: whether to reduce the weight of large datasets when forming random samples selection,
                                None or 0 < rebalance_parameter <= 1
                                rebalance_parameter == 1: equal number of samples from each dataset
        :param poisson_noise_level: coeff for Poisson noise level (larger - higher noise)
        :param uniform_noise_level: coeff for uniform noise level (larger - higher noise)
        :param dirichlet_samples_proportion: fraction of cell mixes that will be formed through the dirichlet distribution
                                            for method 'concat_ratios_with_dirichlet'
                                            Value must be in the range from 0 to 1.
        :param num_av: number of random samples of cell type that will be averaged to form the resulting sample
        :param num_points: number of resulting samples for each cell type
        :param genes: genes to consider in mixing. Uses all genes from cells_config if none provided.
        :param random_seed: fixed random state
        """
        self.num_points = num_points
        self.cell_types = cell_types
        self.rebalance_param = rebalance_param
        self.num_av = num_av
        self.proportions = get_proportions_series(cell_types)
        self.gene_length = pd.read_csv(gene_length, sep='\t', index_col=0)
        self.cells_annot = cells_annot
        
        self.genes_in_expression = []
        with open(genes_in_expression_path, "r") as f:
            for line in f:
                self.genes_in_expression.append(line.strip())
        
        print('Checking normal cells expressions...')
        self.check_expressions(cells_expr)
        print('Checking cancer cells expressions...')
        self.check_expressions(tumor_expr)

        # renormalizing expressions
        cells_expr = cells_expr.loc[self.genes_in_expression]
        cells_expr = (cells_expr / cells_expr.sum()) * 10**6
        tumor_expr = tumor_expr.loc[self.genes_in_expression]
        tumor_expr = (tumor_expr / tumor_expr.sum()) * 10**6

        self.tumor_mean = tumor_mean
        self.tumor_sd = tumor_sd
        self.hyperexpression_fraction = hyperexpression_fraction
        self.max_hyperexpr_level = max_hyperexpr_level
        
        if all_genes:
            genes = cells_expr.index
        else: 
            genes = cell_types.genes

        self.tumor_expr = tumor_expr.loc[genes]
        self.cells_expr = cells_expr.loc[genes]

    def check_expressions(self, expr):
        '''
        Checks if expressions have the right format.
        '''
        if not any(expr.max(axis=1) > np.log2(10**6)):
            raise ValueError("MODEL DOES NOT WORK WITH LOG NORMALIZED DATA. LINEARIZE YOUR EXPRESSION MATRIX.")
        diff = set(self.cell_types.genes).difference(set(expr.index))
        if diff:
            raise ValueError(f'MISSING GENES IN EXPRESSION: \n {diff}')
        diff = set(self.cell_types.genes).symmetric_difference(set(expr.index))
        if not diff:
            print(f'WARNING: YOU USING ONLY FEATURE GENES. MAKE SURE THAT EXPRESSIONS WERE NORMALIZED ON THIS SET OF GENES.')
        else:
            print("Expressions OK")

    def generate(self,
                 modeled_cell: str,
                 make_noise: bool = True,
                 random_seed: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generates mixes for cell model training.
        :modeled_cell: Cell type for which model training mixes is to be assambled
        :random_seed: random seed
        :returns: tuple with dataframes of mixed expressions and rna proportions
        """
        np.random.seed(random_seed)

        mixed_cells_expr = pd.DataFrame(np.zeros((len(self.cells_expr.index), self.num_points)),
                                        index=self.cells_expr.index,
                                        columns=range(self.num_points), dtype=float)

        cells_to_mix = self.get_cells_to_mix(modeled_cell)
        average_cells = self.generate_pure_cell_expressions(self.num_av, cells_to_mix + [modeled_cell])
        mixed_cells_values = self.concat_ratios_with_dirichlet(cells_to_mix)

        for cell in mixed_cells_values.index:
            mixed_cells_expr += mixed_cells_values.loc[cell] * average_cells[cell]

        modeled_cell_values = self.normal_cell_distribution(mean=self.cell_types[modeled_cell].cell_proportion)

        other_cells_values = (1 - modeled_cell_values)
        mixed_cells_values *= other_cells_values
        mixed_cells_expr *= other_cells_values
        mixed_cells_expr += modeled_cell_values * average_cells[modeled_cell]
        mixed_cells_values.loc[modeled_cell] = modeled_cell_values        

        tumor_values = self.normal_cell_distribution(mean=self.tumor_mean,
                                                     sd=self.tumor_sd)

        microenvironment_fractions = 1 - tumor_values
        tumor_expr_reshaped = self.tumor_expr.sample(self.num_points, replace=True,
                                                     axis=1, random_state=random_seed)
        tumor_expr_reshaped.columns = range(self.num_points)
        tumor_expr_reshaped = self.add_tumor_hyperexpression(tumor_expr_reshaped,
                                                             hyperexpression_fraction=self.hyperexpression_fraction,
                                                             max_hyperexpr_level=self.max_hyperexpr_level)
        tumor_with_cells_expr = tumor_expr_reshaped * tumor_values + mixed_cells_expr * microenvironment_fractions
        tumor_with_cells_values = mixed_cells_values * microenvironment_fractions
        tumor_with_cells_values.loc['Tumor'] = tumor_values
        tumor_with_cells_expr = self.make_noise(tumor_with_cells_expr)
        gc.collect()
        return tumor_with_cells_expr, tumor_with_cells_values


    @staticmethod
    def rebalance_samples_by_type(annot: pd.DataFrame, k: float) -> pd.Index:
        """
        Function rebalances the annotation dataset: rare types (type is based on column 'col')
        appears more often due to the multiplication of their samples in the dataset.
        All NaN samples will be deleted.

        k == 0: no rebalance
        k == 1: number of samples of each type in 'col' increases to maximum
        0 < k < 1: rebalance based on 'func'

        :param annot: pandas annotation dataframe (samples as indices)
        :param k: rebalance parameter 0 < k < 1
        :return: list of samples
        """
        type_counter = annot['Dataset'].value_counts()

        func = lambda x: x**(1 - k)

        max_counter = type_counter.max()
        type_counter = np.round(func(type_counter / max_counter) * max_counter).astype(int)

        samples = []
        for t, counter in type_counter.items():
            samples.extend(np.random.choice(annot.loc[annot['Dataset'] == t].index, counter))

        return pd.Index(samples)

    def generate_pure_cell_expressions(self, num_av: int, cells_to_mix: List[str]) -> Dict[str, float]:
        """
        Function makes averaged samples of random cellular samples, taking into account the nested structure
        of the subtypes and the desired proportions of the subtypes for cell type.
        :param cells_to_mix: list of cell types for which averaged samples from random selection will be formed
        :param num_av: number of random samples of cell type that will be averaged to form the resulting sample
        :returns: dict with matrix of average of random num_av samples for each cell type with replacement
        """
        average_cells = {}
        for cell in cells_to_mix:
            cells_selection = self.select_cells_with_subtypes(cell)
            expressions_matrix = pd.DataFrame(np.zeros((len(self.cells_expr.index), self.num_points)),
                                              index=self.cells_expr.index,
                                              columns=range(self.num_points), dtype=float)
            for i in range(num_av):
                if self.rebalance_param is not None:
                    cells_index = pd.Index(self.rebalance_samples_by_type(self.cells_annot.loc[cells_selection.index],
                                                                          k=self.rebalance_param))
                else:
                    cells_index = cells_selection.index
                if self.proportions is not None:
                    cell_subtypes = self.cell_types.get_all_subtypes(cell)
                    specified_subtypes = set(self.proportions.dropna().index).intersection(cell_subtypes)
                    if len(specified_subtypes) > 1:
                        cells_index = self.change_subtype_proportions(cell=cell,
                                                                      cells_index=cells_index)
                samples = random.choice(cells_index, self.num_points)
                expressions_matrix += self.cells_expr.loc[:, samples].values
            average_cells[cell] = expressions_matrix / float(num_av)
        return average_cells

    def select_cells_with_subtypes(self, cell: str) -> pd.DataFrame:
        """
        Method makes a selection of all cell type samples with all level nested subtypes.
        :param cell: cell type from names in 'Cell_type'
        :returns: pandas Series with samples indexes and cell names
        """
        selected_cells = [cell] + self.cell_types.get_all_subtypes(cell)
        return self.cells_annot[self.cells_annot['Cell_type'].isin(selected_cells)]

    def change_subtype_proportions(self, cell: str, cells_index: pd.Index) -> pd.Index:
        """
        Function changes the proportions of the cell subtypes when they are considered as types for random selection.
        The proportions of the subtypes will be changed including samples of deeper subtypes
        :param cell: string with the name of cell type for which the proportions of the subtypes will be changed
        :param cells_index: pandas index of samples for cell type
        :returns: array of sample indexes oversampled for needed proportions
        """
        cell_subtypes = self.cell_types.get_direct_subtypes(cell)
        specified_subtypes = set(self.proportions.dropna().index).intersection(cell_subtypes)

        # cell type samples and samples without specified subtype proportion
        unspecified_types = list(set(cell_subtypes).difference(specified_subtypes)) + [cell]
        unspecified_samples = cells_index[self.cells_annot.loc[cells_index, 'Cell_type'].isin(unspecified_types)]
        min_num = min(self.proportions.loc[specified_subtypes])

        subtype_proportions = {cell: dict(self.proportions.loc[specified_subtypes])}

        subtype_samples = {}
        subtype_size = {}
        oversampled_subtypes = {}
        for subtype in specified_subtypes:
            subtype_subtypes = self.cell_types.get_direct_subtypes(subtype)
            subtype_has_subtypes = len(set(self.proportions.dropna().index).intersection(subtype_subtypes)) > 1

            subtype_samples[subtype] = self.select_cells_with_subtypes(subtype).index

            if subtype_has_subtypes:
                subtype_samples[subtype] = self.change_subtype_proportions(cell=subtype,
                                                                           cells_index=subtype_samples[subtype])
            subtype_size[subtype] = len(subtype_samples[subtype])
        max_size = max(subtype_size.values())
        result_samples = unspecified_samples
        for subtype in specified_subtypes:
            oversampled_subtypes[subtype] = np.random.choice(subtype_samples[subtype],
                                                             int(subtype_proportions[cell][
                                                                 subtype] * max_size / min_num + 1))
            result_samples = np.concatenate((result_samples, oversampled_subtypes[subtype]))
        return result_samples

    def concat_ratios_with_dirichlet(self, cells_to_mix: List[str], dirichlet_samples_proportion=0.4):
        """
        Function generates the values of the proportion of mixed cells by combining simple_ratios and dirichlet methods.
        :param num_points: int number of how many mixes to create
        :param likely_proportions: None for uniform proportions or pandas Series with numbers for proportions for each type
        :param cells_to_mix: list of cell types to mix
        :param dirichlet_samples_proportion: fraction of cell mixes that will be formed through the dirichlet distribution
                                             for method 'concat_ratios_with_dirichlet'
                                             Value must be in the range from 0 to 1.
        :returns: pandas dataframe with generated cell type fractions
        """
        num_dirichlet_points = int(self.num_points * dirichlet_samples_proportion)
        mix_ratios_values = self.simple_ratios_mixing(self.num_points - num_dirichlet_points, cells_to_mix)
        mix_dirichlet_values = self.dirichlet_mixing(num_dirichlet_points, cells_to_mix)
        mix_cell_values = pd.concat([mix_ratios_values, mix_dirichlet_values], axis=1)
        mix_cell_values.columns = range(self.num_points)
        return mix_cell_values

    def simple_ratios_mixing(self, num_points: int, cells_to_mix: List[str]):
        """
        Method generates the values of the proportion of mixed cells by simple ratios method.
        In this method, the areas of values in the region of given proportions are well enriched,
        but very few values that far from these areas.
        :param num_points: int number of how many mixes to create
        :param cells_to_mix: list of cell types to mix
        :returns: pandas dataframe with generated cell type fractions
        """
        mixed_cell_values = pd.DataFrame(np.random.random(size=(len(cells_to_mix), num_points)),
                                         index=cells_to_mix, columns=range(num_points))
        mixed_cell_values = (mixed_cell_values.T * self.proportions.loc[cells_to_mix]).T
        return mixed_cell_values / mixed_cell_values.sum()

    def dirichlet_mixing(self, num_points: int, cells_to_mix: List[str]):
        """
        Method generates the values of the proportion of mixed cells by dirichlet method.
        The method guarantees a high probability of the the presence of each cell type from 0 to 100%
        at the expense of enrichment of fractions close to zero.
        :param num_points: int number of how many mixes to create
        :param cells_to_mix: list of cell types to mix
        :returns: pandas dataframe with generated cell type fractions
        """
        return pd.DataFrame(np.random.dirichlet([1.0 / len(cells_to_mix)]*len(cells_to_mix), size=num_points).T,
                            index=cells_to_mix, columns=range(num_points))

    def get_cells_to_mix(self, modeled_cell: str) -> List[str]:
        """
        Returns list of cells to mix for modeld cell type.
        """
        cells_to_remove = [modeled_cell]
        cells_to_remove += self.cell_types.get_all_parents(modeled_cell)
        cells_to_remove += self.cell_types.get_all_subtypes(modeled_cell)
        cells_to_mix = []
        for cell in cells_to_remove:
            cells_to_mix += self.cell_types.get_direct_subtypes(cell)

        cells_to_mix = [cell for cell in cells_to_mix if cell not in cells_to_remove]
        return  cells_to_mix

    def normal_cell_distribution(self, sd=0.5, mean=0.5) -> float:
        """
        Generates vector with normal distribution truncated on [0,1] for cell mixing.
        :param sd: Standard deviation
        :param mean: mean
        :returns: np.array with values
        """
        values = sd * np.random.randn(self.num_points) + mean
        values[values < 0] = np.random.uniform(size=len(values[values < 0]))
        values[values > 1] = np.random.uniform(size=len(values[values > 1]))
        return values

    def make_noise(self,
                   data: pd.DataFrame,
                   poisson_noise_level=0.5,
                   uniform_noise_level=0) -> pd.DataFrame:
        """
        Method adds Poisson noise (very close approximation) and uniform noise for expressions in TPM.
        Uniform noise - proportional to gene expressions noise from a normal distribution.
        :param data: pandas dataframe with expressions in TPM with genes as indexes
        :returns: dataframe data with added noise
        """
        length_normed_data = (data.T * 1000.0 / self.gene_length.loc[data.index, 'length']).T.astype(float)
        data = data + np.sqrt(length_normed_data * poisson_noise_level) * np.random.normal(size=data.shape) + \
            uniform_noise_level * data * np.random.normal(size=data.shape)
        return data.clip(lower=0)

    @staticmethod
    def add_tumor_hyperexpression(data, hyperexpression_fraction, max_hyperexpr_level):
        """
        :param data: pandas dataframe with expressions in TPM
        :param hyperexpression_fraction: probability for gene to be hyperexpressed
        :param max_hyperexpr_level: maximum level of tumor expression
        :return:
        """
        tumor_noise = np.random.random(size=data.shape)
        tumor_noise = np.where(tumor_noise < hyperexpression_fraction, max_hyperexpr_level, 0)
        tumor_noise = tumor_noise * np.random.random(size=data.shape)
        data = data + tumor_noise
        return data

    def select_datasets_fraction(self, annotation, selector_col, bootstrap_fraction):
        """
        Function selects random datasets for every cell name (without nested subtypes) without replacement
        :param annotation: pandas dataframe with colum 'Dataset' and samples as index
        :param bootstrap_fraction: fraction of datasets to select
        :returns: list of sample indexes for selected datasets
        """
        selected_samples = []
        values = annotation[selector_col].unique()
        for value in values:
            value_inds = annotation.loc[annotation[selector_col] == value].index
            value_inds = value_inds.difference(selected_samples)
            cell_datasets = set(annotation.loc[value_inds, 'Dataset'].unique())
            if cell_datasets:
                selected_datasets = np.random.choice(list(cell_datasets),
                                                     int(len(cell_datasets) * bootstrap_fraction) + 1,
                                                     replace=False)
                selected_samples.extend(annotation[
                    annotation['Dataset'].isin(selected_datasets)].index.intersection(value_inds))
        return selected_samples
