import yaml
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
import pandas as pd

def get_proportions_series(cell_types):
    return pd.Series({cell_type.name: cell_type.cell_proportion for cell_type in cell_types}).dropna()

class CellType:
    def __init__(self, name: str, genes: list, parent_type: str,
                 cell_proportion: Union[float, None],
                 rna_per_cell: Union[float, None], 
                 model: Union[str, None]):
        self.name = name
        self._genes = sorted(list(set(genes)))
        self.cell_proportion = cell_proportion
        self.rna_per_cell = rna_per_cell
        self.parent_type = parent_type
        self.model = model
        
    def __repr__(self):
        return f'{self.name}_object'

    @property
    def genes(self):
        return self._genes

class CellTypes:
    """
    Class stores cell types properties and hands calculations on the types tree
    """
    def __init__(self, types_list: list, show_tree=False):
        """
        Generates cell types tree from list 
        :param cell_types: list with CellType objects
        """
        self._types_dict = {cell_type.name: cell_type for cell_type in types_list}
        self._types_tree = nx.DiGraph() 
        self._names = sorted([cell_type.name for cell_type in self])
        self._models = sorted([cell_type.name for cell_type in self if cell_type.model])
        self._genes = sorted(list(set([gene for cell_type in self for gene in cell_type.genes if cell_type.name in self.models])))
        self._root = [cell.name for cell in self if not cell.parent_type][0]
        
        for cell_type in self:
            if cell_type.parent_type:
                self._types_tree.add_edge(cell_type.parent_type, cell_type.name)

        if not nx.is_tree(self.types_tree):
            raise Exception("The resulting graph has to be a tree")
    
    def get_parent(self, cell_type):
        """
        :param cell_type: str cell type name 
        :return: str cell type parent type
        """
        parent_list = list(self.types_tree.predecessors(cell_type))
        if parent_list==[]:
            return None
        else:
            return parent_list[0]
    
    def get_all_parents(self, cell_type):
        parents = []
        parent = self.get_parent(cell_type)
        while parent:
            parents.append(parent)
            parent = self.get_parent(parent)
        
        return parents
    
    def get_direct_subtypes(self, cell_type: str) -> list:
        """
        Returns list of direct subtypes(one level down) of given cell type
        :param cell_type: str cell type
        :return: list of direct subtypes
        """
        assert type(cell_type) == str, 'Argument `cell_type` must be a string'
        return list(self.types_tree.neighbors(cell_type))
    
    def get_all_subtypes(self, cell_type: str) -> list:
        """
        Returns ALL subtypes of the for given cell type
        :param cell_type: str cell type
        :return: list of all subtypes
        """
        assert type(cell_type) == str, 'Argument `cell_type` must be a string'
        subtypes = list(nx.algorithms.traversal.dfs_tree(self.types_tree, cell_type).nodes)
        subtypes.remove(cell_type)
        return subtypes
    
    @classmethod
    def load(cls, config, show_tree=False):
        with open(config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        types_list = [] 
        for ct in config.keys(): 
            ct_genes = list(set(config[ct]['genes']))
            types_list.append(
                CellType(name=ct, genes=ct_genes,
                         cell_proportion=config[ct]['cell_proportion'],
                         rna_per_cell=config[ct]['rna_per_cell'],
                         parent_type=config[ct]['parent_type'],
                         model=config[ct]['model']))
        
        return cls(types_list, show_tree)
    
    def __iter__(self):
        for cell_type in self._types_dict.values():
            yield cell_type

    def __getitem__(self, item):
        return self._types_dict[item]

    def __getattr__(self, item):
        return self._types_dict[item]

    @property
    def genes(self):
        return self._genes
    
    @property
    def model_genes(self):
        return self._model_genes
    
    @property
    def names(self):
        return self._names
    
    @property
    def models(self):
        return self._models
        
    @property
    def types_dict(self):
        return self._types_dict
    
    @property
    def types_tree(self):
        return self._types_tree
    
    @property
    def root(self):
        return self._root
