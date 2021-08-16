import pandas as pd 

def renorm_expressions(expr, genes_in_expressions_file):
    genes_in_expressions = []
    with open(genes_in_expressions_file, 'r') as file:
        for line in file:
            gene = line[:-1]
            genes_in_expressions.append(gene)
    genes = list(set(expr.index) & set(genes_in_expressions))
    renormed_expr = expr.loc[genes].div(expr.loc[genes].sum(axis=0), axis='columns') * 1e6
    return renormed_expr

def tr_to_genes(expr_data,
                tr_ids_path,
                id2_path='data/id2gene_gencode23_uniq.txt'):
    tr = pd.read_csv(tr_ids_path, sep='\t', header=None, index_col=0).index
    id2 = pd.read_csv(id2_path, sep='\t', index_col=0).Gene
    expr_data = expr_data.loc[expr_data.index.isin(tr)]
    expr_data = expr_data.assign(Gene = id2.loc[expr_data.index]).groupby('Gene').sum()
    expr_data = expr_data.divide(expr_data.sum())*1000000
    return expr_data
