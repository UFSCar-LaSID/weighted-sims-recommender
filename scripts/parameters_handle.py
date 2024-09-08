
import pandas as pd


def _display_options(options_table: pd.DataFrame, name_column: str):
    '''
    Printa as alternativas disponíveis para o usuário escolher

    params:
        options_table: Tabela com as opções disponíveis (precisa ter a coluna name_column)
        name_column: Nome da coluna que contém o nome da opção
    '''
    for idx, row in options_table.iterrows():        
        print('[{}] {}\n'.format(idx, row.loc[name_column]))


def ask_options(options_name: str, options_table: pd.DataFrame, name_column: str) -> 'list[int]':
    '''
    Pergunta e coleta as opções escolhidas pelo usuário.

    params:
        options_name: Nome do que o usuário está escolhendo (ex: 'algorithm', 'dataset', etc.)
        options_table: Tabela com as opções disponíveis (precisa ter a coluna name_column). Cada linha da tabela é uma opção.
        name_column: Nome da coluna que contém o nome da opção
    
    return:
        Lista de inteiros com as opções escolhidas pelo usuário
    '''
    print('\nAvailable {}:\n'.format(options_name))
    _display_options(options_table, name_column)
    options = input('Select which {} to execute: '.format(options_name))
    return list(set(map(int, options.split(' '))))