
import pandas as pd
from src.dataset import DATASETS_TABLE
from src.recommenders import RECOMMENDERS_TABLE
import src as kw
import argparse

from typing import TypedDict

class InputInfo(TypedDict):
    name: str
    description: str
    options: pd.DataFrame
    id_column: str
    name_column: str


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

def get_input(description: str, inputs_info: 'list[InputInfo]') -> 'list[list[int]]':
    '''
    Coleta as opções escolhidas pelo usuário (por meio do input ou comandos de linha)

    return:
        Lista de listas de inteiros, onde cada lista de inteiros representa as opções escolhidas pelo usuário para um InputInfo
    '''
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for input_info in inputs_info:
        parser.add_argument('--{}'.format(input_info['name']), type=str, default=None, help=input_info['description'])

    args = vars(parser.parse_args())

    options = []
    for input_info in inputs_info:

        current_options = []
        current_arg = args[input_info['name']]

        if current_arg is None:
            current_options = ask_options(input_info['name'], input_info['options'], input_info['name_column'])
        elif current_arg == 'all':
            current_options = input_info['options'].index.tolist()
        elif current_arg.replace(" ", "").isdigit():
            current_options = []
            current_arg = current_arg.split(' ')
            print(current_arg)
            for option in current_arg:
                if int(option) not in input_info['options'].index:
                    raise ValueError('{} index {} not found!'.format(input_info['name'], option))
                current_options.append(int(option))
        else:
            current_options = []
            options_names = current_arg.split(' ')
            for option_name in options_names:
                if not input_info['options'][input_info['name_column']].str.contains(option_name).any():
                    raise ValueError('{} {} not found!'.format(input_info['name'], option_name))
                current_options.append(input_info['options'][input_info['options'][input_info['name_column']].str.contains(option_name)].index.tolist()[0])
        
        print(current_options)
        options.append(list(set(current_options)))

    return options


def get_algo_and_dataset_parameters(description: str) -> 'list[list[int]]':
    '''
    Coleta as opções de algoritmos e datasets escolhidos pelo usuário (por meio do input ou comandos de linha)

    return:
        Tupla com duas listas de inteiros, a primeira com os algoritmos escolhidos e a segunda com os datasets escolhidos
    '''
    inputs_info = [
        {
            'name': 'algorithms',
            'description': 'Algorithm names (or indexes) to execute. If not provided, a interactive menu will be shown. If "all" is provided, all algorithms will be executed.',
            'options': RECOMMENDERS_TABLE,
            'id_column': kw.RECOMMENDER_ID,
            'name_column': kw.RECOMMENDER_NAME
        },
        {
            'name': 'datasets',
            'description': 'Dataset names (or indexes) to preprocess. If not provided, a interactive menu will be shown. If "all" is provided, all datasets will be preprocessed.',
            'options': DATASETS_TABLE,
            'id_column': kw.DATASET_ID,
            'name_column': kw.DATASET_NAME
        }
    ]
    return get_input(description, inputs_info)