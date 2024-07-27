import logging
from .h4am_reader import H4AM_Reader


__generator = {
    'H4AM': H4AM_Reader,
}


def create(args):
    dataset = 'H4AM'
    dataset_args = args.dataset_args[dataset]
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](args, **dataset_args)
