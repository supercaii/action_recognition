import logging

from .graphs import Graph
from .h4am_feeder import H4AM_Feeder, NTU_Location_Feeder


__data_args = {
    'H4AM': {'class': 12, 'shape': [3, 6, 1200, 32, 1], 'feeder': H4AM_Feeder}
}


def create(dataset, root_folder, transform, num_frame, inputs, **kwargs):
    graph = Graph(dataset)
    try:
        data_args = __data_args[dataset]
        data_args['shape'][0] = len(inputs)
        data_args['shape'][2] = num_frame
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.foramt(dataset))
        raise ValueError()
    if transform:
        dataset_path = '{}/transformed/{}'.format(root_folder, dataset)
    else:
        dataset_path = '{}/original/{}'.format(root_folder, dataset)
    kwargs.update({
        'dataset_path': dataset_path,
        'inputs': inputs,
        'num_frame': num_frame,
        'connect_joint': graph.connect_joint,
    })
    feeders = {
        'train': data_args['feeder']('train', **kwargs),
        'eval' : data_args['feeder']('eval', **kwargs),
    }
    feeders.update({'location': NTU_Location_Feeder(data_args['shape'])})
    return feeders, data_args['shape'], data_args['class'], graph.A, graph.parts
