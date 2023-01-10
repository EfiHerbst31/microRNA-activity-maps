import multiprocessing as mp
import os
from pathlib import Path
from time import time
import warnings

from absl import app
from absl import flags
from absl import logging
import pandas as pd

import constants
import utils


warnings.simplefilter(action='ignore', category=FutureWarning)

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'cpus', None, 
    ('Number of CPUs to use in parallel.'
     'default: all available cpus are used.')) 
flags.DEFINE_string(
    'data_path', None, 
    ('Path to dataset. '
     'if scRNAseq: path to dataset. '
     'if spatial: path to the folder containing '
     '\'spatial\' and \'filtered_feature_bc_matrix\' folders.'))
flags.DEFINE_string(
    'dataset_name', None, 
    ('Name of your dataset. '
     'This is used to generate results path'))
flags.DEFINE_boolean(
    'debug', False, 
    ('Produces debugging output.'))
flags.DEFINE_string(
    'miR_figures', 'top_10', 
    ('Which microRNAs activity maps to draw. '
     'Options: %s .' % ' or '.join(constants._SUPPORTED_DRAW)))
flags.DEFINE_list(
    'miR_list', None, 
    ('Comma-separated list of microRNAs to compute. '
     'default: all microRNAs are computed.'
     'Example use: --miR_list=hsa-miR-300,hsa-miR-6502-5p,hsa-miR-6727-3p'))
flags.DEFINE_boolean(
    'preprocess', False, 
    ('Performs additional preprocessing on scRNAseq data before computations, '
     'merges all tables found in \data_path\' and samples 10K columns.'))
flags.DEFINE_string(
    'results_path', None, 
    'Path to save results.')
flags.DEFINE_string(
    'species', 'homo_sapiens', 
    'Options: %s .' % ' or '.join(constants._SUPPORTED_SPECIES))
flags.DEFINE_float(
    'thresh', 0.00001, 
    ('Threshold of the microRNA activity p-value. '
     'if a microRNA receives a lower score than \'thresh\' it is considered active. '
     'used in order to find the most active microRNAs across the provided data.'))

flags.register_validator('species', 
                         lambda value: value in constants._SUPPORTED_SPECIES,
                         message=('Either %s are supported.' % 
                                  ' or '.join(constants._SUPPORTED_SPECIES)))
flags.register_validator('miR_figures', 
                         lambda value: value in constants._SUPPORTED_DRAW,
                         message=('Either %s are supported.' %
                                  ' or '.join(constants._SUPPORTED_DRAW)))
      
flags.mark_flag_as_required('dataset_name')
flags.mark_flag_as_required('data_path')

def main(argv):

    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        logging.debug('Debug mode is on')
    else:
        logging.set_verbosity(logging.INFO)

    logging.info('Dataset name: %s', FLAGS.dataset_name)
    logging.info('Path to dataset: %s', FLAGS.data_path)

    if FLAGS.results_path:
        results_path = os.path.join(FLAGS.results_path, FLAGS.dataset_name)
    else:
        results_path = os.path.join(FLAGS.data_path, 'results')
    logging.info('Results path: %s', results_path)

    cpus = FLAGS.cpus or mp.cpu_count()
    logging.info('Using %i cpus', cpus)
    
    data_type = utils.check_data_type(FLAGS.data_path)
    logging.info('Data type detected: %s', data_type)
    
    mti_data = utils.mti_loader(FLAGS.species)
    miR_list = list(FLAGS.miR_list) or list(set(mti_data["miRNA"]))
    logging.info('Number of microRNAs detected: %i',len(miR_list))
    logging.debug('MicroRNA list: %s', miR_list)
    if  FLAGS.species == 'homo_sapiens' and 'hsa' not in miR_list[0]:
        raise utils.UsageError('species is \'homo_sapiens\' but some of the '
                               'microRNAs in the list do not belong to humans')
    elif FLAGS.species == 'mus_musculus' and 'mmu' not in miR_list[0]:
        raise utils.UsageError('species is \'mus_musculus\' but some of the '
                               'microRNAs in the list do not belong to mice')
    else:
        logging.debug('Species match microRNAs in the list')

    if data_type == 'spatial':
        counts = utils.visium_loader(FLAGS.data_path)
        spatial_coors = utils.get_spatial_coors(FLAGS.data_path, counts)
        spots = spatial_coors.shape[0]
    else:
        if FLAGS.preprocess:
            counts = utils.scRNAseq_preprocess_loader(
                FLAGS.dataset_name, FLAGS.data_path)
        else:
            counts = utils.scRNAseq_loader(FLAGS.data_path)

    counts_norm = utils.normalize_counts(counts)
    
    start = time() 
    miR_activity_pvals = utils.compute_mir_activity(
        counts_norm, miR_list, mti_data, results_path, cpus, FLAGS.debug)
    logging.info('Computation time: %f minutes', (time() - start)//60)
    
    if data_type == 'spatial':
        logging.info('Generating activity map figures')
        if FLAGS.miR_figures == constants._DRAW_BOTTOM_10 or len(miR_list) <= 10:
            miR_list_figures = miR_list
        elif FLAGS.miR_figures == constants._DRAW_TOP_10:
            mir_activity = utils.sort_activity_spatial(
                miR_activity_pvals, FLAGS.thresh, spots, 
                results_path, FLAGS.dataset_name)
            miR_list_figures = mir_activity.index[:10].tolist()
            logging.debug('Figures are produced for the following microRNAs: %s', 
                          miR_list_figures)
        else: #'bottom_10'
            mir_activity = utils.sort_activity_spatial(
                miR_activity_pvals, FLAGS.thresh, spots, 
                results_path, FLAGS.dataset_name)
            miR_list_figures = mir_activity.index[-10:].tolist()
            logging.debug('Figures are produced for the following microRNAs: %s', 
                          miR_list_figures)
        utils.produce_spatial_maps(
            miR_list_figures, miR_activity_pvals, spatial_coors, 
            results_path, FLAGS.dataset_name)

if __name__ == '__main__': 
    app.run(main)