import multiprocessing as mp
import os
from pathlib import Path
from time import time
from typing import Optional, Tuple
import warnings

from absl import app
from absl import flags
from absl import logging
import pandas as pd

import constants
import utils

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'cpus', None, 
    'Number of CPUs to use in parallel. default: all available cpus are used.')
flags.DEFINE_string(
    'data_path', None, 
    ('Path to dataset. if %s: path to dataset. if %s: path to the folder containing %s '
     'folders.', (constants._DATA_TYPE_SINGLE_CELL, constants._DATA_TYPE_SPATIAL,
     ' and '.join(constants._SPATIAL_FOLDERS))))
flags.DEFINE_string(
    'dataset_name', None, 
    'Name of your dataset. This is used to generate results path')
flags.DEFINE_boolean(
    'debug', False, 'Produces debugging output.')
flags.DEFINE_string(
    'miR_figures', constants._DRAW_TOP_10, 
    'Which microRNAs activity maps to draw. Options: %s .' % ' or '.join(constants._SUPPORTED_DRAW))
flags.DEFINE_list(
    'miR_list', None, 
    ('Comma-separated list of microRNAs to compute. default: all microRNAs are computed.'
     'Example use: -miR_list=hsa-miR-300,hsa-miR-6502-5p,hsa-miR-6727-3p'))
flags.DEFINE_list(
    'populations', None,
    ('Comma-separated list of two population string identifiers embedded in cell id.' 
    'default: None. Example use: -populations=\'DESEASE_\',\'CONTROL_\''))
flags.DEFINE_boolean(
    'preprocess', False, 
    ('Performs additional preprocessing on %s data before computations, merges all '
     'tables found in \data_path\' and samples 10K columns.', constants._DATA_TYPE_SINGLE_CELL))
flags.DEFINE_string(
    'results_path', None, 
    'Path to save results.')
flags.DEFINE_string(
    'species', constants._SPECIES_HOMO_SAPIENS, 
    'Options: %s .' % ' or '.join(constants._SUPPORTED_SPECIES))
flags.DEFINE_float(
    'thresh', constants._ACTIVITY_THRESH, 
    ('Threshold of the microRNA activity p-value. If a microRNA receives a lower score than '
     ' \'thresh\' it is considered active. used in order to find the most active microRNAs '
     'across the provided data.'))

flags.register_validator('species', 
                         lambda value: value in constants._SUPPORTED_SPECIES,
                         message=('Either %s are supported.', 
                                  ' or '.join(constants._SUPPORTED_SPECIES)))
flags.register_validator('miR_figures', 
                         lambda value: value in constants._SUPPORTED_DRAW,
                         message=('Either %s are supported.',
                                  ' or '.join(constants._SUPPORTED_DRAW)))

flags.mark_flag_as_required('dataset_name')
flags.mark_flag_as_required('data_path')

def data_handling(data_path: str, dataset_name: str, data_type: Optional[str], 
    preprocess: Optional[bool]=True) -> pd.DataFrame:
    '''Loading, preprocessing (if needed) and normalizing data.

    Args:
        data_path: path to data.
        dataset_name: dataset name.
        data_type: (optional) data type 'spatial' or 'scRNAseq'.
        preprocess: (optional) if True, performing data preprocessing if data is too big or is
            composed of multiple files. If False, will not perform data preprocessing. 

    Returns:
        counts_norm: normalized reads table
        counts: raw reads table
    '''
    if not data_type:
       data_type =  utils.check_data_type(data_path)

    if data_type == constants._DATA_TYPE_SPATIAL:
        counts = utils.visium_loader(data_path)
    else:
        if preprocess:
            counts = utils.scRNAseq_preprocess_loader(dataset_name, data_path)
        else:
            counts = utils.scRNAseq_loader(data_path)
    
    counts_norm = utils.normalize_counts(counts)
    return counts_norm, counts

def computing_mir_activity(counts_norm: pd.DataFrame, results_path: str, miR_list: Optional[list], 
    cpus: Optional[int], species: Optional[str]=constants._SPECIES_HOMO_SAPIENS, 
    debug: Optional[bool]=False) -> Tuple[list, pd.DataFrame]:
    '''Computing microRNA activity across the data.

    Args:
        counts_norm: normalized reads table.
        results_path: path to save results.
        miR_list: (optional) list of microRNAs to compute.
        cpus: (optional) amount of cpus to use in parallel.
        species: (optional) either 'homo_sapiens' (default) or 'mus_musculus' are supported. 
        debug: (optional) if True, provides aditional information. Default=False.
    
    Returns:
        miR_activity_pvals: microRNA activity score per cell/spot.
    '''
    cpus = cpus or mp.cpu_count()
    logging.info('Using %i cpus', cpus)

    mti_data, miR_list = utils.mir_data_loading(
        miR_list=miR_list, 
        species=species, 
        debug=debug)
    
    start = time() 
    miR_activity_pvals = utils.compute_mir_activity(
        counts_norm, 
        miR_list, 
        mti_data, 
        results_path, 
        cpus, 
        debug=debug)
    logging.info('Computation time: %f minutes', (time() - start)//60)

    return miR_list, miR_activity_pvals


def mir_post_processing_spatial(data_path: str, counts_norm: pd.DataFrame, miR_activity_pvals: pd.DataFrame, 
    miR_list: list, results_path: str, dataset_name: str, data_type: Optional[str],
    miR_figures: Optional[str]=constants._DRAW_TOP_10, 
    thresh: Optional[float]=constants._ACTIVITY_THRESH) -> None:
    '''Perfoms post processing on spatial data.

    First sorting microRNAs by their overall level of activity, 
    and then plotting according to user's requirement.
    If there are <= 10  microRNAs, or the user wants plots for all microRNAs, 
    the function produces activity maps for all microRNAs without sorting first.

    Args:
        data_path: path to data.
        counts_norm: normalized reads table.
        miR_activity_pvals: microRNA activity results per spot.
        miR_list: list of microRNAs.
        results_path: path to save results.
        dataset_name: dataset name.         
        data_type: (optional) data type 'spatial' or 'scRNAseq'.
        miR_figures: which microRNAs to plot. 
        thresh: thresold to define what is considered active.

    Returns:
        None.

    Raises:
        UsageError if spatial data was not found in data_path.
    '''
    if not data_type:
       data_type =  utils.check_data_type(data_path)
    if data_type is not constants._DATA_TYPE_SPATIAL:
        raise utils.UsageError('No spatial data was found for post-processing in %s', data_path)
    logging.info('Generating activity map figures')
    spatial_coors = utils.get_spatial_coors(data_path, counts_norm)
    spots = spatial_coors.shape[0]

    mir_activity_list = utils.sort_activity_spatial(
            miR_activity_pvals, 
            thresh, 
            spots, 
            results_path, 
            dataset_name)

    miR_list_figures = utils.get_figure_list(
        miR_list,
        miR_figures,
        mir_activity_list)

    utils.produce_spatial_maps(
        miR_list_figures, 
        miR_activity_pvals, 
        spatial_coors, 
        results_path, 
        dataset_name, 
        mir_activity_list)

def mir_post_processing_sc(data_path: str, counts: pd.DataFrame, miR_activity_pvals: pd.DataFrame, 
    miR_list: list, results_path: str, dataset_name: str, data_type: Optional[str],
    miR_figures: Optional[str]=constants._DRAW_TOP_10, populations: Optional[list]=None) -> None:
    '''Perfoms post processing on scRNAseq data.

    Computes UMAP based on gene expression, sorts microRNAs by their overall level of activity, 
    and plots UMAP according to user's requirement.
    If there are <= 10  microRNAs, or the user wants plots for all microRNAs, 
    the function produces activity maps for all microRNAs without sorting first.

    Args:
        data_path: path to data.
        counts: raw reads table.
        miR_activity_pvals: microRNA activity results per spot.
        miR_list: list of microRNAs.
        results_path: path to save results.
        dataset_name: dataset name.         
        data_type: (optional) data type 'spatial' or 'scRNAseq'.
        miR_figures: (optional) which microRNAs to plot. 
        populations: (optional) list of two population string identifiers embedded in cell id.

    Returns:
        None.

    Raises:
        UsageError if spatial data was not found in data_path.
    '''
    if not data_type:
       data_type =  utils.check_data_type(data_path)
    if data_type is not constants._DATA_TYPE_SINGLE_CELL:
        raise utils.UsageError('No scRNAseq data was found for post-processing in %s', data_path)
    logging.info('Single cell post processing')
    enriched_counts = utils.generate_umap(
        counts, 
        miR_activity_pvals,
        populations)

    mir_activity_list = utils.sort_activity_sc(
            miR_activity_pvals, 
            populations)

    miR_list_figures = utils.get_figure_list(
        miR_list,
        miR_figures,
        mir_activity_list)

    utils.plot_sc(
        miR_list_figures, 
        enriched_counts,
        results_path, 
        dataset_name, 
        mir_activity_list,
        miR_activity_pvals,
        populations)


def compute(data_path: str, dataset_name: str, miR_list: Optional[list], cpus: Optional[int],
    results_path: Optional[str], species: Optional[str]=constants._SPECIES_HOMO_SAPIENS,  
    miR_figures: Optional[str]=constants._DRAW_TOP_10, 
    preprocess: Optional[bool]=True, thresh: Optional[float]=constants._ACTIVITY_THRESH,
    populations: Optional[list]=None, debug: Optional[bool]=False):
    '''Performing end-to-end microRNA activity map computation.

    Loading spatial/scRNAseq data and preprocessing if needed.
    Computing microRNA for all spots/cells.
    Saving results locally and producing maps for spatial data.

    Args:
        data_path: path to data.
        dataset_name: dataset name. 
        miR_list: (optional) list of microRNAs to compute.
        cpus: (optional) amount of cpus to use in parallel.
        results_path: (optional) path to save results.
        species: (optional) either 'homo_sapiens' (default) or 'mus_musculus' are supported. 
        miR_figures: (optional) which microRNAs to plot. 
        preprocess: (optional) if True, performing data preprocessing if data is too big or is
            composed of multiple files. If False, will not perform data preprocessing. 
        thresh: (optional) thresold to define what is considered active.
        populations: (optional) list of two population string identifiers embedded in cell id.
        debug: (optional) if True, provides aditional information. Default=False.

    Returns:
        None
    '''
    if debug:
        logging.set_verbosity(logging.DEBUG)
        logging.debug('Debug mode is on')
    else:
        logging.set_verbosity(logging.INFO)

    logging.info('Dataset name: %s', dataset_name)
    logging.info('Path to dataset: %s', data_path)

    if results_path:
        results_path = os.path.join(results_path, dataset_name)
    else:
        results_path = os.path.join(data_path, 'results')
    logging.info('Results path: %s', results_path)
    
    data_type = utils.check_data_type(data_path)
   
    counts_norm, counts = data_handling(
        data_path, 
        dataset_name, 
        data_type=data_type, 
        preprocess=preprocess)

    miR_list, miR_activity_pvals = computing_mir_activity(
        counts_norm, 
        results_path, 
        miR_list=miR_list, 
        cpus=cpus, 
        species=species,
        debug=debug)

    if data_type == constants._DATA_TYPE_SPATIAL:
        mir_post_processing_spatial(
            data_path, 
            counts_norm, 
            miR_activity_pvals, 
            miR_list=miR_list, 
            results_path=results_path, 
            dataset_name=dataset_name,
            data_type=data_type,
            miR_figures=miR_figures, 
            thresh=thresh)
    else:
        mir_post_processing_sc(
            data_path, 
            counts, 
            miR_activity_pvals, 
            miR_list=miR_list, 
            results_path=results_path, 
            dataset_name=dataset_name,
            data_type=data_type,
            miR_figures=miR_figures,
            populations=populations)
    logging.info('Done.')

def main(argv):
    compute(
        data_path=FLAGS.data_path, 
        dataset_name=FLAGS.dataset_name, 
        miR_list=FLAGS.miR_list, 
        cpus=FLAGS.cpus, 
        results_path=FLAGS.results_path, 
        species=FLAGS.species, 
        miR_figures=FLAGS.miR_figures,
        preprocess=FLAGS.preprocess, 
        thresh=FLAGS.thresh,
        populations=FLAGS.populations,
        debug=FLAGS.debug)

if __name__ == '__main__': 
    app.run(main)