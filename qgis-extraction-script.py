import os
from datetime import datetime
from qgis.core import QgsProject
from qgis.core import QgsApplication
from qgis.core import QgsCoordinateReferenceSystem
import processing
from processing.core.Processing import Processing

def disp_algs():
    """
    Display all available processing algorithms.

    :return:
    """
    for alg in QgsApplication.processingRegistry().algorithms():
        print("{}:{} --> {}".format(alg.provider().name(), alg.name(), alg.displayName()))


def make_samples(aom_path_list, aom_name_list, output_dir, input_layer_name):
    """
    Use the 'gdal:cliprasterbymasklayer' to extract AOMs by iterating over a list of file paths for the AOM shape
    files.

    :param aom_list: A list of the AOM shapefiles file paths.
    :param output_dir: The directory path to extracted AOMs
    :param input_layer_name: The layer from which the AOMs will be extracted.
    :return: None
    """
    for (aom_path, aom_name) in zip(aom_path_list, aom_name_list):

        print('[INFO] Output_dir: {0}'.format(output_dir))

        output_path = os.path.join(output_dir, "{0}.tif".format(aom_name[:-4]))

        parameters = {
            'ALPHA_BAND': False,
            'CROP_TO_CUTLINE': True,
            'DATA_TYPE': 1,
            'INPUT': input_layer_name,
            'KEEP_RESOLUTION': True,
            'MASK': aom_path,
            'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:3857'),
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:3857'),
            'NODATA': None,
            'SET_RESOLUTION': False,
            'X_RESOLUTION': None,
            'Y_RESOLUTION': None,
            'OUTPUT': output_path,
        }

        print('[INFO] Writing AOM to: {0}'.format(output_path))

        processing.run('gdal:cliprasterbymasklayer', parameters)


if __name__ == '__main__':

    # Choose a QGIS map to work from.
    my_map = '/home/will/USDA_PIVOT_MAP_2019/usda_pivot_map_2019.qgz'

    # Get date to tag output.
    raw_time = datetime.now()
    formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

    # Extracted samples output directory
    output_directory = '/home/will/peanut_growth_curves/extracted_aoms'

    # Create a list of raster layers from which the sample spaces will be extracted.
    raster_directory_path = '/home/will/USDA_PIVOT_MAP_2019/tif_composites'
    raster_filename_list = [s for s in os.listdir(raster_directory_path) if s.endswith('.tif')]
    raster_filepath_list = [os.path.join(raster_directory_path, s) for s in raster_filename_list]

    for s in raster_filepath_list:
        print('[INFO] Extraction layer found: {0}'.format(s))

    # Create a list of the vector files that will be used as the mask.
    vector_directory_path = '/home/will/peanut_growth_curves/individual_aom_shapefiles'
    vector_filename_list = os.listdir(vector_directory_path)
    vector_filepath_list = [os.path.join(vector_directory_path, s) for s in vector_filename_list]

    for s in vector_filename_list:
        print('[INFO] Mask layer found: {0}'.format(s))

    # Create a reference to the QGIS application.
    qgs = QgsApplication([], False)

    # Load providers.
    qgs.initQgis()

    # Create a project instance.
    project = QgsProject.instance()

    # Load a project.
    project.read(my_map)

    # Initialize processing.
    Processing.initialize()

    # Create an out sub-directory.
    for (raster, raster_path) in zip(raster_filename_list, raster_filepath_list):
        directory_path = os.path.join(output_directory, "{0}-extractions".format(raster[:-4]))
        print('[INFO] Creating output directory at: {0}'.format(directory_path))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Process sample spaces.
        params = {
            'aom_path_list': vector_filepath_list,
            'aom_name_list': vector_filename_list,
            'output_dir': directory_path,
            'input_layer_name': raster_path,
        }

        make_samples(**params)

        # Write a meta-data file with the details of this extraction for future reference.
        with open(os.path.join(directory_path, "sample_meta_data.txt"), "w") as tester:
            tester.write("""Sample Layer ID: {0}\n
                            Number of Samples: {1}\n
                            Samples Generated On: {2}\n
                            """.format(raster_path, len(vector_filepath_list), formatted_time))

    # Close project.
    qgs.exitQgis()
