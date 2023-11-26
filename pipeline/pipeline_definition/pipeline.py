#!/usr/bin/env python 
"""pipeline.py
pipeline definition. render component definitions from templates in components directory.
define pipeline from components.
"""

from kfp import dsl
from components import load_data, validate_data
import yaml

CONFIG_FILE_PATH = '../config/default_config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)


@dsl.pipeline
def training_pipeline(
    train_test_split_seed: int,
    test_split_data_fraction: float,

) -> None:
    
    arff_file_location = 'arff_file gcs_url or /gcs path'
    stratify_column_name = 'Class'
    schema_file_path = 'schema file path in gcs'

    load_data_task = load_data(
        input_file=arff_file_location,
        split_seed=train_test_split_seed,
        test_fraction=test_split_data_fraction,
        label_column=stratify_column_name)
    
    validate_train_data_task = validate_data(
        input_file_path=load_data_task.Output['output_train_path'],
        schema_file_path=schema_file_path
        )
    validate_test_data_task = validate_data(
        input_file_path=load_data_task.Output['output_test_path'],
        schema_file_path=schema_file_path
        )
    
 

# kfp v2 dsl

# import yaml
# import jinja2

# from typing import Dict
# import os

# config_file_path = '../config/default_config.yaml'
# with open(config_file_path,'r') as config_file:
#     config = yaml.safe_load(config_file)

# def load_components(
#     component_names: list[str],
#     component_dir: str) -> Dict[str,str]:
#     """_summary_

#     Args:
#         components (list[str]): _description_
#         component_dir (str): _description_

#     Returns:
#         Dict[str,str]: _description_
#     """

#     components = {}
#     for comp in component_names:
#         template_file_path = os.path.join(component_dir, f'{comp}.yaml' )
#         comp_template = jinja2.Template(open(template_file_path,'r').read())
#         component = comp_template.render(
#             project_id = config.get('project_id'),
#             ARTIFACT_REGISTRY_URI = config.get('artifact_registry'),
#             BASE_IMAGE_SHA = config.get('base_image_name'),
#             gcs_bucket = config.get('gcs_bucket'))
#         components[comp] = component
#     return components

# component_dir = '../components'
# components = load_components(['load_data','validate_data'], component_dir)

# print(components['load_data'])




