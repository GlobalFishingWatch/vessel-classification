import os

this_dir = os.path.abspath(os.path.dirname(__file__))
classification_dir = os.path.abspath(os.path.join(this_dir, '../classification'))
pipeline_dir = os.path.abspath(os.path.join(this_dir, '../pipeline'))
top_dir = os.path.abspath(os.path.join(this_dir, '../..'))
treniformis_dir = os.path.abspath(os.path.join(top_dir, 'treniformis'))
logdir = os.path.abspath(os.path.join(this_dir, '../logs'))