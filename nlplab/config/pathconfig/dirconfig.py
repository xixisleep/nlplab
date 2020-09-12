import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
roberta_dir = os.path.join(project_path,'nlplab/config/chinese_roberta_wwm_ext_L-12_H-768_A-12')
data_dir = os.path.join(project_path,'nlplab/data')
model_dir = os.path.join(project_path,'nlplab/weights')