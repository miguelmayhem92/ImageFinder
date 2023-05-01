import pandas as pd
import os
from configs import configs

my_local_path = os.getcwd()
target_file = configs.csv_file 
raw_data = configs.raw_data_interview 
clean_data_name = configs.clean_data_interview

csv_path = my_local_path + f"/{target_file}{raw_data}"
image_id = pd.read_csv(csv_path,sep = ',', error_bad_lines=False)
image_id['jpgName'] = image_id.url.str.split('/',0).str[-1]

data_int_no_dup = image_id.groupby(['url','id'], as_index = False).agg(count = ('id','count')).sort_values('count',ascending = True)
data_int_no_dup['jpgName'] = data_int_no_dup.url.str.split('/',0).str[-1]
data_int_no_dup = data_int_no_dup.drop(columns = ['count'])

data_int_no_dup.to_csv(my_local_path + f"/{target_file}{clean_data_name}", index = False, header = True)