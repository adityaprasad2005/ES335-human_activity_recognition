# This file is used to combine the data from the different folders into one dataset

import numpy as np
import pandas as pd
import os

# get the current dir
current_dir = os.path.dirname(os.path.abspath(__file__))

# the list of all files in the current dir
files = os.listdir(current_dir)

activity_code = {"Walking":1,"Upstair":2,"Downstair":3,"Sitting":4,"Standing":5,"Laying":6}

combined_df = pd.DataFrame()
y_df = pd.DataFrame()

for file in files:
    if file.endswith('.csv'):
        activity = file.strip(".csv").split("_")[1]
        print(activity)
        code =  activity_code[activity]

        df = pd.read_csv(os.path.join(current_dir,file),sep=",",header=0)
    
        df= df.iloc[100:600, 1:4] # remove the first 100 rows and the last 400 rows and select the first 3 columns
        
        combined_df = pd.concat([combined_df, df], axis=0)
        y_df = pd.concat([y_df, pd.DataFrame([code])], axis=0)

combined_df.to_csv(os.path.join(current_dir, 'combined_X_data.csv'), index=False)
y_df.to_csv(os.path.join(current_dir, 'combined_y_data.csv'), index=False)
        