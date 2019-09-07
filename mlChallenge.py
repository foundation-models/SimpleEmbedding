"""
https://docs.google.com/document/d/1b0n25RRhpL6_U2OQ_AfKoSYoa6sEQ-gzWwDGW4Jeqm0

switch to a different dataset
https://www.kaggle.com/joniarroba/noshowappointments/downloads/KaggleV2-May-2016.csv/5
"""
import pandas as pd
# first reading csv files
df = pd.read_csv('patients.csv', parse_dates=['ScheduledDay', 'AppointmentDay'])
print(df.dtypes)
print(df.head())

import pdb; pdb.set_trace()