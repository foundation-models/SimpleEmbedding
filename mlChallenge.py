"""
https://docs.google.com/document/d/1b0n25RRhpL6_U2OQ_AfKoSYoa6sEQ-gzWwDGW4Jeqm0

switch to a different dataset
download from
https://www.kaggle.com/joniarroba/noshowappointments/downloads/KaggleV2-May-2016.csv/5
upload to google drive with sharable link
https://drive.google.com/file/d/1FemK0G2bpDf5J_950CULqVa7SwqtAFi4/view?usp=drivesdk
then get the link from https://sites.google.com/site/gdocs2direct/

In SQL
SELECT Column1, Column2, mean(Column3), sum(Column4)
FROM SomeTable
GROUP BY Column1, Column2

"""
import pandas as pd

data_set_url = "https://drive.google.com/uc?export=download&id=1FemK0G2bpDf5J_950CULqVa7SwqtAFi4"
df = pd.read_csv(data_set_url, compression='zip', parse_dates=['ScheduledDay', 'AppointmentDay'])
print(df.dtypes)
print(df.head())
print(df.shape)

from enum import Enum


class cols(Enum):
    NO_SHOW = 'No-Show'
    SCHEDULED_DAY = 'ScheduledDay'
    PATIENT_ID = 'PatientId'
    NEIGHBOURHOOD = 'Neighbourhood'

df['Noshow'] = df['No-show'] == 'Yes'

# Change NoShow from Yes No to 1 0


# for PatientId = 29872499824296
print('one patient', df[df['PatientId'] == 29872499824296])


print(df.set_index(cols.SCHEDULED_DAY.value).resample('M').size())
dfg = df.groupby([cols.PATIENT_ID.value, pd.Grouper(key=cols.SCHEDULED_DAY.value, freq='M')])


print('One record from groupby', dfg.get_group( (29872499824296, pd.Timestamp('2016-04-30')) ))
gsize = dfg.size()
import numpy as np

def f(group):
    return pd.DataFrame({'sample': ['xx']})#np.square(group.Age)})
#    return group[group[cols.NEIGHBOURHOOD.value] == 'PRAIA DO SUÁ']
#    return group[group[cols.NEIGHBOURHOOD.value].str.contains('JARDIM')]



five = gsize[gsize == 9]
five.rename(columns={0: 'nvisit'}, inplace=True)
import pdb; pdb.set_trace()

special = five.apply(f)
print(special.header(3))
noshow = df[df['No-show'] == 'Yes']
noshow.shape


import matplotlib.pylab as plt

plt.clf()
result = noshow.groupby(['Gender', 'Age']).size()
result.plot(kind='bar')
# .sum()['PatientId']
plt.show()

