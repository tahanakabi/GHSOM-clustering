import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.preprocessing import MinMaxScaler
import math


df2 = pd.read_excel('average_loads.xlsx')
input_data = np.array(df2.ix[:,:])
scaler = MinMaxScaler().fit(input_data)

df3 = pd.read_excel('load_shape_factors.xlsx')
input_data1 = np.array(df3.ix[:,:])
scaler1 = MinMaxScaler().fit(input_data1)


# costumers= [6002,6005,6020,6021,6026,6027,6029,6037,6038,6041]
# costumers= set(df['id'].values)
# df0=pd.DataFrame()
# for id in costumers:
#     try:
#         df0[str(id)]= df[df['id']==id].ix[:,'load'][0:10000].values
#     except ValueError:
#         pass

# df1=pd.DataFrame()
# for col in df0.columns:
#     loadlist=[]
#     for i in range(0,10000,48):
#         loadlist.append(sum(df0.ix[i:i+48,col].values))
#     df1[col]=loadlist

# df2=pd.DataFrame()
# for col in df0.columns:
#     load_list = []
#     for i in range(0,10000,48)[:-1]:
#         load_list.append(np.array(list(df0.ix[i :i + 47, col].values)))
#     load_array= np.array(load_list)
#     mean_loads=np.mean(load_array, axis=0)
#     df2[col]=mean_loads


# df0=df0.transpose()
#
# writer = pd.ExcelWriter('loads.xlsx')
# df0.to_excel(writer,'Sheet1')
# writer.save()

# df1=df1.transpose()
#
# writer1 = pd.ExcelWriter('daily_loads.xlsx')
# df1.to_excel(writer1,'Sheet1')
# writer1.save()

# df2=df2.transpose()
#
# writer1 = pd.ExcelWriter('average loads.xlsx')
# df2.to_excel(writer1,'Sheet1')
# writer1.save()
clusters3=[[6024,
6006
],[6007
],[6025,
6028,
6036,
6000
],[6018,
6019,
6040,
6003,
6004,
6008,
6010
],[6031,
6005
],[],[],[6016,
6023,
6026,
6027,
6033,
6011
],[6021,
6034,
6038,
6039,
6002,
6009,
6012,
6013
],[ 6020,
6029,
6030,
6035
],[6032,
6037,
6041,
6001
],[6017,
6014
],[],[6022
]]

clusters=[[6001,6007,6021,6035,6039,6030,6029],[6002,6013,6009,6012,6023],[6008],[6004],[6032],[6016,6024,6027,6033,6006,6026,6034,6038,6011],[6031],[6010,6003],[6014,6022],[6025],[6041],[6028, 6005],[6036, 6019, 6000,6018,6040],[6017, 6020, 6037]]
clusters1=[[6040],[6024
],[],[6036,
6003,
6006,
6010,
], [6018,
6019,
6025,
6031,
6000,
6008
],[6005], [],[6016,
6020,
6021,
6023,
6030,
6032,
6033,
6035,
6037,
6039,
6041,
6002,
6009,
6012,
6013
],[6027,
6004,
6011
],[],[6017,
6026,
6028,
6029,
6038,
6014
],[6007
],[],[6022,
6034,
6001
]]
clusters2=[[6024,6040,6005,6006,6007],
           [],[],[],[6018,6019,6025,6028,6031,6036,6000,6003,6004,6008,6010],
           [],[],[6016,6020,6021,6023,6026,6029,6030,6032,6033,6034,6035,6037,6038,6039,6001,6002,6009,6011,6012,6013,6014],
           [6027,6041],
           [],[6017],[],[],[6022]]

clusters4 = [[],[6024,
6037,
6040,
6041,
6002,
6012,
6013
],[],[6028,
6006,
6007,
6010
],[6018,
6025,
6036,
6000,
6003,
6004,
6008
],[6019,
6031,
6005
],[],[6016,
6017,
6020,
6021,
6029,
6030,
6035,
6038,
6039,
6011
],[6027,
6009
],[],[6023,
6026,
6032,
6033,
6034,
6001,
6014
],[],[],[6022
]]
clusters5=[[],[6041,
6003,
6004,
],[6010
],[6019,
6024,
6031
],[6025,
6027,
6000,
6008
],[],[],[6016,
6017,
6018,
6020,
6021,
6026,
6028,
6030,
6033,
6039,
6040,
6007,
6011,
6013
],[6038,
6005
],[],[6022,
6029,
6035
],[6023,
6037,
6009,
6014
],[6036,
6002,
6006,
6012
],[6032,
6034,
6001,
]]

distances=[]
for cls in clusters:
    # tempo=[str(s) for s in str
    if len(cls)>0:
        cls_array=np.array(df3.ix[cls, :])
        cls_array=scaler1.transform(cls_array)
        avg =np.average(cls_array,axis=0)
        distance=[]
        for i, col in enumerate(cls_array):
            distance.append(np.linalg.norm(col - avg)**2)
        distances.append(np.average(np.array(distance)))
    else:
        distances.append(0.0)
MIA=math.sqrt(np.average(np.array(distances)))

print(MIA)


# array_distances=np.array(distances)
# workbook = xlsxwriter.Workbook('distances' + ".xlsx")
# worksheet = workbook.add_worksheet()
#
# for col, data in enumerate(array_distances):
#     worksheet.write_column(0, col, data)
#
# workbook.close()




