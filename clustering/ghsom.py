import numpy as np
import pandas as pd
import xlsxwriter
from som import SOM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# -----------------------------------Data pre-process-------------------------------------------------------
# get  training data
# df = pd.read_excel('../data/test_customers.xlsx')
# df = pd.read_excel('daily_loads.xlsx')
df = pd.read_excel('average_loads.xlsx')
# df = pd.read_excel('../data/Customers_data.xlsx')
# df1 = pd.read_excel('../data/WPG_data_test.xlsx')
# -----------------------------------input data random pre-process------------------------------------
df = df.sample(frac=1)
# df_ran1 = df1.sample(frac=1)

# define which title to be noimal

# df_nominal = df["ID"]
df_nominal = df.index
# df_numerical_tmp1 = df1.ix[:, ['OH WK', 'OH FCST WK', 'BL WK', 'BL FCST WK', 'Last BL', 'Backlog', 'BL <= 9WKs', 'DC OH', 'On the way', 'Hub OH', 'Others OH', 'Avail.', 'Actual WK', 'FCST WK', 'Actual AWU', 'FCST AWU', 'FCST M', 'FCST M1', 'FCST M2', 'FCST M3']]
df_numerical_tmp = df.ix[:,1:]
# df_numerical_tmp = df.ix[:, ['SEX','age','employment status','SOCIAL CLASS','people','people over 15','people under 15','accomodation']]
df_numerical = df_numerical_tmp.apply(pd.to_numeric, errors='coerce').fillna(-1)
# df_numerical1 = df_numerical_tmp1.apply(pd.to_numeric, errors='coerce').fillna(-1)



# get data dim to latter SOM prcess
input_dim = len(df_numerical.columns)
input_num = len(df_numerical.index)


# -----------------------------------input data random pre-process------------------------------------
# change data to np array (SOM accept nparray format)
input_data = np.array(df_numerical).astype(float)

scaler = StandardScaler().fit(input_data)
input_data = scaler.transform(input_data)



# input_data = np.array(df_numerical1)



# input_data = np.array([[2., 1., 1., 1., 3.],
#                         [2., 1., 0., 0., 3.],
#                         [2., 1., 0., 1., 2.],
#                         [2., 1., 0., 1., 3.],
#                         [4., 0., 0., 0., 2.],
#                         [4., 0., 1., 0., 1.],
#                         [4., 0., 1., 0., 3.],
#                         [4., 0., 1., 0., 2.],
#                         [0., 0., 1., 0., 3.],
#                         [0., 0., 1., 0., 2.]])
# input_dim = 5
# input_data = np.array([[10., 5., 5., 5., 15.],
#                        [10., 5., 0., 0., 15.],
#                        [10., 5., 0., 5., 10.],
#                        [10., 5., 0., 5., 15.],
#                        [20., 0., 0., 0., 10.],
#                        [20., 0., 5., 0., 5.],
#                        [20., 0., 5., 0., 15.],
#                        [20., 0., 5., 0., 10.],
#                        [0., 0., 5., 0., 15.],
#                        [0., 0., 5., 0., 10.]])

# calculate mqe with input,  if no result_index then cal mqe with all input
# @param result_index: python list format


def cal_clustered_mqe(input_data, result_index = None):

    mqe = []

    if result_index is None:
        return cal_mqe(input_data)
    else:
        for i in result_index:
            indices = i
            if indices == []:
                mqe.append(0.0)
            else:
                clustered_input = np.take(input_data, indices, 0)
                # print(clustered_input)
                mqe.append(cal_mqe(clustered_input))
    return mqe


def cal_mqe(input_data):
    # if input data equal 1 dimensions array , then expaned to 2 diemnsion  [2., 1., 0., 0., 3.] ==> [[2., 1., 0., 0., 3.]]
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)

    return np.mean(
                np.sqrt(
                    np.sum(
                        np.power(
                            np.subtract(
                                input_data,
                                np.stack( np.mean(input_data, axis=0) for i in range(len(input_data)) )
                            )
                        , 2)
                    , axis=1)
                )
            )

# call tensorflow som class
def call_som(m, n, dim, input_data, weight_after_insertion=None, alpha=None, sigma=None):
    som = SOM(m, n, dim, weight_after_insertion)
    trained_weight = som.train(input_data)
    mapped = som.map_vects(input_data)
    result = np.array(mapped)
    return trained_weight, result


def som_neuron_locations(m, n):
    for i in range(m):
        for j in range(n):
            yield np.array([i, j])


# get each som result cluster (node) contains how many input
def clustered_location_input_index(m, n, trained_weight, som_result_map, input_data):
    filter_map_value = np.array(list(som_neuron_locations(m, n)))

    # filter map output:
    # 3-D array
    # [[[0 0]   [[0 1]   [[1 0]   [[1 1]
    # [0 0]     [0 1]    [1 0]    [1 1]
    # [0 0]     [0 1]    [1 0]    [1 1]
    # [0 0]     [0 1]    [1 0]    [1 1]
    # [0 0]     [0 1]    [1 0]    [1 1]
    # [0 0]     [0 1]    [1 0]    [1 1]
    # [0 0]     [0 1]    [1 0]    [1 1]
    # [0 0]     [0 1]    [1 0]    [1 1]
    # [0 0]     [0 1]    [1 0]    [1 1]
    # [0 0]],   [0 1]]   [1 0]]   [1 1]]]
    # for i in range(0, m*n*2, 2):
    #     filter_map.append(np.stack( [np.squeeze(filter_map_value.reshape((1, -1))) for i in range(len(input_data))])[:,[i,i+1]])
    filter_map = []
    som_clusting_result_location_list = []
    for i in range(0, m*n*2, 2):
        filter_map_tmp = np.stack([np.squeeze(filter_map_value.reshape((1, -1))) for i in range(len(input_data))])[:,[i,i+1]]
        # print(filter_map_tmp)
        myarray_tmp = np.array(np.squeeze(np.array(np.where(np.all(som_result_map == filter_map_tmp, axis=1)))))

        if myarray_tmp.size == 1:
            som_clusting_result_location_list.append([myarray_tmp.tolist()])
        else:
            som_clusting_result_location_list.append(myarray_tmp.tolist())

    return som_clusting_result_location_list

# find which topology location is near to error unit
def find_neighborhood_location(topology_map, m, n, error_unit_location):
    stacked_error_unit = np.stack(error_unit_location  for i in range(m*n))

    # print(np.absolute(np.sum(np.subtract(stacked_error_unit,topology_map), axis=1)))
    neighborhood_location_index_tmp = np.sum(np.absolute(np.subtract(stacked_error_unit,topology_map)), axis=1)
    neighborhood_location_index = np.squeeze(np.array(np.where(neighborhood_location_index_tmp == 1)))

    return neighborhood_location_index


def get_dissimilar_weight_location(topology_map, error_unit_index, neighborhood_location_index, trained_weight):
    error_unit_weight = np.take(trained_weight, error_unit_index, 0)
    neighborhood_location_weight = np.take(trained_weight, neighborhood_location_index, 0)

    dissimilar_weight_location_tmp_index = np.argmax(
                                                np.sqrt(
                                                    np.sum(
                                                        np.power(
                                                            np.subtract(
                                                                np.stack([error_unit_weight for i in range(len(neighborhood_location_weight))])
                                                            , neighborhood_location_weight)
                                                        , 2)
                                                    ,axis=1)
                                                )
                                            )
    dissimilar_weight_location_index = np.take(neighborhood_location_index, dissimilar_weight_location_tmp_index)
    dissimilar_weight_location = np.take(topology_map, dissimilar_weight_location_index, 0)

    return dissimilar_weight_location, dissimilar_weight_location_index


def insert_units(slice_point, weight_topology_map):
    print('------after insertation -------')
    units_to_be_inserted = np.divide(np.add(np.take(weight_topology_map, slice_point, 0), np.take(weight_topology_map, slice_point-1, 0)), 2)
    new_weight_topology_map = np.insert(weight_topology_map, slice_point, units_to_be_inserted, 0)

    return new_weight_topology_map


def get_map_weight_after_unit_insertion(m, n, topology_map, error_unit_location, error_unit_index, dissimilar_weight_location, dissimilar_weight_location_index, trained_weight, input_dim):
    print('-----------get_map_weight_before_unit_insertion----------')
    print(trained_weight)
    print(topology_map)
    # print(error_unit_location)
    # print(dissimilar_weight_location)

    # check which layer should be insert
    if error_unit_index > dissimilar_weight_location_index:
        slice_point = error_unit_index
    else:
        slice_point = dissimilar_weight_location_index

    # check insert row or column
    if  np.argmax(np.absolute(np.subtract(error_unit_location, dissimilar_weight_location))) == 1:
        # insert one row
        print('insert row - add y direction')

        # slice point rem
        print('slice_point')
        slice_point = slice_point%n

        print(slice_point)

        print('-----------AfterReshape-----------')
        # rearrange weight array by row index
        trained_weight_default_index = np.arange(m*n)
        new_order = []

        for i in range(n):
            trained_weight_remainder = trained_weight_default_index%n
            new_order = np.append(new_order, np.where(trained_weight_remainder == i)[0])

        new_order = new_order.astype(int)
        # print('-----------new_order-----------')
        # print(new_order)
        # print(new_order)
        weight_topology_map  = trained_weight[new_order,:][:].reshape(n, m, -1).astype(np.float32)

        # print('------default_weight_topology_map----------')
        # print(weight_topology_map)

        new_weight_topology_map = insert_units(slice_point, weight_topology_map)
        # print('------new_weight_topology_map-------')
        # print(new_weight_topology_map)


        # reformat back to generator order
        # TODO: number 5 in reshpae must be equal to input_data ttribute length
        new_weight_after_insertion = np.swapaxes(new_weight_topology_map,0,1).reshape(-1, input_dim).astype(np.float32)
        # print('---------reshape back to default weight result-----------')
        # print(new_weight_after_insertion)

        # add row to initial
        new_n = n+1

        return new_weight_after_insertion, m, new_n

    else:
        # insert one column
        print('insert column - add x dirction')

        # check if slice point equal 0 if 0 then return 1
        print('slice_point')
        if slice_point//n == 0:
            slice_point = 1
        else:
            slice_point = slice_point//n
        print(slice_point)

        print('-----------AfterReshape-----------')
        # todo : this 5 must be input_data attribute lengths
        weight_topology_map = trained_weight.reshape((m,n,-1))
        # print('------default_weight_topology_map')
        # print(weight_topology_map)
        new_weight_topology_map = insert_units(slice_point, weight_topology_map)
        # print('------new_weight_topology_map-------')
        # print(new_weight_topology_map)

        # print('---------reshape back to default weight result-----------')
        # print(new_weight_after_insertion)
        new_weight_after_insertion = new_weight_topology_map.reshape(-1, input_dim).astype(np.float32)


        new_m = m+1

        return new_weight_after_insertion, new_m, n



def check_tau2_condition (clustered_result_by_index, input_data, reault_mqe, mqe0, input_dim,m,mapname,level):
    tau2 = 0.6
    print('---------------tau2*mqe0--------------')
    print(tau2*mqe0)
    for idx, each_mqe in enumerate(reault_mqe):
        if each_mqe < tau2*mqe0:
            print('satisfy tau2 condition')
        else:
            row=idx//m
            column=idx%m
            input_index = clustered_result_by_index[idx]
            print('-------------------')

            new_input_that_not_satisfy_tau2_condition = np.take(input_data, input_index, 0)

            m = 2
            n = 2
            level += 1

            new_mqe0 = cal_mqe(new_input_that_not_satisfy_tau2_condition)

            print("new mqe0: "+ str(new_mqe0))

            check_tau1_condition(m, n, new_mqe0, new_input_that_not_satisfy_tau2_condition, input_dim,level,mapname=mapname,row=row,column=column)
            level -= 1

    # for i in range(0,6):
    #
    #     if  reault_mqe[i] < tau2*mqe0:
    #         print('satisfy tau2 condition')
    #     else:
    #         input_index = clustered_result_by_index[i]
    #
    #
    #         new_input_that_not_satisfy_tau2_condition = np.take(input_data, input_index, 0)
    #
    #         m = 2
    #         n = 2
    #
    #         check_tau1_condition(m, n, mqe0, new_input_that_not_satisfy_tau2_condition)
    #




def check_tau1_condition (m, n,  mqe0, input_data, dim,level, row=0,column=0, mapname="Average_Map0"):

    tau1 = 0.4

    tau1_iter_time = 1

    # if not satisfy then call som
    trained_weight, som_result_map = call_som(m, n, dim, input_data)
    # find inputs in each unit
    clustered_result_by_index = clustered_location_input_index(m, n, trained_weight, som_result_map, input_data)
    # print('input that belong to same cluster:')
    # print(clustered_result_by_index)
    mqe = cal_clustered_mqe(input_data, clustered_result_by_index)
    print('each unit mqe:')
    print(mqe)
    n_mqe = list(filter(lambda x: x!=0.0,mqe))


    while True:
        if len(n_mqe)<=1:
            break
        if np.mean(mqe) < tau1*mqe0:

            print('mqe < tau1*mqe0')
            print('-------------------------------------after_tau1_iter_time-------------------------------------')
            print(tau1_iter_time)
            print('-------------------------------------mqe after insertation-------------------------------------')
            print(mqe)
            print('-------------------------------------input belong to unit  after insertation-------------------------------------')
            print('input that belong to same cluster:')


            result_by_ID = [[df_nominal[i] for i in j] for j in clustered_result_by_index]
            # result_by_ID = [[df_nominal.ix[i] for i in j] for j in clustered_result_by_index]


            #print(result_by_ID)
            results=np.array(result_by_ID)
            if level!=0:
                mapname = mapname  + '__unit' + '(' + str(row) + 'x' + str(column) + ')' + 'level' + str(level)+"shape="+str(m)+"x"+str(n)

            workbook = xlsxwriter.Workbook(mapname+".xlsx")
            worksheet = workbook.add_worksheet("shape="+str(m)+"x"+str(n))

            for col, data in enumerate(results):
                worksheet.write_column(0,col , data)
            workbook.close()


            print('-------------------------------------topology_map  after insertation-------------------------------------')
            print('m: ')
            print(m)
            print('n:')
            print(n)
            check_tau2_condition(clustered_result_by_index, input_data, mqe, mqe0, dim,m,mapname,level)

            break

        else:
            print('mqe > tau1*mqe0')

            topology_map = np.array(list(som_neuron_locations(m, n)))
            # find max mqe index
            error_unit_index = np.argmax(mqe)
            loaciton_map = np.array(list(som_neuron_locations(m, n)))
            error_unit_location = np.take(loaciton_map, error_unit_index, 0)
            # print('error_unit_location:')
            # print(error_unit_location)

            neighborhood_location_index = find_neighborhood_location(topology_map, m, n, error_unit_location)
            print('dissimilar_weight_location:')
            dissimilar_weight_location, dissimilar_weight_location_index = get_dissimilar_weight_location(topology_map, error_unit_index, neighborhood_location_index, trained_weight)
            print(dissimilar_weight_location)
            # insert Unit
            new_weight_after_insertion, m, n = get_map_weight_after_unit_insertion(m, n, topology_map, error_unit_location, error_unit_index, dissimilar_weight_location, dissimilar_weight_location_index, trained_weight, dim)
            print('re-call SOM:')
            print(m)
            print(n)
            # print(new_weight_after_insertion)

            # 2,3,4.... SOM call
            trained_weight, som_result_map = call_som(m, n, dim, input_data, new_weight_after_insertion)

            # find how many input in each unit
            clustered_result_by_index = clustered_location_input_index(m, n, trained_weight, som_result_map, input_data)
            mqe = cal_clustered_mqe(input_data, clustered_result_by_index)
            tau1_iter_time += 1



#
# Below is GHSOM Train step
#
# initial variable
import time

t0=time.time()
mqe0 = cal_clustered_mqe(input_data)
print(mqe0)
# m => x direction, n => y direction
m = 2
n = 2

check_tau1_condition(m, n, mqe0, input_data, input_dim,level=0)
t=time.time()
print (str(t-t0))