# below is GHSOM Train step
# initial variable
mqe0 = cal_clustered_mqe(input_data)
print(mqe0)
# m => x direction, n => y direction
m = 2
n = 2
tau1 = 0.1
iter_time = 1
mqe = []

while True:
    if np.mean(mqe) < tau1*mqe0:
        print('mqe < mqe0')
        print('-------------------------------------after_iter_time-------------------------------------')
        print(iter_time)
        print('-------------------------------------mqe after insertation-------------------------------------')
        print(mqe)
        print('-------------------------------------input belong to unit  after insertation-------------------------------------')
        print('input that belong to same cluster:')
        print(clustered_result_by_index)
        print('-------------------------------------topology_map  after insertation-------------------------------------')
        print('m: ')
        print(m)
        print('n:')
        print(n)
        break
    else:
        print('mqe > mqe0')

        topology_map = np.array(list(som_neuron_locations(m, n)))

        if (iter_time == 1 ):
            print('------------------------------------- first 2*2 SOM-------------------------------------')
            # first 2*2 SOM
            trained_weight,  som_result_map = call_som(m, n, 5, input_data)

            # find how many input in each unit
            clustered_result_by_index = clustered_location_input_index(m, n, trained_weight, som_result_map, input_data)
            print('input that belong to same cluster:')
            print(clustered_result_by_index)
            mqe = cal_clustered_mqe(input_data, clustered_result_by_index)
            print('each unit mqe:')
            print(mqe)


        # find max mqe index
        error_unit_index = np.argmax(mqe)
        loaciton_map = np.array(list(som_neuron_locations(m, n)))
        error_unit_location = np.take(loaciton_map, error_unit_index, 0)
        print('error_unit_location:')
        print(error_unit_location)

        neighborhood_location_index = find_neighborhood_location(topology_map, m, n, error_unit_location)
        print('dissimilar_weight_location:')
        dissimilar_weight_location, dissimilar_weight_location_index = get_dissimilar_weight_location(topology_map, error_unit_index, neighborhood_location_index, trained_weight)
        print(dissimilar_weight_location)
        # insert Unit
        new_weight_after_insertion, m, n = get_map_weight_after_unit_insertion(m, n, topology_map, error_unit_index, dissimilar_weight_location_index, trained_weight)
        print('second time SOM:')
        print(new_weight_after_insertion)

        # 2,3,4.... SOM call
        trained_weight, som_result_map = call_som(m, n, 5, input_data, new_weight_after_insertion)

        # find how many input in each unit
        clustered_result_by_index = clustered_location_input_index(m, n, trained_weight, som_result_map, input_data)
        mqe = cal_clustered_mqe(input_data, clustered_result_by_index)
        iter_time += 1
        # print('iter_time')
        # print(iter_time)
        # print('-------------------------------------mqe after insertation-------------------------------------')
        # print(mqe)
        # print('-------------------------------------input belong to unit  after insertation-------------------------------------')
        # print('input that belong to same cluster:')
        # print(clustered_result_by_index)
        # print('-------------------------------------topology_map  after insertation-------------------------------------')
        # print('m: ')
        # print(m)
        # print('n:')
        # print(n)
