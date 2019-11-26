import solver
import output_validator

file_name_suffix = '50.in'
output_suffix = '50.out'
input_folder = 'inputs/'
output_folder = 'outputs/'
file_name_prefixes = list(range(100, 366))

def solve():
    for fn in file_name_prefixes:
        file_name = input_folder + str(fn) + '_' + file_name_suffix
        print("Solving:",file_name, flush = True)
        try:
            solver.solve_from_file(file_name, output_folder, params=['greedy_clustering_three_opt',2])
        except:
            print(file_name, "doesn't exist")

def validate():
    for fn in file_name_prefixes:
        file_name = input_folder + str(fn) + '_' + file_name_suffix
        output = output_folder + str(fn) + '_' + output_suffix
        output_validator.validate_output(file_name,output,params=['greedy_clustering_three_opt',2])

solve()
#validate()