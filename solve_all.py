import solver
import output_validator
import sys
import cython
import solver_cython


file_name_suffix = '100.in'
output_suffix = '100.out'
input_folder = 'inputs/'
output_folder = 'outputs/'
file_name_prefixes = list(range(int(sys.argv[1]), int(sys.argv[2])))

def solve():
    for fn in file_name_prefixes:
        file_name = input_folder + str(fn) + '_' + file_name_suffix
        #print("Solving:",file_name, flush = True)
        try:
            solver_cython.improve_from_file(file_name, output_folder, params=['greedy_clustering_three_opt', 2])
        except:
            print(file_name, "doesn't exist")

def validate():
    for fn in file_name_prefixes:
        file_name = input_folder + str(fn) + '_' + file_name_suffix
        output = output_folder + str(fn) + '_' + output_suffix
        output_validator.validate_output(file_name,output,params=['greedy_clustering_three_opt',1])

solve()
#print(file_name_prefixes)
#solve()
#validate()

# solver_cython.improve_from_file('inputs/1_200.in', 'outputs/', params=['two_opt'])
# solver_cython.improve_from_file('inputs/1_200.in', 'outputs/', params=['three_opt'])
# solver_cython.improve_from_file('inputs/1_200.in', 'outputs/', params=['three_opt'])
#solver.solve_from_file('inputs/50.in', 'outputs/', params=['two_opt'])