import input_generator
import input_validator
import output_validator
import solver
def create_inputs(range):
    input_generator.generate_input('inputs/50.in', 50, 25, range)
    input_generator.generate_input('inputs/100.in', 100, 50, range)
    input_generator.generate_input('inputs/200.in', 200, 100, range)

#create_inputs()
def run(range):
    create_inputs(range)
    solver.solve_all('inputs/', 'outputs/', params = ['bad'])
    output_validator.validate_all_outputs('inputs/','outputs/')

run(1200000000)
# input_generator.generate_input('inputs/200.in', 200, 100, 500)
# solver.solve_all('inputs/', 'outputs/', params = ['bad'])
# output_validator.validate_all_outputs('inputs/','outputs/')