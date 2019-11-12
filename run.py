import input_generator

def create_inputs():
    input_generator.generate_input('inputs/50.in', 50, 25, 500)
    input_generator.generate_input('inputs/100.in', 100, 50, 500)
    input_generator.generate_input('inputs/200.in', 200, 100, 500)

create_inputs()
