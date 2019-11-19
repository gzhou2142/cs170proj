#   py solver.py inputs/50.in outputs/ three_opt
#   py output_validator.py inputs/50.in outputs/50.out three_opt
#   py solver.py inputs/50.in outputs/ greedy
#   py output_validator.py inputs/50.in outputs/50.out greedy
#  py solver.py inputs/100.in outputs/ greedy
#  py output_validator.py inputs/100.in outputs/100.out greedy

 #py solver.py --all inputs/ outputs/ naive
 #py output_validator.py --all inputs/ outputs/ naive

#  py solver.py --all inputs/ outputs/ greedy
#  py output_validator.py --all inputs/ outputs/ greedy
 py solver.py --all inputs/ outputs/ three_opt
 py output_validator.py --all inputs/ outputs/ three_opt