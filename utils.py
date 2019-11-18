import sys, os

def get_files_with_extension(directory, extension):
    files = []
    for name in os.listdir(directory):
        if name.endswith(extension):
            files.append(f'{directory}/{name}')
    return files


def read_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = [line.replace("Â", " ").strip().split() for line in data]
    return data


def write_to_file(file, string, append=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(file, mode) as f:
        f.write(string)


def write_data_to_file(file, data, separator, append=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(file, mode) as f:
        for item in data:
            f.write(f'{item}{separator}')


def input_to_output(input_file):
    return input_file.replace('input', 'output').replace('.in', '.out')

def clear_file(file):
    open(file,'w').close()

def append_next_line(file):
    write_to_file(file, '\n', append = 'a')

def append_data_next_line(file, data, separator, append = False):
    write_data_to_file(file, data, separator, append)
    write_to_file(file, '\n', append = 'a')

def clear_logs():
    print('Clearing Log Files')
    clear_file('logs/naive.log')
    clear_file('logs/greedy.log')