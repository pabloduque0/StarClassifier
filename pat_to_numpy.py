import numpy as np

class PatToNumpy():

    def perform_parsing(self, file_path):
        input_file = open(file_path, 'r')
        lines = input_file.readlines()

        data_lines = self.get_meta_and_truncate(lines)

        data = []
        labels = []
        for index, line in enumerate(data_lines):
            line = line.strip('\n').strip(' ')
            if index % 2 == 0:
                values = line.split('  ')
                data.append(list(values))
            else:
                values = line.split(' ')
                labels.append(list(values))

        return data, labels if len(labels) == len(data) else None

    def get_meta_and_truncate(self, lines):

        num_patters = None
        num_inputs = None
        num_outputs = None

        data_start_index = 0
        for index, line in enumerate(lines):
            if 'No. of patterns' in line:
                num_patters = int(line.split(':')[1].strip(' '))
                print(num_patters)

            elif 'No. of input units' in line:
                num_inputs = int(line.split(':')[1].strip(' '))

            elif 'No. of output units' in line:
                num_outputs = int(line.split(':')[1].strip(' '))

            if num_patters is not None and num_inputs is not None and num_outputs is not None:
                data_start_index = index
                break

        data_lines = lines[data_start_index+1:]
        return data_lines