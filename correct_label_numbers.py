import os
from natsort import natsorted






def replace_label_number(input_dir, output_dir):

    # open the input file for reading
    with open(input_dir, 'r') as input_file:
        # read the contents of the file into a list of lines
        lines = input_file.readlines()

    # create an empty list to hold the modified lines
    modified_lines = []

    # iterate over each line in the list of lines
    for line in lines:
        # split the line into a list of words
        words = line.split()

        # replace the first word in the list with a new value
        if words[0] == '3':
            words[0] = '0'

        elif words[0] == '0':
            words[0] = '2'

        elif words[0] == '1':
            words[0] = '3'

        #words[0] = 'NEW_VALUE'

        # join the list of words back into a single line
        modified_line = ' '.join(words)

        # add the modified line to the list of modified lines
        modified_lines.append(modified_line)

    # open the output file for writing
    with open(output_dir+'.txt', 'w') as output_file:
        # write each modified line to the output file
        for modified_line in modified_lines:
            output_file.write(modified_line + '\n')






input_dir = '../data/raw/Highway_Roboflow/train/labels'

files = natsorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

new_label_dir = '../data/raw/Highway/labels'

if not os.path.exists(new_label_dir):
    os.makedirs(new_label_dir)



for file in files:
    input_dir_tmp = os.path.join(input_dir,file)
    output_dir_tmp = os.path.join(new_label_dir, file[4:10])
    replace_label_number(input_dir_tmp,output_dir_tmp)
A = 1
