import io
import argparse
from collections import defaultdict

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_filename")
argParser.add_argument("-o", "--output_filename")
args = argParser.parse_args()

input_file = io.open(args.input_filename, encoding='utf8', mode='r')
output_file = io.open(args.output_filename, encoding='utf8', mode='w')
projected_dep_column = 6
token_id_column = 0

current_effective_length = 0
for input_line in input_file:
  if len(input_line.strip()) == 0:
    output_file.write(input_line)
    continue
  fields = input_line.split()
  fields[projected_dep_column] = str(int( fields[token_id_column] )-1)
  output_file.write('\t'.join(fields)+'\n')

input_file.close()
output_file.close()

