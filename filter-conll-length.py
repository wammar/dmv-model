
import io
import argparse
from collections import defaultdict

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_filename")
argParser.add_argument("-o", "--output_filename")
argParser.add_argument("-l", "--max_length")
args = argParser.parse_args()

input_file = io.open(args.input_filename, encoding='utf8', mode='r')
output_file = io.open(args.output_filename, encoding='utf8', mode='w')
all_counter, short_counter = 0, 0
pos_tag_column = 7

sentence_lines = []
current_effective_length = 0
for input_line in input_file:
  if len(input_line.strip()) == 0 and len(sentence_lines) > 0:
    #if len(sentence_lines) <= int(args.max_length):
    if current_effective_length <= int(args.max_length):
      for output_line in sentence_lines: output_file.write(output_line)
      output_file.write(input_line)
      short_counter += 1
    all_counter += 1
    current_effective_length = 0
    sentence_lines = []
    continue
  sentence_lines.append(input_line)
  if input_line.split()[pos_tag_column] != "P":
    current_effective_length += 1

input_file.close()
output_file.close()

print all_counter, ' sentences were read. ', short_counter, ' sentences with length <= ', args.max_length, ' were written.'
