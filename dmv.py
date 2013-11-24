import io
import argparse
from collections import defaultdict, namedtuple

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_filename")
argParser.add_argument("-o", "--output_filename")
args = argParser.parse_args()

# given a conll file, return a list of 'sentences'. 
# each sentence -in fact- is a string of space-separated pos tags.
def read_conll_sents(filename):
  file = io.open(filename, encoding='utf8', mode='r')
  projected_dep_column = 6
  pos_tag_column = 3
  token_id_column = 0
  sents = []
  current_sent = []
  for input_line in file:
    if len(input_line.strip()) == 0:
      if len(current_sent) > 0: sents.append('{}\n'.format(' '.join(current_sent))) 
      current_sent = []
      continue
    current_sent.append(input_line.split()[pos_tag_column])
  file.close()
  return sents

# type could be 'sealed', 'half-sealed' or 'not-sealed'
SEALED, HALF_SEALED, NOT_SEALED, ROOT = 'sealed', 'half_sealed', 'not_sealed', 'ROOT'
Nonterminal = namedtuple('Nonterminal', 'type, pos')

# dir could be 'right' or 'left'
# lhs must be a nonterminal
# rhs must be a list of {terminals, nonterminals)
Rule = namedtuple('Rule', 'lhs, rhs')

# terminals, nonterminals, rules, and their indexes 
sealed_nonterminals, half_sealed_nonterminals, not_sealed_nonterminals, all_nonterminals, terminals = set(), set(), set(), set(), set()
sealing_rules, half_sealing_rules, terminal_rules, binary_rules = set(), set(), set(), set()
all_rules = {}
reverse_rules = defaultdict(set)

# add the specified rule to 1) all_rules, 2) reverse_rules, 3) sealing_rules, half_sealing_rules, terminal_rules, or binary_rules
def add_rule(lhs, rhs, init_logprob):
  rule=Rule(lhs=lhs, rhs=rhs)
  if rule in all_rules: return
  # 1
  all_rules[rule] = init_logprob
  # 2
  reverse_rules[rhs].add(rule)
  assert len(reverse_rules[rhs]) == 1
  # 3
  if lhs.type == SEALED and len(rhs) == 1 and rhs[0].type == HALF_SEALED:
    sealing_rules.add(rule)
  elif lhs.type == HALF_SEALED and len(rhs) == 1 and rhs[0].type == NOT_SEALED:
    half_sealing_rules.add(rule)
  elif lhs.type == NOT_SEALED and len(rhs) == 1:
    terminal_rules.add(rule)
  elif lhs.type == HALF_SEALED and len(rhs) == 2 and rhs[0].type == SEALED and rhs[1].type == HALF_SEALED \
        or lhs.type == NOT_SEALED and len(rhs) == 2 and rhs[0].type == NOT_SEALED and rhs[1].type == SEALED:
    binary_rules.add(rule)
  else:
    assert False

# given the sentences, determine the terminals, nonterminals and rules. also, index them for efficient retrieval.
def create_dmv_params(sents):
  global sealed_nonterminals, half_sealed_nonterminals, not_sealed_nonterminals, all_nonterminals, terminals, sealing_rules, half_sealing_rules, terminal_rules, binary_rules, all_rules
  init_rule_logprob = 0
  for sent in sents:
    tokens = sent.split()
    tokens.append(ROOT)
    for token in set(tokens) - terminals:
      # identify nonterminals
      sealed = Nonterminal(type=SEALED, pos=token)
      half_sealed = Nonterminal(type=HALF_SEALED, pos=token)
      not_sealed = Nonterminal(type=NOT_SEALED, pos=token)
      # add to terminals, nonterminals
      terminals.add(token)
      sealed_nonterminals.add(sealed)
      half_sealed_nonterminals.add(half_sealed)
      not_sealed_nonterminals.add(not_sealed)
      all_nonterminals = all_nonterminals | set([sealed, half_sealed, not_sealed])
      # identify rules
      terminal_rule = Rule(lhs=not_sealed, rhs=(token,))
      half_sealing_rule = Rule(lhs=half_sealed, rhs=(not_sealed,))
      sealing_rule = Rule(lhs=sealed, rhs=(half_sealed,))
      # add to rules
      all_rules[terminal_rule] = init_rule_logprob
      all_rules[half_sealing_rule] = init_rule_logprob
      all_rules[sealing_rule] = init_rule_logprob
      sealing_rules.add(sealing_rule)
      half_sealing_rules.add(half_sealing_rule)
      terminal_rules.add(terminal_rule)
  # identify and add binary rules
  for parent in half_sealed_nonterminals:
    for left_child in sealed_nonterminals - set([Nonterminal(type=SEALED, pos=ROOT)]):
      rule = Rule(lhs=parent, rhs=(left_child, parent))
      binary_rules.add(rule)
      all_rules[rule] = init_rule_logprob
  for parent in not_sealed_nonterminals:
    for right_child in sealed_nonterminals - set([Nonterminal(type=SEALED, pos=ROOT)]):
      rule = Rule(lhs=parent, rhs=(parent, right_child))
      binary_rules.add(rule)
      all_rules[rule] = init_rule_logprob
  # done!
  return None

# items to compute inside/outside scores for
ChartItem = namedtuple('ChartItem', 'nonterminal, from_, to')
# inside and outside scores
inside, outside = {}, {}
# index to retrieve chart items by (from_, to)
reverse_inside, reverse_outside = defaultdict(set), defaultdict(set)

def add_chart_item(nonterminal, from_, to, logprob, chart, reverse_chart):
  item = ChartItem(nonterminal=nonterminal, from_=from_, to=to)
  if item not in chart:
    chart[item] = float('-inf')
    reverse_chart[(from_, to)].add(item)
  chart[item] = logadd(chart[item], logprob)

def compute_inside_scores(sent):
  global inside, reverse_inside
  tokens = sent.split()
  tokens.append(ROOT)
  inside = {}
  # span 1 is a special case
  for i in xrange(len(tokens)):
    # not sealed
    not_sealed_nonterminal = Nonterminal(type=NOT_SEALED, pos=tokens[i])
    add_chart_item(nonterminal=not_sealed_nonterminal, from_=i, to=i+1, logprob=0, 
                   chart=inside, reverse_chart=reverse_inside)
    # half sealed
    half_sealed_nonterminal = Nonterminal(type=HALF_SEALED, pos=tokens[i])
    add_chart_item(nonterminal=half_sealed_nonterminal, from_=i, to=i+1, 
                   logprob=all_rules[Rule(lhs=half_sealed_nonterminal, rhs=(not_sealed_nonterminal,))],
                   chart=inside, reverse_chart=reverse_inside)
    # sealed
    sealed_nonterminal = Nonterminal(type=SEALED, pos=tokens[i])
    add_chart_item(nonterminal=sealed_nonterminal, from_=i, to=i+1,
                   logprob=all_rules[Rule(lhs=sealed_nonterminal, rhs=(half_sealed_nonterminal,))] + \
                     inside[ChartItem(nonterminal=half_sealed_nonterminal, from_=i, to=i+1)],
                   chart=inside, reverse_chart=reverse_inside)
  print 'unit spans:\n\n'
  for item in inside.keys():
    print item, '-->', inside[item]
  # spans > 1 are similar, start from span = 2
  for span in range(2, len(tokens)+1):
    # determine which cell to add chart items to
    for from_ in range(0, len(tokens)):
      to = from_ + span
      if to > len(tokens): continue
      # determine a split point
      print 'at cell (', from_, ', ', to, ')'
      for mid in range(from_ + 1, from_ + span):
        # potential left children
        for left_child_item in reverse_inside[(from_, mid)]:
          # potential right children
          for right_child_item in reverse_outside[(mid, to)]:
            # now, find out if this is the rhs of any rule
            rhs = (left_child_item, right_child_item)
            if rhs not in reverse_rules: continue
            # sweet! lets visit each applicable rules
            for rule in reverse_rules[rhs]:
              print ' using ', rule, ' to combine ', left_child_item, ' and ', right_child_item
              add_chart_item(nonterminal=rule.lhs, from_=from_, to=to,
                             logprob=all_rules[rule] + inside[left_child_item] + inside[right_child_item],
                             chart=inside, reverse_chart=reverse_inside)
              

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_filename")
argParser.add_argument("-o", "--output_filename")
args = argParser.parse_args()

sents = read_conll_sents(args.input_filename)

def print_first_n(container, n):
  for item in container:
    print item
    n-=1
    if n <= 0: break

create_dmv_params(sents)

print '\nexample sealed nonterminals (total = {}):\n'.format(len(sealed_nonterminals))
print_first_n(sealed_nonterminals, 10)
print '\nexample half sealed nonterminals (total = {}):\n'.format(len(half_sealed_nonterminals))
print_first_n(half_sealed_nonterminals, 10)
print '\nexample not sealed nonterminals (total = {}):\n'.format(len(not_sealed_nonterminals))
print_first_n(not_sealed_nonterminals, 10)
print '\nexample all nonterminals (total = {}):\n'.format(len(all_nonterminals))
print_first_n(all_nonterminals, 10)
print '\nexample terminals (total = {}):\n'.format(len(terminals))
print_first_n(terminals, 10)
print '\nsealing_rules (total = {}):\n'.format(len(sealing_rules))
print_first_n(sealing_rules, 10)
print '\nhalf_sealing_rules (total = {}): \n'.format(len(half_sealing_rules))
print_first_n(half_sealing_rules, 10)
print '\nterminal_rules (total = {}): \n'.format(len(terminal_rules))
print_first_n(terminal_rules, 10)
print '\nbinary_rules (total = {}): \n'.format(len(binary_rules))
print_first_n(binary_rules, 10)
print '\nall_rules (total = {}): \n'.format(len(all_rules))
print_first_n(all_rules, 10)

compute_inside_scores(sents[0])

