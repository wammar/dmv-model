#!/usr/bin/env python

import io
import argparse
from collections import defaultdict, namedtuple
from numpy import logaddexp
import math
from math import log, exp
import sys

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_filename", required=True)
argParser.add_argument("-o", "--output_filename", required=True)
args = argParser.parse_args()

NEGINF = -300000.0

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
Nonterminal = namedtuple('Nonterminal', 'type, pos, fertility')

# dir could be 'right' or 'left'
# lhs must be a nonterminal
# rhs must be a list of {terminals, nonterminals)
Rule = namedtuple('Rule', 'lhs, rhs')

# terminals, nonterminals, rules, and their indexes 
sealed_nonterminals, half_sealed_nonterminals, not_sealed_nonterminals, all_nonterminals, terminals = set(), set(), set(), set(), set()
all_rules, sealing_rules, half_sealing_rules, terminal_rules, binary_rules = set(), set(), set(), set(), set()
reverse_rules = defaultdict(set)
# model parameters: STOP or NO_STOP given terminal string, direction, and adjacency
# for example stop_params[('NN', 'RIGHT', 'ADJ')][STOP] = log of some probability
# for example stop_params[('NN', 'RIGHT', 'ADJ')][STOP] = log of (1 - the other probability)
STOP, NO_STOP, RIGHT, LEFT, ADJ, NOT_ADJ = 'STOP', 'NO_STOP', 'RIGHT', 'LEFT', 'ADJ', 'NOT_ADJ'
stop_params = {} 
# model parameters: terminal string given terminal string, direction
# for example prod_params[('NN', 'RIGHT')]['VP'] = log of some prob
prod_params = {}

# expected counts of each event
stop_counts, prod_counts = {}, {}

# zero expected counts
def zero_expected_counts():
  global stop_counts, prod_counts
  for context in stop_params.keys():
    stop_counts[context] = {}
    for decision in stop_params[context].keys():
      stop_counts[context][decision] = 0.0
  for context in prod_params.keys():
    prod_counts[context] = {}
    for decision in prod_params[context].keys():
      prod_counts[context][decision] = 0.0

def normalize_expected_counts():
  global stop_counts, prod_counts, stop_params, prod_params
  # first, update stop_params
  for context in stop_counts.keys():
    # compute marginals of this context
    marginal_stop_counts = 0.0
    for decision in stop_counts[context].keys():
      assert stop_counts[context][decision] >= 0.0
      marginal_stop_counts += stop_counts[context][decision]
    # update stop_params
    for decision in stop_counts[context].keys():
      if marginal_stop_counts <= 0.0:
        continue
      if stop_counts[context][decision] == 0.0:
        stop_params[context][decision] = NEGINF
      else:
        stop_params[context][decision] = log(1.0 * stop_counts[context][decision] / marginal_stop_counts)
  # now, update prod_params    
  for context in prod_counts.keys():
    # compute marginals 
    marginal_prod_counts = 0.0
    for decision in prod_counts[context].keys():
      assert prod_counts[context][decision] >= 0.0
      marginal_prod_counts += prod_counts[context][decision]
    # update prod_params
    for decision in prod_counts[context].keys():
      if marginal_prod_counts <= 0.0:
        continue
      if prod_counts[context][decision] == 0.0:
        prod_params[context][decision] = NEGINF
      else:
        prod_params[context][decision] = log(1.0 * prod_counts[context][decision] / marginal_prod_counts)

# add expected counts (after having built the inside and outside charts, of course)
def add_expected_counts(inside_score):
  global reverse_rules, paths_to, outside, inside
  
  # consider each parent item in the inside chart
  for parent_item in paths_to.keys():
    # ignore items which don't exist in the outside chart
    if parent_item not in outside: continue
    for children_items in paths_to[parent_item]:
      if len(children_items) == 1:
        # determine the probability of using this path: 
        # exp(outside[parent] + inside[child] + stop_params[parent_terminal, dir, adj][stop])
        parent_terminal = parent_item.nonterminal.pos
        direction = RIGHT if parent_item.nonterminal.type == HALF_SEALED else LEFT
        adjacency = ADJ if children_items[0].nonterminal.fertility == 0 else NOT_ADJ
        logprob = outside[parent_item] + inside[children_items[0]] + \
            stop_params[(parent_terminal, direction, adjacency)][STOP] \
            - inside_score
        # update the counts
        stop_counts[(parent_terminal, direction, adjacency)][STOP] += exp(logprob)
      elif len(children_items) == 2:
        # determine the probability of using this path:
        # exp(outside[parent] + inside[child1] + inside[child2] 
        #     + stop_params[parent_terminal, dir, adj][no_stop] + prod_params[head_terminal, dir][dependent_terminal]
        direction = RIGHT if children_items[1].nonterminal.type == SEALED else LEFT
        adjacency = ADJ if direction == RIGHT and children_items[0].nonterminal.fertility == 0 or \
            direction == LEFT and children_items[1].nonterminal.fertility == 0 \
            else NOT_ADJ
        head_terminal = parent_item.nonterminal.pos
        dependent_terminal = children_items[0].nonterminal.pos if direction == LEFT else children_items[1].nonterminal.pos
        logprob = outside[parent_item] + inside[children_items[0]] + inside[children_items[1]] \
            + stop_params[(head_terminal, direction, adjacency)][NO_STOP] \
            + prod_params[(head_terminal, direction)][dependent_terminal] \
            - inside_score
        # update the counts
        stop_counts[(head_terminal, direction, adjacency)][NO_STOP] += exp(logprob)
        prod_counts[(head_terminal, direction)][dependent_terminal] += exp(logprob)
      else:
        assert False
    
# add the specified rule to 1) all_rules, 2) reverse_rules, 3) sealing_rules, half_sealing_rules, terminal_rules, or binary_rules
def add_rule(lhs, rhs):
  rule=Rule(lhs=lhs, rhs=rhs)
  if rule in all_rules: return
  # 1
  all_rules.add(rule)
  # 2
  reverse_rules[rhs].add(rule)
  # 3
  if lhs.type == SEALED and len(rhs) == 1 and rhs[0].type == HALF_SEALED:
    sealing_rules.add(rule)
  elif lhs.type == HALF_SEALED and len(rhs) == 1 and rhs[0].type == NOT_SEALED:
    half_sealing_rules.add(rule)
  elif lhs.type == HALF_SEALED and len(rhs) == 1 and rhs[0].type == HALF_SEALED and lhs.fertility == rhs[0].fertility + 1:
    pass
  elif lhs.type == NOT_SEALED and len(rhs) == 1 and type(rhs[0]) != Nonterminal:
    terminal_rules.add(rule)
  elif lhs.type == NOT_SEALED and len(rhs) == 1 and rhs[0].type == NOT_SEALED and lhs.fertility == rhs[0].fertility + 1:
    pass
  elif lhs.type == HALF_SEALED and len(rhs) == 2 and rhs[0].type == SEALED and rhs[1].type == HALF_SEALED \
        or lhs.type == NOT_SEALED and len(rhs) == 2 and rhs[0].type == NOT_SEALED and rhs[1].type == SEALED:
    binary_rules.add(rule)
  else:
    assert False

# given the sentences, determine the terminals, nonterminals and rules. also, index them for efficient retrieval.
def create_dmv_params(sents):
  global sealed_nonterminals, half_sealed_nonterminals, not_sealed_nonterminals, all_nonterminals, terminals
  global sealing_rules, half_sealing_rules, terminal_rules, binary_rules, all_rules
  for sent in sents:
    tokens = sent.split()
    tokens.append(ROOT)
    for token in set(tokens) - terminals:
      # identify nonterminals
      sealed = Nonterminal(type=SEALED, pos=token, fertility=0)
      half_sealed_infertile = Nonterminal(type=HALF_SEALED, pos=token, fertility=0)
      half_sealed_fertile = Nonterminal(type=HALF_SEALED, pos=token, fertility=1)
      not_sealed_infertile = Nonterminal(type=NOT_SEALED, pos=token, fertility=0)
      not_sealed_fertile = Nonterminal(type=NOT_SEALED, pos=token, fertility=1)
      # add to terminals, nonterminals
      terminals.add(token)
      sealed_nonterminals.add(sealed)
      half_sealed_nonterminals |= set([half_sealed_fertile, half_sealed_infertile])
      not_sealed_nonterminals |= set([not_sealed_fertile, not_sealed_infertile])
      all_nonterminals |= set([sealed, half_sealed_fertile, half_sealed_infertile, not_sealed_fertile, not_sealed_infertile])
      # create sealing rules
      add_rule(lhs=not_sealed_infertile, rhs=(token,)) # at no cost
      add_rule(lhs=half_sealed_infertile, rhs=(not_sealed_infertile,)) #incurs right stop cost given no children
      add_rule(lhs=half_sealed_infertile, rhs=(not_sealed_fertile,)) #incurs right stop cost given children 
      add_rule(lhs=sealed, rhs=(half_sealed_fertile,)) # incurs left stop cost given no children
      add_rule(lhs=sealed, rhs=(half_sealed_infertile,)) # incurs left stop cost given children
#RMME
#      # create fertility rules
#      add_rule(lhs=not_sealed_fertile, rhs=(not_sealed_infertile,)) # at no cost
#      add_rule(lhs=half_sealed_fertile, rhs=(half_sealed_infertile,)) # at no cost
  # identify and add binary rules
  for parent in half_sealed_nonterminals:
    if parent.fertility == 0: continue # infertile nonterminals cannot be parents of binary rules
    for left_child in sealed_nonterminals - set([Nonterminal(type=SEALED, pos=ROOT, fertility=0)]):
      add_rule(lhs=parent, rhs=(left_child, parent))
      add_rule(lhs=parent, rhs=(left_child, Nonterminal(type=parent.type, pos=parent.pos, fertility=0)))
  for parent in not_sealed_nonterminals:
    if parent.fertility == 0: continue # infertile nonterminals cannot be parents of binary rules
    for right_child in sealed_nonterminals - set([Nonterminal(type=SEALED, pos=ROOT, fertility=0)]):
      add_rule(lhs=parent, rhs=(parent, right_child))
      add_rule(lhs=parent, rhs=(Nonterminal(type=parent.type, pos=parent.pos, fertility=0), right_child))
  # done identifying rules
  # now, create the actual parameters
  # for each terminal, there are four stop_params distributions
  for terminal in terminals:
    for direction in [RIGHT, LEFT]:
      for adjacency in [ADJ, NOT_ADJ]:
        stop_params[(terminal, direction, adjacency)] = {}
        stop_params[(terminal, direction, adjacency)][STOP] = math.log(0.5) 
        stop_params[(terminal, direction, adjacency)][NO_STOP] = math.log(0.5)
  # override stop_params for ROOT
  stop_params[(ROOT, RIGHT, ADJ)][STOP] = stop_params[(ROOT, RIGHT, NOT_ADJ)][STOP] = math.log(1.0)
  stop_params[(ROOT, RIGHT, ADJ)][NO_STOP] = stop_params[(ROOT, RIGHT, NOT_ADJ)][NO_STOP] = NEGINF#math.log(0.0)
  stop_params[(ROOT, LEFT, ADJ)][STOP] = stop_params[(ROOT, LEFT, NOT_ADJ)][NO_STOP] = NEGINF#math.log(0.0)
  stop_params[(ROOT, LEFT, ADJ)][NO_STOP] = stop_params[(ROOT, LEFT, NOT_ADJ)][STOP] = math.log(1.0)
  # for each terminal, there are two prod_params distributions
  for parent in terminals:
    for direction in [RIGHT, LEFT]:
      prod_params[(parent, direction)] = {}
      children_count = float(len(terminals) - 1) # ROOT is not a possible child
      for child in terminals:
        prod_params[(parent, direction)][child] = log(1/children_count)
      # override prod_params when ROOT is the child
      prod_params[(parent, direction)][ROOT] = NEGINF#log(0.0)
  return None

# items to compute inside/outside scores for
ChartItem = namedtuple('ChartItem', 'nonterminal, from_, to')
# inside and outside scores
inside, outside = {}, {}
# index to retrieve chart items by (from_, to)
reverse_inside = defaultdict(list)
# index to retrieve all the ways an inside chart item was created
# for example paths_to[X] = [ (A,B), (C,D), (E,) ] means
# that ChartItem X is reachable by combining ChartItems A and B, or ChartItems C and D, or with a unary rule from ChartItem E
paths_to = {}

def add_outside_chart_items(parent_inside_item, children_inside_items):
  global inside, outside

  # first, make sure the parent item has been added to the outside chart.
  # if it were not added, this suggests one of two things: 
  # 1) there's a bug. 
  # 2) even though this item was created by the inside algorithm, it didn't participate in any complete parses
  #    which is why the outside chart does not contain it. to save space and time, we will not create such items
  if parent_inside_item not in outside:
    return

  # add the left (or only) child item to the outside chart
  if children_inside_items[0] not in outside:
    outside[children_inside_items[0]] = NEGINF

  # sealing rules require special processing
  if len(children_inside_items) == 1:
    # first, determine stop cost
    direction = RIGHT if children_inside_items[0].nonterminal.type == NOT_SEALED and \
        parent_inside_item.nonterminal.type == HALF_SEALED \
        else LEFT
    adjacency = ADJ if children_inside_items[0].nonterminal.fertility == 0 else NOT_ADJ
    stop_cost = stop_params[parent_inside_item.nonterminal.pos, direction, adjacency][STOP]
    # then, update the outside score of this child
    outside[children_inside_items[0]] = logaddexp( outside[children_inside_items[0]],
                                                   outside[parent_inside_item] + stop_cost )
  # now, binary rules
  elif len(children_inside_items) == 2:
    # add the right child item to the outside chart
    if children_inside_items[1] not in outside:
      outside[children_inside_items[1]] = NEGINF
    
    # first, determine the cost of not stopping
    direction = LEFT if children_inside_items[0].nonterminal.type == SEALED else RIGHT
    adjacency = ADJ if direction == LEFT and children_inside_items[1].nonterminal.fertility == 0 or \
        direction == RIGHT and children_inside_items[0].nonterminal.fertility == 0 \
        else NOT_ADJ
    head_terminal = parent_inside_item.nonterminal.pos
    dependent_terminal = children_inside_items[0].nonterminal.pos if direction == LEFT else children_inside_items[1].nonterminal.pos
    no_stop_cost = stop_params[(head_terminal, direction, adjacency)][NO_STOP]
    prod_cost = prod_params[(head_terminal, direction)][dependent_terminal]

    # now, update the outside score of first child
    outside[children_inside_items[0]] = logaddexp( outside[children_inside_items[0]],
                                                   outside[parent_inside_item] + \
                                                     inside[children_inside_items[1]] + \
                                                     no_stop_cost + prod_cost)
    outside[children_inside_items[1]] = logaddexp( outside[children_inside_items[1]],
                                                   outside[parent_inside_item] + \
                                                     inside[children_inside_items[0]] + \
                                                     no_stop_cost + prod_cost)
  else:
    assert False

def add_inside_chart_item(from_, to, children_items, rule):
  global inside, reverse_inside, paths_to
  # assertions
  if children_items != None: assert len(rule.rhs) == len(children_items)

  nonterminal = rule.lhs
  # create the new item
  item = ChartItem(nonterminal=nonterminal, from_=from_, to=to)
  if item not in inside:
    inside[item] = NEGINF
    paths_to[item] = []

  # this is important cuz it determines the order in which the outside algorithm works
  if item  in reverse_inside[(from_, to)]: 
    reverse_inside[(from_, to)].remove(item)
  reverse_inside[(from_, to)].append(item)

  # compute the logprob of this item from those children
  if children_items == None:
    logprob = log(1.0)
  elif len(children_items) == 1:
    paths_to[item].append( (children_items[0],) )
    # this is either a sealing rule which incurs a cost of stop_params[(terminal, direction, adjacency)][STOP]
    if nonterminal.type == SEALED and nonterminal.fertility == 0 and rule.rhs[0].type == HALF_SEALED:
      direction = LEFT
      adjacency = ADJ if rule.rhs[0].fertility == 0 else NOT_ADJ
      logprob = stop_params[(nonterminal.pos, direction, adjacency)][STOP] + inside[children_items[0]]
    elif nonterminal.type == HALF_SEALED and nonterminal.fertility == 0 and rule.rhs[0].type == NOT_SEALED:
      direction = RIGHT
      adjacency = ADJ if rule.rhs[0].fertility == 0 else NOT_ADJ
      logprob = stop_params[(nonterminal.pos, direction, adjacency)][STOP] + inside[children_items[0]]
    # or a fertility rule which comes at no cost
    elif rule.lhs.type == rule.rhs[0].type and rule.lhs.fertility == rule.rhs[0].fertility + 1:
      assert False
    else:
      assert False
  elif len(children_items) == 2:
    paths_to[item].append((children_items[0], children_items[1]))
    # this is a binary rule, which incurs a cost of stop_params[(head_terminal, direction, adjacency)][NO_STOP] + prod_params[(head_terminal, direction)][child_terminal]
    direction = adjacency = None
    if nonterminal.type == HALF_SEALED:
      direction = LEFT
      adjacency = ADJ if children_items[1].nonterminal.fertility == 0 else NOT_ADJ
      child_terminal = children_items[0].nonterminal.pos
    elif nonterminal.type == NOT_SEALED:
      direction = RIGHT
      adjacency = ADJ if children_items[0].nonterminal.fertility == 0 else NOT_ADJ
      child_terminal = children_items[1].nonterminal.pos
    else:
      assert False
    logprob = stop_params[(nonterminal.pos, direction, adjacency)][NO_STOP] + \
        prod_params[(nonterminal.pos, direction)][child_terminal] + \
        inside[children_items[0]] + inside[children_items[1]]
  else:
    assert False

  # add this logprob to the logprob of other ways to reach this item
  inside[item] = logaddexp(inside[item], logprob)
  return item

def add_viterbi_chart_item(from_, to, children_items, rule):
  global viterbi_items, reverse_viterbi_items, paths_to
  # assertions
  if children_items != None: assert len(rule.rhs) == len(children_items)

  nonterminal = rule.lhs
  # create the new item
  item = ChartItem(nonterminal=nonterminal, from_=from_, to=to)
  if item not in inside:
    inside[item] = NEGINF
    paths_to[item] = []

  # this is important cuz it determines the order in which the outside algorithm works
  if item  in reverse_inside[(from_, to)]: 
    reverse_inside[(from_, to)].remove(item)
  reverse_inside[(from_, to)].append(item)

  # compute the logprob of this item from those children
  if children_items == None:
    logprob = log(1.0)
  elif len(children_items) == 1:
    paths_to[item].append( (children_items[0],) )
    # this is either a sealing rule which incurs a cost of stop_params[(terminal, direction, adjacency)][STOP]
    if nonterminal.type == SEALED and nonterminal.fertility == 0 and rule.rhs[0].type == HALF_SEALED:
      direction = LEFT
      adjacency = ADJ if rule.rhs[0].fertility == 0 else NOT_ADJ
      logprob = stop_params[(nonterminal.pos, direction, adjacency)][STOP] + inside[children_items[0]]
    elif nonterminal.type == HALF_SEALED and nonterminal.fertility == 0 and rule.rhs[0].type == NOT_SEALED:
      direction = RIGHT
      adjacency = ADJ if rule.rhs[0].fertility == 0 else NOT_ADJ
      logprob = stop_params[(nonterminal.pos, direction, adjacency)][STOP] + inside[children_items[0]]
    # or a fertility rule which comes at no cost
    elif rule.lhs.type == rule.rhs[0].type and rule.lhs.fertility == rule.rhs[0].fertility + 1:
      assert False
    else:
      assert False
  elif len(children_items) == 2:
    paths_to[item].append((children_items[0], children_items[1]))
    # this is a binary rule, which incurs a cost of stop_params[(head_terminal, direction, adjacency)][NO_STOP] + prod_params[(head_terminal, direction)][child_terminal]
    direction = adjacency = None
    if nonterminal.type == HALF_SEALED:
      direction = LEFT
      adjacency = ADJ if children_items[1].nonterminal.fertility == 0 else NOT_ADJ
      child_terminal = children_items[0].nonterminal.pos
    elif nonterminal.type == NOT_SEALED:
      direction = RIGHT
      adjacency = ADJ if children_items[0].nonterminal.fertility == 0 else NOT_ADJ
      child_terminal = children_items[1].nonterminal.pos
    else:
      assert False
    logprob = stop_params[(nonterminal.pos, direction, adjacency)][NO_STOP] + \
        prod_params[(nonterminal.pos, direction)][child_terminal] + \
        inside[children_items[0]] + inside[children_items[1]]
  else:
    assert False

  # contrast this logprob to the logprob of other ways to reach this item
  if viterbi[item] < logprob:
    viterbi[item] = logprob
    viterbi_backtrack[item] = children_items

  assert(False) # double check what needs to be returned here. item? regardless of its score?
  return item

def compute_outside_scores(tokens):
  global reverse_inside, paths_to, outside

  # initialize the outside chart
  outside = {}
  outside[ ChartItem(nonterminal=Nonterminal(type=SEALED, pos=ROOT, fertility=0), 
                     from_=0, to=len(tokens)) ] = 0.0
  for span in reversed(range(1, len(tokens)+1)):
    for from_ in range(0, len(tokens)):
      to = from_ + span
      if to > len(tokens): continue
      for parent_item in reversed(reverse_inside[(from_, to)]):
        if parent_item not in outside: continue
        #print 'outside[', parent_item, '] = ', outside[parent_item]
        for children_items in paths_to[parent_item]:
          add_outside_chart_items(parent_item, children_items)

  return outside[ChartItem(nonterminal=Nonterminal(type=HALF_SEALED, pos=ROOT, fertility=0), 
                           from_=len(tokens)-1, to=len(tokens))]

def compute_viterbi_parse(tokens):
  global viterbi_items, reverse_viterbi_items, reverse_rules, all_rules

  # clear the inside and outside charts
  viterbi_items, paths_to = {}, {}
  reverse_viterbi_items = {}
  # span 1 is a special case
  for i in xrange(len(tokens)):
    
    # add this (from_, to) pair to reverse_inside
    reverse_viterbi_items[(i, i+1)] = []
    
    # not sealed
    not_sealed_nonterminal = Nonterminal(type=NOT_SEALED, pos=tokens[i], fertility=0)
    not_sealed_item = add_viterbi_chart_item(from_=i, to=i+1, 
                                            children_items=None, 
                                            rule=Rule(lhs=not_sealed_nonterminal, rhs=(tokens[i],)))
    #print '(', i, ', ', i+1, '): ', not_sealed_item, ' => logprob = ', inside[not_sealed_item]
    # half sealed
    half_sealed_nonterminal = Nonterminal(type=HALF_SEALED, pos=tokens[i], fertility=0)
    half_sealed_item = add_viterbi_chart_item(from_=i, to=i+1,
                                             children_items=[not_sealed_item],
                                             rule=Rule(lhs=half_sealed_nonterminal, rhs=(not_sealed_nonterminal,)))
    #print '(', i, ', ', i+1, '): ', half_sealed_item, ' => logprob = ', inside[half_sealed_item]
    # sealed
    sealed_nonterminal = Nonterminal(type=SEALED, pos=tokens[i], fertility=0)
    sealed_item = add_viterbi_chart_item(from_=i, to=i+1,
                                        children_items=[half_sealed_item], 
                                        rule=Rule(lhs=sealed_nonterminal, rhs=(half_sealed_nonterminal,)))
    #print '(', i, ', ', i+1, '): ', sealed_item, ' => logprob = ', inside[sealed_item]
  # spans > 1 are similar, start from span = 2
  for span in range(2, len(tokens)+1):
    # determine which cell to add chart items to
    for from_ in range(0, len(tokens)):
      to = from_ + span
      if to > len(tokens): continue
      # add this (from_, to) pair to reverse_inside
      reverse_viterbi_items[(from_, to)] = []
      # cache of the items generated in this cell so that we apply unary rules on them before moving on to other cells
      cell_items = set()
      # determine a split point
      for mid in range(from_ + 1, from_ + span):
        # potential left children
        for left_child_item in reverse_viterbi_items[(from_, mid)]:
          # potential right children
          for right_child_item in reverse_viterbi_items[(mid, to)]:
            # now, find out if this is the rhs of any rule
            rhs = (left_child_item.nonterminal, right_child_item.nonterminal)
            if rhs not in reverse_rules: continue
            # sweet! lets visit each applicable rules
            for rule in reverse_rules[rhs]:
              item = add_viterbi_chart_item(from_=from_, to=to,
                                           children_items=[left_child_item, right_child_item],
                                           rule=rule)
              cell_items.add(item)
              #print '(', from_, ', ', to, '): ', item, ' => logprob = ', inside[item]
      # now, apply sealing rules to this cell (from_, to), by processing cell_items as if it's a queue
      cell_items = list(cell_items)
      while(len(cell_items) > 0):
        child_item = cell_items[-1]
        del cell_items[-1]
        for rule in reverse_rules[(child_item.nonterminal,)]:
          parent_item=add_viterbi_chart_item(from_=from_, to=to, children_items=[child_item], rule=rule)
          cell_items.append(parent_item)
          #print '(', from_, ', ', to, '): ', parent_item, ' => logprob = ', inside[parent_item]
  
  # backtrack to find the parent of each token and return it
  parents = []
  assert(False)
  return parents

def compute_inside_scores(tokens):
  global inside, reverse_inside, reverse_rules, all_rules

  # clear the inside and outside charts
  inside, outside, paths_to = {}, {}, {}
  reverse_inside = {}
  # span 1 is a special case
  for i in xrange(len(tokens)):
    
    # add this (from_, to) pair to reverse_inside
    reverse_inside[(i, i+1)] = []
    
    # not sealed
    not_sealed_nonterminal = Nonterminal(type=NOT_SEALED, pos=tokens[i], fertility=0)
    not_sealed_item = add_inside_chart_item(from_=i, to=i+1, 
                                            children_items=None, 
                                            rule=Rule(lhs=not_sealed_nonterminal, rhs=(tokens[i],)))
    #print '(', i, ', ', i+1, '): ', not_sealed_item, ' => logprob = ', inside[not_sealed_item]
    # half sealed
    half_sealed_nonterminal = Nonterminal(type=HALF_SEALED, pos=tokens[i], fertility=0)
    half_sealed_item = add_inside_chart_item(from_=i, to=i+1,
                                             children_items=[not_sealed_item],
                                             rule=Rule(lhs=half_sealed_nonterminal, rhs=(not_sealed_nonterminal,)))
    #print '(', i, ', ', i+1, '): ', half_sealed_item, ' => logprob = ', inside[half_sealed_item]
    # sealed
    sealed_nonterminal = Nonterminal(type=SEALED, pos=tokens[i], fertility=0)
    sealed_item = add_inside_chart_item(from_=i, to=i+1,
                                        children_items=[half_sealed_item], 
                                        rule=Rule(lhs=sealed_nonterminal, rhs=(half_sealed_nonterminal,)))
    #print '(', i, ', ', i+1, '): ', sealed_item, ' => logprob = ', inside[sealed_item]
  # spans > 1 are similar, start from span = 2
  for span in range(2, len(tokens)+1):
    # determine which cell to add chart items to
    for from_ in range(0, len(tokens)):
      to = from_ + span
      if to > len(tokens): continue
      # add this (from_, to) pair to reverse_inside
      reverse_inside[(from_, to)] = []
      # cache of the items generated in this cell so that we apply unary rules on them before moving on to other cells
      cell_items = set()
      # determine a split point
      for mid in range(from_ + 1, from_ + span):
        # potential left children
        for left_child_item in reverse_inside[(from_, mid)]:
          # potential right children
          for right_child_item in reverse_inside[(mid, to)]:
            # now, find out if this is the rhs of any rule
            rhs = (left_child_item.nonterminal, right_child_item.nonterminal)
            if rhs not in reverse_rules: continue
            # sweet! lets visit each applicable rules
            for rule in reverse_rules[rhs]:
              item = add_inside_chart_item(from_=from_, to=to,
                                           children_items=[left_child_item, right_child_item],
                                           rule=rule)
              cell_items.add(item)
              #print '(', from_, ', ', to, '): ', item, ' => logprob = ', inside[item]
      # now, apply sealing rules to this cell (from_, to), by processing cell_items as if it's a queue
      cell_items = list(cell_items)
      while(len(cell_items) > 0):
        child_item = cell_items[-1]
        del cell_items[-1]
        for rule in reverse_rules[(child_item.nonterminal,)]:
          parent_item=add_inside_chart_item(from_=from_, to=to, children_items=[child_item], rule=rule)
          cell_items.append(parent_item)
          #print '(', from_, ', ', to, '): ', parent_item, ' => logprob = ', inside[parent_item]
  return inside[ChartItem(nonterminal=Nonterminal(type=SEALED, pos=ROOT, fertility=0), from_=0, to=len(tokens))]

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_filename")
argParser.add_argument("-o", "--output_filename")
args = argParser.parse_args()

sents = read_conll_sents(args.input_filename)

# create params
create_dmv_params(sents)

# better initialization
pass

# em iterations
prev_iteration_logprob = 0
iterations_count = 0
while True:
  iterations_count += 1
  # E step:
  sents_counter = 0
  zero_expected_counts()
  iteration_logprob = 0.0
  for sent in sents:
    tokens = sent.split()
    tokens.append(ROOT)
    # build inside and outside charts
    inside_score = compute_inside_scores(tokens)
    outside_score = compute_outside_scores(tokens)
    iteration_logprob += inside_score
    # add expected counts of each possible event here
    add_expected_counts(inside_score)
    sents_counter += 1
    if sents_counter % 10 == 0: 
      sys.stdout.write('.')
      sys.stdout.flush()
    #print inside_score, outside_score
  if abs((inside_score - outside_score) / inside_score) > 0.01:
    print 'potentially a bug, inside_score = ', inside_score, ', but outside_score = ', outside_score
  
  # M step:
  # normalize expected counts and take the log
  normalize_expected_counts()
  # determine convergence
  print 'logprob = ', iteration_logprob, ', over ', sents_counter, ' sentences'
  if abs((prev_iteration_logprob - iteration_logprob) / iteration_logprob) < 0.01 or iterations_count > 10:
    print 'logprob converged after {} iterations'.format(iterations_count)
    break
  prev_iteration_logprob = iteration_logprob

