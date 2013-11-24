import itertools
import re
import time
import io
import sys
import argparse
from collections import defaultdict, namedtuple
import math
import pprint
import os
import ann_bies_map

Rule = namedtuple('Rule', 'lhs, rhs')
Terminal = namedtuple('Terminal', 'value')
Nonterminal = namedtuple('Nonterminal', 'value')
Subtree = namedtuple('Subtree', 'parent, children')

atb_nonterminals_complex_parts = set( [ 'J', 'I', 'D', 'MP', 'IV', 'POSS', 'MS', 'P', 'FS', 'DTV', 'FP', 'MD', 'FD', 'SBJ','OBJ','VN','NSUFF','MASC','FEM','GEN','DU','ACCGEN','ACC','CASE','DEF','INDEF','IVSUFF','SUBJ','MP','MOOD','SJ','I','S', 'ACC','NOM','GEN','MASC','FEM','DU','PL','SG','TTL','DIR', 'TMP', 'PRP','CLR','LOC','PRD','MNR','TPC','ADV','NOM','HLN','GEN','LGS','VOC', 'DO', 'PVSUFF', 'PART', 'F', 'DET', 'COMP' ] )

#avoid_complex_parts = set( [ 'DET_TYPO', 'DET_DIALECT', 'DIALECT', 'X' ] )

vocab_encoder = {}
vocab_decoder = []

def __encode(object):
  if object in vocab_encoder:
    return vocab_encoder[object]
  else:
    id = len(vocab_encoder)
    vocab_encoder[object] = id
    vocab_decoder.append(object)
    assert len(vocab_encoder) == len(vocab_decoder)
    return id

def __decode(id):
  return vocab_decoder[id]

'''
reads the parse string by iteratively replacing ( stuff ) with a subtree id. elegant but not efficient.

for example: 
>> ReadParseTree('(S (NP (A good) (N boys)) (VP (V sleep)))')
'''
uniq_nont = defaultdict(int)
def ReadParseTree(parse_string):
  rule_matcher = re.compile('(\([^\(\)]+\))')
  placeholder_prefix = '_|+|_'
  nodes = {}
  id = 0
  while parse_string.startswith('('):
    match = rule_matcher.search(parse_string)
    assert match, 'no match!!!'
    parts = match.groups()[0].strip('()').split()
    for i in xrange(len(parts)):
      if parts[i].startswith(placeholder_prefix):
        parts[i] = nodes[parts[i]]
      elif i == 0:
        if args.simplify_atb_nonterminals:
          if parts[i] in ann_bies_map.ann_bies.keys():
            parts[i] = ann_bies_map.ann_bies[parts[i]]
          else:
            all_symbols = re.findall(r'[A-Z]+', parts[i])
            good_symbols = [ all_symbols[0] ]
            for j in range(1, len(all_symbols)):
              if all_symbols[j] in atb_nonterminals_complex_parts:
                continue
              good_symbols.append( all_symbols[j] )
            if len(good_symbols) > 1 and good_symbols[0] == 'DET':
              del good_symbols[0]
            parts[i] = '_'.join(good_symbols)
        uniq_nont[ parts[i] ] += 1
        parts[i] = Nonterminal(value=parts[i])
      else:
        parts[i] = Terminal(value=parts[i])
    new_subtree = Subtree(parent=parts[0], children=parts[1:])
    nodes[placeholder_prefix+str(id)] = new_subtree
    parse_string = parse_string[:match.span()[0]] + placeholder_prefix + str(id) + parse_string[match.span()[1]:]
    id+=1
  assert parse_string.startswith(placeholder_prefix), 'parse string doesnt start with the placeholder'
  return new_subtree

'''
recursively prints out a parse tree. elegant but not efficient

for example: 
>> WriteParseSubtree(Subtree(parent='S', children=[Subtree(parent='NP', children=[Subtree(parent='A', children=['good']), Subtree(parent='N', children=['boys'])]), Subtree(parent='VP', children=[Subtree(parent='V', children=['sleep'])])]))

'( S ( NP ( A good ) ( N boys ) ) ( VP ( V sleep ) ) )'
'''
def WriteParseSubtree(subtree, decode=False):
  assert subtree is not None, 'subtree is none!'
  if decode:
    new_parts = ['(', str( __decode(subtree.parent.value) )]
  else:
    new_parts = ['(', str(subtree.parent.value)]
  for child in subtree.children:
    if type(child) is Terminal:
      if decode:
        new_parts.append( __decode(child.value) )
      else:
        new_parts.append(child.value)
    else:
      new_parts.append( WriteParseSubtree(child, decode) )
  new_parts.append(')')
  return ' '.join(new_parts)

'''
recursively returns a list of rules used in a subtree
'''
def ExtractRulesFromSubtree(subtree):
  rules = []
  rhs = []
  for child in subtree.children:
    if type(child) is Terminal:
      rhs.append( child )
    else:
      rhs.append( child.parent )
      rules.extend( ExtractRulesFromSubtree(child) )
  rules.append( Rule(lhs=subtree.parent, rhs=rhs) )
  return rules

'''
estimate a PCFG
'''
SOS = 'sentence_boundary'
def EstimatePcfgFromParseTrees(parse_trees):
  # count
  #unique_lhs, unique_rhs = set(), set()
  pcfg = {}
  counter = 0
  for subtree in parse_trees:
    counter += 1
    if counter % 1000 == 0:
      print counter, ' trees used to estimate pcfg so far.'
    #unique_lhs.add(Nonterminal(SOS))
    #if Nonterminal(value=subtree.parent) not in pcfg[ Nonterminal(value=SOS) ]:
    #  pcfg[ Nonterminal(value=SOS) ][ Nonterminal(value=subtree.parent) ] = 0
    #pcfg[ Nonterminal(value=SOS) ][ Nonterminal(value=subtree.parent) ] += 1 
    for rule in ExtractRulesFromSubtree(subtree):
      if rule.lhs not in pcfg:
        pcfg[rule.lhs] = {}
      if tuple(rule.rhs) not in pcfg[rule.lhs]:
        pcfg[rule.lhs][tuple(rule.rhs)] = 0
      pcfg[ rule.lhs ][ tuple(rule.rhs) ] += 1.0
  #    unique_lhs.add(rule.lhs)
  #    unique_rhs.add(tuple(rule.rhs))
  # dirichlet prior
  #dirichlet_prior = 0.01
  #for context in unique_lhs:
  #  for decision in unique_rhs:
  #    if decision not in pcfg[context]:
  #      pcfg[context][decision] = 0.0
  #    pcfg[context][decision] += dirichlet_prior
  # normalize
  for context in pcfg:
    context_count = 0.0
    for decision in pcfg[context].keys():
      context_count += pcfg[context][decision]
    for decision in pcfg[context].keys():
      pcfg[context][decision] = math.log(pcfg[context][decision]/context_count)
  return pcfg

'''
convert a parse tree into a pos tagging sequence
assumption: terminal items appear at the rhs of unary rules only
'''
def ConvertParseTreeIntoPosSequence(parse_tree):
  tokens, tags = [], []
  for rule in ExtractRulesFromSubtree(parse_tree):
    if type(rule.rhs[0]) is Terminal:
      tokens.append(rule.rhs[0].value)
      tags.append(rule.lhs.value)
  return (tokens, tags)

'''
estimate an hmm tagger using parse trees
'''
def EstimateHmmPosTaggerFromParseTrees(parse_trees):
  emissions, transitions = {}, {}
  unique_tags, unique_words = set(), set()
  # count
  for parse_tree in parse_trees:
    tokens, tags = ConvertParseTreeIntoPosSequence(parse_tree)
    tags.append(SOS)
    for i in xrange(len(tags)):
      if i != len(tags)-1:
        if tags[i] not in emissions:
          emissions[tags[i]] = {}
        if tokens[i] not in emissions[tags[i]]:
          emissions[tags[i]][tokens[i]] = 0.0
        emissions[tags[i]][tokens[i]] += 1.0
        unique_words.add(tokens[i])
        # end of i != len(tags)
      if tags[i-1] not in transitions:
        transitions[tags[i-1]] = {}
      if tags[i] not in transitions[tags[i-1]]:
        transitions[tags[i-1]][tags[i]] = 0.0
      transitions[tags[i-1]][tags[i]] += 1.0
      unique_tags.add(tags[i])
      # end of processing this position
    #end of processing this parse
  # tag set stats
  print '|tag set| = ', len(unique_tags)
  tagset_file = io.open('postagset', encoding='utf8', mode='w')
  for tag in unique_tags:
    if type(tag) is unicode:
      tagset_file.write(tag)
    elif type(tag) is Nonterminal or type(tag) is Terminal:
      print tag, 'is of type ', type(tag), '!!!' 
  tagset_file.close()
  # add symmetric dirichlet priors for emissions and transitions
  #transitions_dirichlet_alpha = 0.01
  #emissions_dirichlet_alpha = 0.1
  #for context in unique_tags:
  #  for decision in unique_tags:
  #    if decision not in transitions[context]:
  #      transitions[context][decision] = 0.0
  #    transitions[context][decision] += transitions_dirichlet_alpha
  #  if context == Nonterminal(value=SOS):
  #    continue
  #  for decision in unique_words:
  #    if decision not in emissions[context]:
  #      emissions[context][decision] = 0.0
  #    emissions[context][decision] += emissions_dirichlet_alpha
  # normalize
  for distribution in [emissions, transitions]:
    for context in distribution.keys():
      context_count = 0.0
      for decision in distribution[context].keys():
        context_count += distribution[context][decision]
      for decision in distribution[context].keys():
        distribution[context][decision] = math.log(distribution[context][decision]/context_count)
  return (transitions, emissions)

'''
three operations to convert general CFG to CNF: terminal->nonterminal, united-siblings, unary-merge
'''
UNARY_SEPARATOR = u'-unarycollapsed-'
TERMINAL_SEPARATOR = u'terminalinduced-'
UNITED_CHILDREN_SEPARATOR = u'-unitedchildren-'
def ConvertSubtreeIntoChomskyNormalForm(subtree):
  # base case to stop the recursion at CNF tree leaves
  if len(subtree.children) == 1 and type(subtree.children[0]) is Terminal:
    return subtree
  # CNF may have been violated. Lets make local fixes first.
  # one non-terminal child
  if len(subtree.children) == 1 and type(subtree.children[0]) is Subtree:
    #  new_parent = UNARY_SEPARATOR.join([subtree.parent.value, subtree.children[0].parent.value])
    new_parent = subtree.children[0].parent.value
    new_children = subtree.children[0].children
    subtree = Subtree(parent=Nonterminal(new_parent), children=new_children)
    return ConvertSubtreeIntoChomskyNormalForm(subtree)
  # a child needs to be a nonterminal
  if len(subtree.children) > 1:
    for i in xrange(len(subtree.children)):
      if type(subtree.children[i]) is Terminal:
        subtree.children[i] = Subtree(parent=Nonterminal(TERMINAL_SEPARATOR+subtree.children[i].value), \
                                        children = [subtree.children[i]])
  # more than two children
  if len(subtree.children) > 2:
    assert type(subtree.children[0]) is Subtree, 'leftmost child should be a subtree at this point'
    united_children_parent = subtree.children[1].parent.value
    united_children_children = [subtree.children[1]]
    for i in range(2, len(subtree.children)):
      united_children_parent += UNITED_CHILDREN_SEPARATOR + subtree.children[i].parent.value
      united_children_children.append(subtree.children[i])
    united_children = Subtree(parent = Nonterminal(united_children_parent), children = united_children_children)
    subtree.children[1:] = [united_children]
  # now, locally, this subtree should be in CNF
  if len(subtree.children) != 2 or type(subtree.children[0]) is not Subtree or type(subtree.children[1]) is not Subtree:
    print 'parent = ', subtree.parent
    print 'left_child = ', subtree.children[0]
    print 'right_child = ', subtree.children[1]
    pprint.pprint(subtree)
  assert len(subtree.children) == 2 and type(subtree.children[0]) is Subtree and type(subtree.children[1]) is Subtree, \
         'subtree not in cnf'
  # recursively call this method to make sure individual children subtrees are also in CNF
  for i in xrange(len(subtree.children)):
    subtree.children[i] = ConvertSubtreeIntoChomskyNormalForm(subtree.children[i])
  return subtree

'''
convert a CNF parse to the original parse
'''
# THIS IS SCREWED UP
def ConvertCnfSubtreeIntoOriginalSubtree(subtree):
  if subtree.parent.startswith(TERMINAL_SEPARATOR):
    assert len(subtree.children) == 1 and type(subtree.children[0]) is Terminal, 'mysterious ' + TERMINAL_SEPARATOR
    return subtree.children[0]
  if subtree.parent.find(UNARY_SEPARATOR) >= 0:
    united_parents = subtree.parent.split(UNARY_SEPARATOR)
    new_subtree = Subtree(parent = UNARY_SEPARATOR.join(united_parents[0:-1]), \
                            children = Subtree(parent = Nonterminal(united_parents[-1]), children = subtree.children))
    return ConvertCnfSubtreeIntoOriginalSubtree(new_subtree)
  if len(subtree.children) == 2 and subtree.children[1].find(UNITED_CHILDREN_SEPARATOR) >= 0:
    united_children = subtree.children[1]
    subtree.children[1:] = []
    for i in xrange(united_children.children):
      divided_child = Subtree(parent = united_children.children[i].parent, children = united_children.children[i].children)
      subtree.children.append(divided_child)
  for i in xrange(len(subtree.children)):
    if type(subtree.children[i]) is not Terminal:
      subtree.children[i] = ConvertCnfSubtreeIntoOriginalSubtree(subtree.children[i])
  return subtree

'''
takes a list and count how many each element repeats
'''
def CountRuleFrequencies(rules):
  freq = defaultdict(int)
  for rule in rules:
    assert type(rule.rhs) is list, 'rhs should be a list'
    rule_hash = [rule.lhs]
    rule_hash.extend(rule.rhs)
    rule_hash = tuple(rule_hash)
    freq[rule_hash] += 1
  return freq

''' 
returns a tuple: (unnormalized_precision, candidate_rules_count, unnormalized_recall, reference_rules_count)
'''
def EvaluateParseTree(candidate_parse, reference_parse):
  # read candidate/reference rules
  candidate_rules = CountRuleFrequencies( ExtractRulesFromSubtree(candidate_parse) )
  reference_rules = CountRuleFrequencies( ExtractRulesFromSubtree(reference_parse) )
  # compute precision
  candidate_rules_count, unnormalized_precision = 0.0, 0.0
  for rule in candidate_rules:
    candidate_rules_count += candidate_rules[rule]
    unnormalized_precision += min(candidate_rules[rule], reference_rules[rule])
  unnormalized_precision /= candidate_rules_count
  # compute recall
  reference_rules_count, unnormalized_recall = 0.0, 0.0
  for rule in reference_rules:
    reference_rules_count += reference_rules[rule]
    unnormalized_recall += min(candidate_rules[rule], reference_rules[rule])
  unnormalized_recall /= reference_rules_count
  return (unnormalized_precision, candidate_rules_count, unnormalized_recall, reference_rules_count)

'''
returns a tuple (correct, all)
'''
def EvaluatePos(candidate_postags, reference_postags):  
  if len(candidate_postags) != len(reference_postags):
    return (0, len(reference_postags))
  correct = 0
  for i in xrange(len(candidate_postags)):
    if candidate_postags[i] == reference_postags[i]:
      correct += 1
  return (correct, len(reference_postags))

'''
given a complete table
'''
def CykBacktrack(table, index):
  subtree = Subtree(parent = table[index].lhs, children = [])
  # basecase
  if table[index].leftchild_index is None:
    subtree.children.append( table[index].rhs[0] )
    return subtree
  subtree.children.append(CykBacktrack(table, table[index].leftchild_index))
  subtree.children.append(CykBacktrack(table, table[index].rightchild_index))
  return subtree

'''
cyk
table maps a tuple (length, start, nonterminal) to CykTuple
'''
def CykParse(tokens, pcfg, start_symbol, preterminals, rhs_to_lhs_index, dd_u=defaultdict(float)):
  CykTuple = namedtuple('CykTuple', 'lhs, rhs, logprob, leftchild_index, rightchild_index')
  table = {}
  cell_to_nonterminals_index = defaultdict(set)
  for length in range(1, len(tokens)+1):
    #print 'length = ', length
    for start in range(0, len(tokens)-length+1):
      #print 'start = ', start
      # find a rule that matches this token
      if length == 1:
        rhs = (Terminal(value=tokens[start]),)
        # token is an OOV
        if rhs not in rhs_to_lhs_index:
          if not args.handle_oov: 
            return None
          #print 'token ', __decode(tokens[start]), ' is OOV :-/'
          for nonterminal in preterminals:
            if (start, nonterminal.value) in dd_u:
              dd_term = dd_u[ (start, nonterminal.value) ]
            else:
              dd_term = 0.0
            cell_to_nonterminals_index[(length, start)].add(nonterminal)
            table[(length, start, nonterminal)] = CykTuple(lhs=nonterminal,
                                                           rhs=(Terminal(value=tokens[start]),),
                                                           logprob=0.0 + dd_term,
                                                           leftchild_index=None,
                                                           rightchild_index=None)
          # end of token is an OOV
        else:  
          for lhs in rhs_to_lhs_index[rhs]:
            cell_to_nonterminals_index[(length, start)].add(lhs)
            if (start, lhs.value) in dd_u:
              dd_term = dd_u[ (start, lhs.value) ]
            else:
              dd_term = 0.0
            table[(length, start, lhs)] = CykTuple(lhs=lhs, 
                                                   rhs=rhs, 
                                                   logprob=pcfg[lhs][rhs] + dd_term,
                                                   leftchild_index=None, 
                                                   rightchild_index=None)
        # end of length == 1
      # for each possible split point
      for split in range(start+1, start+length):
        #print 'split = ', split
        leftchild_index, rightchild_index = (split - start, start), (length - split + start, split)
        #print 'evaluating ', len(cell_to_nonterminals_index[leftchild_index]), ' X ', len(cell_to_nonterminals_index[rightchild_index]), ' possible rules with this split at this cell'
        for left_nonterminal in cell_to_nonterminals_index[leftchild_index]:
          for right_nonterminal in cell_to_nonterminals_index[rightchild_index]:
            rhs = (left_nonterminal, right_nonterminal)
            left_logprob = table[(leftchild_index[0], leftchild_index[1], left_nonterminal)].logprob
            right_logprob = table[(rightchild_index[0], rightchild_index[1], right_nonterminal)].logprob
            if rhs not in rhs_to_lhs_index:
              continue
            for lhs in rhs_to_lhs_index[rhs]:
              match = CykTuple(lhs=lhs, rhs=rhs, 
                               logprob = left_logprob + right_logprob + pcfg[lhs][rhs],
                               leftchild_index=(leftchild_index[0], leftchild_index[1], left_nonterminal), 
                               rightchild_index=(rightchild_index[0], rightchild_index[1], right_nonterminal))
              if (length, start, lhs) not in table or table[(length, start, lhs)].logprob < match.logprob:
                cell_to_nonterminals_index[(length, start)].add(lhs)
                table[ (length, start, lhs) ] = match
            # done processing this rule
          # done processing this lhs
        # done processing this potential split point
      # done processing this cell in the cyk table
    # done processing this row in the cyk table
  # done processing all rows in the cyk table, time to backtrack (if any complete parse is available)
  if (len(tokens), 0, Nonterminal(start_symbol)) not in table:
    return None
  tree = CykBacktrack(table, (len(tokens), 0, Nonterminal(start_symbol)))
  return tree

def HmmViterbi(tokens, transitions, emissions, words_to_states, dd_u=defaultdict(float)):
  alpha = {}
  AlphaIndex = namedtuple('AlphaIndex', 'position, state')
  AlphaValue = namedtuple('AlphaValue', 'logprob, prev_state')
  alpha[ AlphaIndex(position=-1, state=__encode(SOS)) ] = AlphaValue(logprob=0, prev_state=None) # you are at the start-of-sent state with prob 1 at the beginning
  position_to_states = {}
  position_to_states[-1] = [__encode(SOS)]
  for position in range(0, len(tokens)):
    # OOV
    observation = tokens[position]
    if observation not in words_to_states:
      if args.handle_oov:
        words_to_states[observation] = set()
        for state in emissions.keys():
          emissions[state][observation] = 0
          words_to_states[observation].add(state)
      else:
        return None
      # end of OOV
    for current_state in words_to_states[observation]:      
      for previous_state in position_to_states[position-1]:        
        if current_state not in transitions[previous_state]:
          continue
        if (position, current_state) in dd_u:
          dd_term = - dd_u[ (position, current_state) ]
        else:
          dd_term = 0.0
        forward_score = alpha[ AlphaIndex(position=position-1, state=previous_state) ].logprob + \
            transitions[previous_state][current_state] + \
            emissions[current_state][tokens[position]] + dd_term
        if AlphaIndex(position=position, state=current_state) not in alpha or \
              alpha[AlphaIndex(position=position, state=current_state)] < AlphaValue(logprob=forward_score, prev_state=previous_state):
          alpha[AlphaIndex(position=position, state=current_state)] = \
              AlphaValue(logprob=forward_score, prev_state=previous_state)
          if position not in position_to_states:
            position_to_states[position] = set()
          position_to_states[position].add(current_state)
        # done considering a particular previous state
      # done considering a particular current state
    # don't waste your time if none of the states fit this position
    if position not in position_to_states:
      return None
    # done processing a particular position
  # reached the end of the sentence, but haven't considered the transition to end-of-sentence state
  # add alpha entries which take into consideration end of sentence boundaries
  for previous_state in position_to_states[len(tokens)-1]:
    index = AlphaIndex(position=len(tokens)-1, state=previous_state)
    if __encode(SOS) not in transitions[previous_state]:
      continue
    forward_score = alpha[index].logprob + transitions[previous_state][__encode(SOS)]
    if AlphaIndex(position=len(tokens), state=__encode(SOS)) not in alpha or \
          alpha[ AlphaIndex(position=len(tokens), state=__encode(SOS)) ] < AlphaValue(logprob=forward_score, prev_state=previous_state):
      alpha[ AlphaIndex(position=len(tokens), state=__encode(SOS)) ] = AlphaValue(logprob=forward_score, prev_state=previous_state)
  # backtrack
  current_state = __encode(SOS)
  viterbi_tag_sequence = []
  for position in reversed(range(1, len(tokens)+1)):
    if AlphaIndex(position, current_state) not in alpha:
      return None
    prev_state = alpha[ AlphaIndex(position, current_state) ].prev_state
    viterbi_tag_sequence.insert(0, prev_state)
    current_state = prev_state
  return viterbi_tag_sequence

def DualDecomposition(tokens, transitions, emissions, words_to_states, 
                      pcfg, start_symbol, preterminals, rhs_to_lhs_index,
                      max_dd_iters):
  u = defaultdict(float)
  agreement = True
  delta_k = 0.06
  for k in xrange(max_dd_iters):
    delta_k *= 2
    tree = CykParse(tokens, pcfg, start_symbol, preterminals, rhs_to_lhs_index, u)
    tagging = HmmViterbi(tokens, transitions, emissions, words_to_states, u)
    if tree == None or tagging == None:
      return (tree, tagging)
    (dummy_tokens, tree_preterminals) = ConvertParseTreeIntoPosSequence(tree)
    agreement = True
    for position in xrange(len(tagging)):
      if tagging[position] != tree_preterminals[position]:
        agreement = False
        #print tagging[position], ' != ', tree_preterminals[position]
        u[ (position, tree_preterminals[position] ) ] -= delta_k
        #print 'u[(', position, ', ', tree_preterminals[position], ')] = ', u[ (position, tree_preterminals[position] ) ]
        u[ (position, tagging[position] ) ] += delta_k
        #print 'u[(', position, ', ', tagging[position], ')] = ', u[ (position, tagging[position] ) ]
    if agreement:
      print 'DUAL DECOMPOSITION CONVERGED AFTER ', k, ' iterations :)'
      break
    
  if not agreement:
    print 'dual decomposition didnt converge after ', k, ' iterations :('
  
  return (tree, tagging)

def ReadTreebankDir(treebank_dir):
  parses = []
  for filename in os.listdir(treebank_dir):
    filename = os.path.join(treebank_dir, filename)
    if not filename.endswith('.tree'): continue
    for line in io.open(filename, encoding='utf8'):
      parses.append(ReadParseTree(line.strip()))
  #print 'len(uniq_nont)  = ', len(uniq_nont)
  for n in uniq_nont:
    print n
  return parses


# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("--decode_testset", default=True)
argParser.add_argument("--simplify_atb_nonterminals", default=True)
argParser.add_argument("--max_sent_length", type=int, default=20)
argParser.add_argument("--create_hw", default=False)
argParser.add_argument("--solve_hw", default=True)
argParser.add_argument("--handle_oov", default=True)
argParser.add_argument("--treebank_dir", default='arabic_penntreebank_v3.2/')
argParser.add_argument("--vocab_filename", default='vocab')
argParser.add_argument("--dev_sents_filename", default='dev_sents')
argParser.add_argument("--dev_parses_filename", default='dev_parses')
argParser.add_argument("--dev_postags_filename", default='dev_postags')
argParser.add_argument("--test_sents_filename", default='test_sents')
argParser.add_argument("--test_parses_filename", default='test_parses')
argParser.add_argument("--test_postags_filename", default='test_postags')
argParser.add_argument("--train_sents_filename", default='train_sents')
argParser.add_argument("--train_parses_filename", default='train_parses')
argParser.add_argument("--train_postags_filename", default='train_postags')
argParser.add_argument("--hmm_transitions_filename", default='hmm_trans')
argParser.add_argument("--hmm_emissions_filename", default='hmm_emits')
argParser.add_argument("--pcfg_filename", default='pcfg')
argParser.add_argument("--candid_dev_parses_filename", default='candid_dev_parses')
argParser.add_argument("--candid_dev_postags_filename", default='candid_dev_postags')
argParser.add_argument("--candid_test_parses_filename", default='candid_test_parses')
argParser.add_argument("--candid_test_postags_filename", default='candid_test_postags')
argParser.add_argument("--candid_dev_dd_parses_filename", default='candid_dev_dd_parses')
argParser.add_argument("--candid_dev_dd_postags_filename", default='candid_dev_dd_postags')
argParser.add_argument("--candid_test_dd_parses_filename", default='candid_test_dd_parses')
argParser.add_argument("--candid_test_dd_postags_filename", default='candid_test_dd_postags')
args = argParser.parse_args()


if args.create_hw:
  print 'create_hw = ', args.create_hw
  # read all of the Arabic treebank
  parses = ReadTreebankDir(args.treebank_dir)
  print len(parses), ' parses read.'
  
  # each ninth and tenth example will be put in a dev and test set
  tree_id = 0
  dev_sents_file = io.open(args.dev_sents_filename, encoding='utf8', mode='w')
  dev_parses_file = io.open(args.dev_parses_filename, encoding='utf8', mode='w')
  dev_postags_file = io.open(args.dev_postags_filename, encoding='utf8', mode='w')
  test_sents_file = io.open(args.test_sents_filename, encoding='utf8', mode='w')
  test_parses_file = io.open(args.test_parses_filename, encoding='utf8', mode='w')
  test_postags_file = io.open(args.test_postags_filename, encoding='utf8', mode='w')
  train_sents_file = io.open(args.train_sents_filename, encoding='utf8', mode='w')
  train_parses_file = io.open(args.train_parses_filename, encoding='utf8', mode='w')
  train_postags_file = io.open(args.train_postags_filename, encoding='utf8', mode='w')
  train, dev, test = [], [], []
  train_sents, dev_sents, test_sents = [], [], []
  for tree in parses:
    if tree.parent != Nonterminal('S'): continue
    # convert trees into chomsky normal form
    tree = ConvertSubtreeIntoChomskyNormalForm(tree)
    (tokens, tags) = ConvertParseTreeIntoPosSequence(tree)
    sent = u'{0}\n'.format(' '.join(tokens))
    parse_string = u'{0}\n'.format(WriteParseSubtree(tree))
    #print parse_string
    postags_string = u'{0}\n'.format(' '.join(tags))
    # distribute on train, dev, test
    if tree_id % 100 < 4 and len(tokens) <= args.max_sent_length:
      dev.append(tree)
      dev_sents_file.write(sent)
      dev_sents.append(sent)
      dev_parses_file.write(parse_string)
      dev_postags_file.write(postags_string)
    elif tree_id % 100 < 8 and len(tokens) <= args.max_sent_length:
      test.append(tree)
      test_sents_file.write(sent)
      test_sents.append(sent)
      test_parses_file.write(parse_string)
      test_postags_file.write(postags_string)
    else:
      train.append(tree)
      train_sents_file.write(sent)
      train_sents.append(sent)
      train_parses_file.write(parse_string)
      train_postags_file.write(postags_string)
    tree_id += 1
  dev_sents_file.close()
  dev_parses_file.close()
  dev_postags_file.close()
  test_sents_file.close()
  test_parses_file.close()
  test_postags_file.close()
  train_sents_file.close()
  train_parses_file.close()
  train_postags_file.close()

  # estimate hmm model
  (transitions, emissions) = EstimateHmmPosTaggerFromParseTrees(train)
  hmm_transitions_file = io.open(args.hmm_transitions_filename, encoding='utf8', mode='w')
  for context in transitions.keys():
    for decision in transitions[context].keys():
      hmm_transitions_file.write(u'{0}\t{1}\t{2}\n'.format(context.strip(), decision.strip(), transitions[context][decision]))
  hmm_transitions_file.close()
  hmm_emissions_file = io.open(args.hmm_emissions_filename, encoding='utf8', mode='w')
  for context in emissions.keys():
    for decision in emissions[context].keys():
      hmm_emissions_file.write(u'{0}\t{1}\t{2}\n'.format(context.strip(), decision.strip(), emissions[context][decision]))
  hmm_emissions_file.close()

  # estimate pcfg
  pcfg = EstimatePcfgFromParseTrees(train)
  pcfg_file = io.open(args.pcfg_filename, encoding='utf8', mode='w')
  for context in pcfg.keys():
    for decision in pcfg[context].keys():
      decision_string = ''
      for i in xrange(len(decision)):
        decision_string += decision[i].value + u' '
      pcfg_file.write(u'{0}\t{1}\t{2}\n'.format(context.value, decision_string.strip(), pcfg[context][decision]))
  pcfg_file.close()
  pcfg_file.close()

  with io.open(args.vocab_filename, encoding='utf8', mode='w') as vocab_file:
    vocab = set()
    for sent in itertools.chain(train_sents, dev_sents, test_sents):
      for token in sent.strip().split():
        vocab.add(token)
    for wordtype in vocab:
      vocab_file.write(u'{0}\n'.format(wordtype))

if args.solve_hw:
  # read vocab
  with io.open(args.vocab_filename, encoding='utf8') as vocab_file:
    vocab = set()
    for wordtype in vocab_file:
      vocab.add(wordtype.strip())

  # read sents
  print 'reading sents...'
  train_sents_file, dev_sents_file, test_sents_file = io.open(args.train_sents_filename, encoding='utf8'), io.open(args.dev_sents_filename, encoding='utf8'), io.open(args.test_sents_filename, encoding='utf8')
  train_sents, dev_sents, test_sents = train_sents_file.readlines(), dev_sents_file.readlines(), test_sents_file.readlines()
  train_sents_file.close(); dev_sents_file.close(); test_sents_file.close();

  # read pcfg
  print 'reading pcfg...'
  pcfg = {}
  rhs_to_lhs_index = {}
  nonterminals = set()
  with io.open(args.pcfg_filename, encoding='utf8') as pcfg_file:
    for line in pcfg_file:
      context_string, decision_string, logprob = line.strip().split('\t')
      context = Nonterminal(value=__encode(context_string))
      nonterminals.add(context)
      rhs = []
      for rhs_element in decision_string.split():
        if rhs_element in vocab:
          rhs.append( Terminal(value=__encode(rhs_element)) )
        else:
          rhs.append( Nonterminal(value=__encode(rhs_element)) )
      decision = tuple( rhs )
      if context not in pcfg:
        pcfg[context] = {}
      pcfg[context][decision] = float(logprob)
      if decision not in rhs_to_lhs_index:
        rhs_to_lhs_index[decision] = set()
      rhs_to_lhs_index[decision].add(context)
  #print 'len(nonterminals) = ', len(nonterminals)
  #for nonterminal in nonterminals:
  #  print __decode(nonterminal.value)

  # read hmm
  transitions, emissions = {}, {}
  words_to_states = {}
  with io.open(args.hmm_transitions_filename, encoding='utf8') as hmm_transitions_file:
    for line in hmm_transitions_file:
      context, decision, logprob = line.strip().split('\t')
      context, decision = __encode(context), __encode(decision)
      if context not in transitions:
        transitions[context] = {}
      transitions[context][decision] = float(logprob)
  with io.open(args.hmm_emissions_filename, encoding='utf8') as hmm_emissions_file:
    for line in hmm_emissions_file:
      context, decision, logprob = line.strip().split('\t')
      context, decision = __encode(context), __encode(decision)
      if context not in emissions:
        emissions[context] = {}
      emissions[context][decision] = float(logprob)
      if decision not in words_to_states:
        words_to_states[decision] = set()
      words_to_states[decision].add(context)

  # use the preterminals as potential unary parse rules for OOV words
  preterminals = list(emissions.keys())
  for i in xrange(len(preterminals)):
    preterminals[i] = Nonterminal(value=preterminals[i])

  print 'now decoding ...'
  # use the pcfg models to parse something
  # use the hmm model to parse something
  if args.decode_testset:
    candid_parses_file = io.open(args.candid_test_parses_filename, encoding='utf8', mode='w')
    candid_postags_file = io.open(args.candid_test_postags_filename, encoding='utf8', mode='w')
    candid_dd_parses_file = io.open(args.candid_test_dd_parses_filename, encoding='utf8', mode='w')
    candid_dd_postags_file = io.open(args.candid_test_dd_postags_filename, encoding='utf8', mode='w')
    sents = test_sents
  else:
    candid_parses_file = io.open(args.candid_dev_parses_filename, encoding='utf8', mode='w')
    candid_postags_file = io.open(args.candid_dev_postags_filename, encoding='utf8', mode='w')    
    candid_dd_parses_file = io.open(args.candid_dev_dd_parses_filename, encoding='utf8', mode='w')
    candid_dd_postags_file = io.open(args.candid_dev_dd_postags_filename, encoding='utf8', mode='w')    
    sents = dev_sents
  parse_failures, tagging_failures = 0, 0
  sent_id = 0
  for i in xrange(len(sents)):
    sent = sents[i].strip()
    tokens = sent.split()
    for i in xrange(len(tokens)):
      tokens[i] = __encode(tokens[i])
    print 'now processing sent ', sent_id, ': ', sent
    sent_id += 1
    # parse
    tree = CykParse(tokens, pcfg, __encode(u'S'), preterminals, rhs_to_lhs_index)
    if tree is None:
      parse_failures += 1
      candid_parses_file.write(u'\n')
      print 'parse: None'
    else:
      candid_parses_file.write(u'{0}\n'.format(WriteParseSubtree(tree, decode=True)))
      print 'parse: ', WriteParseSubtree(tree, decode=True)
    # tag
    tagging = HmmViterbi(tokens, transitions, emissions, words_to_states)
    if tagging:
      for i in xrange(len(tagging)):
        tagging[i] = __decode(tagging[i])
    print 'tagging: ', tagging
    print 
    if tagging is None:
      tagging_failures += 1
      candid_postags_file.write(u'\n')
    else:
      candid_postags_file.write(u'{0}\n'.format( ' '.join(tagging)))
    # dual decomposition
    (dd_tree, dd_tagging) = DualDecomposition(tokens, transitions, emissions, words_to_states,
                                              pcfg, __encode(u'S'), preterminals, rhs_to_lhs_index, 10)
    if dd_tree is None:
      candid_dd_parses_file.write(u'\n')
    else:
      candid_dd_parses_file.write(u'{0}\n'.format(WriteParseSubtree(dd_tree, decode=True)))
    
    if dd_tagging is None:
      candid_dd_postags_file.write(u'\n')
    else:
      for i in xrange(len(dd_tagging)):
        dd_tagging[i] = __decode(dd_tagging[i])
      candid_dd_postags_file.write(u'{0}\n'.format( ' '.join(dd_tagging)))

  print tagging_failures, ' failures while tagging ', len(sents), ' sents'
  print parse_failures, ' failures while cyk parsing ', len(sents), ' sents' 
  candid_parses_file.close()
  candid_postags_file.close()
