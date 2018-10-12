import heapq
import math
import re

class Beam(object):
    # For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
    # This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an
    # incomplete one since (0.5, False) < (0.5, True)

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        #print("prob: {}, prefix:{}".format(prob, prefix))
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def sort(self, reverse=True):
        self.heap.sort(key=lambda x:x[0], reverse=reverse)

    def best_sentences(self, complete=True):
        self.sort()
        result = [re.sub('\s+([.,])', r'\1'," ".join(l[2]).rstrip()) for l in self.heap if l[1] or not complete]
        return result

    def best_sentence(self, complete=True):
        self.sort()
        for l in self.heap:
            if complete and l[1]:
                return [re.sub('\s+([.,])', r'\1', " ".join(l[2]).rstrip())]
            else:
                if not complete:
                    return [re.sub('\s+([.,])', r'\1', " ".join(l[2]).rstrip())]
        return [""]

    def __iter__(self):
        return iter(self.heap)

def beamsearch(probabilities_function, string=None, beam_width=10, length_min=10, length_max=20):
    string_len = len(string)
    prev_beam = Beam(beam_width)
    prev_beam.add(0.0, False, string)
    depth = 0
    while True:
        curr_beam = Beam(beam_width)
        depth = depth + 1
        #Add complete sentences that do not yet have the best probability to the current beam,
        #the rest prepare to add more words to them.
        for (prefix_prob, complete, prefix) in prev_beam:
            if complete == True:
                curr_beam.add(prefix_prob, True, prefix)
            else:
                #Get probability of each possible next word for the incomplete prefix.
                result = probabilities_function(prefix)
                # print("prefix: {}".format(prefix))
                # print("result: {}".format(result))
                for (next_prob, next_word) in result:
                    if next_word == 'xbos':
                        #if next word is the end token then mark prefix as complete and leave out the end token
                        curr_beam.add(prefix_prob + math.log10(next_prob), True, prefix)
                    else: #if next word is a non-end token then mark prefix as incomplete
                        curr_beam.add(prefix_prob + math.log10(next_prob), False, prefix+[next_word])

        (best_prob, best_complete, best_prefix) = max(curr_beam)
        #print("dept: {}, length: {}, clip_len: {}".format(depth, len(best_prefix), clip_len))

        #if best_complete == True or len(best_prefix)-1 == clip_len:
        if ((depth >= length_min) and best_complete and (len(best_prefix) != string_len)) \
                or depth == length_max:
            # if most probable prefix is a complete sentence or has a length that
            # exceeds the clip length (ignoring the start token) then return it
            #print("depth: {}".format(depth))
            #return (best_prefix[0:], best_prob)
            return curr_beam
            #return best sentence without the start token and together with its probability

        prev_beam = curr_beam
