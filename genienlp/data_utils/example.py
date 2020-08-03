#
# Copyright (c) 2019-2020 The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import NamedTuple, List, Tuple
import itertools
import random

from .numericalizer.sequential_field import SequentialField


class Example(NamedTuple):
    example_id: str
    # for each field in the example, we store the tokenized sentence, and a boolean mask
    # indicating whether the token is a real word (subject to word-piece tokenization)
    # or it should be treated as an opaque symbol
    context: List[str]
    context_word_mask: List[bool]
    context_feature: List[Tuple[float]]
    question: List[str]
    question_word_mask: List[bool]
    question_feature: List[Tuple[float]]
    answer: List[str]
    answer_word_mask: List[bool]
    answer_feature: List[Tuple[float]]
    context_plus_question: List[str]
    context_plus_question_word_mask: List[bool]
    context_plus_question_feature: List[Tuple[float]]

    vocab_fields = ['context', 'question', 'answer']

    @staticmethod
    def from_raw(example_id: str, context: str, question: str, answer: str, tokenize, split, lower=False):
        args = [example_id]

        for argname, arg in (('context', context), ('question', question), ('answer', answer),
                             ('context_question', context+' '+question)):
            words, mask, feature = tokenize(arg.rstrip('\n'), split, field_name=argname, answer=answer)
            if mask is None:
                mask = [True for _ in words]
            if lower:
                words = [word.lower() for word in words]
            args.append(words)
            args.append(mask)
            
            # type would be empty list for answer field
            args.append(feature)
        
        return Example(*args)


class Batch(NamedTuple):
    example_id: List[str]
    context: SequentialField
    question: SequentialField
    answer: SequentialField
    decoder_vocab: object
    
    @staticmethod
    def from_examples(examples, numericalizer, device=None, paired=False, max_pairs=None, groups=None,
                      append_question_to_context_too=False, override_question=None, override_context=None, num_features=0):
        
        assert all(isinstance(ex.example_id, str) for ex in examples)
        decoder_vocab = numericalizer.decoder_vocab.clone()
        max_context_len, max_question_len, max_answer_len = -1, -1, -1
        
        override_question_mask = None
        if override_question:
            override_question = override_question.split()
            override_question_mask = [True for _ in override_question]
            # dummy values
            zip_list = []
            if num_features >= 1:
                zip_list.append([0 for _ in override_question])
            if num_features >= 2:
                zip_list.append([1.0 for _ in override_question])
            override_question_feature = list(zip(*zip_list))
                
        override_context_mask = None
        override_context_feature = []
        if override_context:
            override_context = override_context.split()
            override_context_mask = [True for _ in override_context]
            # dummy values
            zip_list = []
            if num_features >= 1:
                zip_list.append([0 for _ in override_context])
            if num_features >= 2:
                zip_list.append([1.0 for _ in override_context])
            override_context_feature = list(zip(*zip_list))
                
        if paired:
            example_pairs = []
            
            # get all possible combinations of related example pairs
            for i in range(0, len(examples), groups):
                related_examples = [examples[j] for j in range(i, i+groups)]
                example_pairs.extend(itertools.product(related_examples, related_examples))
            # filter out pairs with same sentences
            example_pairs = [ex_pair for ex_pair in example_pairs if ex_pair[0] != ex_pair[1]]
            
            # shuffle example orders and select first max_pairs of them
            random.shuffle(example_pairs)
            example_pairs = example_pairs[:max_pairs]
            
            example_ids = [ex_a.example_id + '@' + ex_b.example_id for ex_a, ex_b in example_pairs]
            if override_question:
                question_inputs = [((override_question, override_question_mask),
                                    (override_question, override_question_mask))
                                   for _ in example_pairs]
            else:
                question_inputs = [((ex_a.question, ex_a.question_word_mask),
                                    (ex_b.question, ex_b.question_word_mask))
                                   for ex_a, ex_b in example_pairs]
                
            if append_question_to_context_too:
                context_inputs = [((ex_a.context_plus_question,
                                    ex_a.context_plus_question_word_mask),
                                    (ex_b.context_plus_question,
                                     ex_b.context_plus_question_word_mask))
                                  for ex_a, ex_b in example_pairs]
            elif override_context:
                context_inputs = [((override_context, override_context_mask),
                                    (override_context, override_context_mask))
                                   for _ in example_pairs]
            else:
                context_inputs = [((ex_a.context, ex_a.context_word_mask),
                                    (ex_b.context, ex_b.context_word_mask))
                                   for ex_a, ex_b in example_pairs]


            answer_inputs = [((ex_a.answer, ex_a.answer_word_mask), (ex_b.answer, ex_b.answer_word_mask))
                             for ex_a, ex_b in example_pairs]

            all_example_ids_pair = example_ids
            all_context_inputs_pair = numericalizer.encode_pair(context_inputs, decoder_vocab, device=device)
            all_question_inputs_pair = numericalizer.encode_pair(question_inputs, decoder_vocab, device=device)
            all_answer_inputs_pair = numericalizer.encode_pair(answer_inputs, decoder_vocab, device=device)

            max_context_len = all_context_inputs_pair.value.size(1)
            max_question_len = all_question_inputs_pair.value.size(1)
            max_answer_len = all_answer_inputs_pair.value.size(1)

        # process single examples
        example_ids = [ex.example_id for ex in examples]
        if override_question:
            question_inputs = [(override_question, override_question_mask, override_question_feature) for _ in examples]
        else:
            question_inputs = [(ex.question, ex.question_word_mask, ex.question_feature) for ex in examples]

        if append_question_to_context_too:
            context_inputs = [(ex.context_plus_question, ex.context_plus_question_word_mask, ex.context_plus_question_feature) for ex in examples]
        elif override_context:
            context_inputs = [(override_context, override_context_mask, override_context_feature) for _ in examples]
        else:
            context_inputs = [(ex.context, ex.context_word_mask, ex.context_feature) for ex in examples]

        answer_inputs = [(ex.answer, ex.answer_word_mask, ex.answer_feature) for ex in examples]
        
        all_example_ids_single = example_ids
        all_context_inputs_single = numericalizer.encode_single(context_inputs, decoder_vocab,
                                                                device=device, max_length=max_context_len-2)
        all_question_inputs_single = numericalizer.encode_single(question_inputs, decoder_vocab,
                                                                 device=device, max_length=max_question_len-2)
        all_answer_inputs_single = numericalizer.encode_single(answer_inputs, decoder_vocab,
                                                               device=device, max_length=max_answer_len-2)
    
        if paired:
            all_example_ids = all_example_ids_single + all_example_ids_pair
            all_context_inputs = SequentialField.from_tensors([all_context_inputs_single, all_context_inputs_pair])
            all_question_inputs = SequentialField.from_tensors([all_question_inputs_single, all_question_inputs_pair])
            all_answer_inputs = SequentialField.from_tensors([all_answer_inputs_single, all_answer_inputs_pair])
        else:
            all_example_ids = all_example_ids_single
            all_context_inputs = all_context_inputs_single
            all_question_inputs = all_question_inputs_single
            all_answer_inputs = all_answer_inputs_single
        
        return Batch(all_example_ids,
                     all_context_inputs,
                     all_question_inputs,
                     all_answer_inputs,
                     decoder_vocab)
