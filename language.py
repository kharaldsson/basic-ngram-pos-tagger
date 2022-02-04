import sys
import re
from collections import Counter
import math
import pandas as pd


class Ngram:
    def __init__(self, text):
        self.text = text
        self.tokens = re.split(r"\s", self.text)
        self.type = len(self.tokens)
        self.lgprob = None
        self.comments = None
        self.output = None

    def format_output(self):
        w1 = self.tokens[0]
        w2 = self.tokens[1]
        if self.type == 3:
            w3 = self.tokens[2]
            if self.comments is None:
                self.output = "lg P(" + str(w3) + " | " + str(w1) + " " + str(w2) + ") = " + str(
                    self.lgprob)
            else:
                self.output = "lg P(" + str(w3) + " | " + str(w1) + " " + str(w2) + ") = " + str(self.lgprob) + " " + str(
                self.comments)
        else:
            self.output = "lg P(" + str(w2) + " | " + str(w1) + ") = " + str(self.lgprob) + " " + str(self.comments)

        self.output = self.output.strip()


class Sentence:
    def __init__(self, tokens, number):
        self.tokens = tokens
        self.text = ' '.join(self.tokens)
        self.number = number
        self.ngrams = []
        self.ngram_probs = {}
        self.num_sents = None
        self.num_words = None
        self.num_oov = None
        self.lgprob = None
        self.ppl = None


class LanguageModel:
    def __init__(self):
        self.training_data = None
        self.calculate_probs = False
        self.num_tokens = {'unigrams': None, 'bigrams': None, 'trigrams': None}
        self.num_types = {'unigrams': None, 'bigrams': None, 'trigrams': None}
        self.unigrams = []
        self.bigrams = []
        self.trigrams = []
        self.ngram_probs = {'unigrams': {}, 'bigrams': {}, 'trigrams': {}}
        self.test_probs = []
        self.test_summary = {'lgprob': None, 'ave_lgprob': None, 'ppl': None
            , 'sent_num': None, 'word_num': None, 'oov_num': None
                             }

    def build_model(self, path_data_in, how):
        bos_symbol = '<s>'
        eos_symbol = '</s>'
        with open(path_data_in, 'r', encoding='utf8') as f:
            lines = f.readlines()
        if how == 'corpus':
            self.training_data = [line.replace('\n', " " + eos_symbol) for line in lines]
            self.training_data = [re.split(r"\s+", line) for line in self.training_data]
            for sentence in self.training_data:
                sentence.insert(0, bos_symbol)
                for unigram in sentence:
                    self.unigrams.append(unigram)

                bigrams = [sentence[i:] for i in range(2)]
                bigrams = list(zip(*bigrams))
                bigrams = [' '.join(tup) for tup in bigrams]
                self.bigrams += bigrams

                trigrams = [sentence[i:] for i in range(3)]
                trigrams = list(zip(*trigrams))
                trigrams = [' '.join(tup) for tup in trigrams]
                self.trigrams += trigrams

            unigram_counts = Counter(self.unigrams)
            bigram_counts = Counter(self.bigrams)
            trigram_counts = Counter(self.trigrams)

            self.ngram_probs['unigrams'] = dict(unigram_counts)
            self.ngram_probs['bigrams'] = dict(bigram_counts)
            self.ngram_probs['trigrams'] = dict(trigram_counts)
        elif how == 'ngram_counts':
            self.calculate_probs = True
            for i in lines:
                line_split = re.split(r"\t+", i, maxsplit=1)
                line_split = [text.replace('\n', "") for text in line_split]
                token = line_split[1]
                token_freq = int(line_split[0])
                if len(token.split()) == 1:
                    self.ngram_probs['unigrams'][token] = token_freq
                elif len(token.split()) == 2:
                    self.ngram_probs['bigrams'][token] = token_freq
                else:
                    self.ngram_probs['trigrams'][token] = token_freq
        else:  # FROM Language Model file
            lines_clean = [line.strip() for line in lines if line.strip()]
            lines_clean = [line.replace('\n', "") for line in lines_clean]
            lines_clean = [line for line in lines_clean if not line.startswith('\\')]

            # summary_stats = [line for line in lines_clean if line.startswith('ngram')]
            # summary_stats = [re.split(r"\s+", line, maxsplit=3) for line in summary_stats]
            lines_clean = [line for line in lines_clean if not line.startswith('ngram')]
            lines_split = [re.split(r"\s+", line, maxsplit=3) for line in lines_clean]
            for line in lines_split:
                token_freq = line[0]
                token_prob = line[1]
                token_log_prob = line[2]
                token = line[3]
                ngram_len = len(re.split(r"\s", token))
                if ngram_len == 1:
                    self.unigrams.append(token)
                    self.ngram_probs['unigrams'][token] = []
                    self.ngram_probs['unigrams'][token].append(token_freq)
                    self.ngram_probs['unigrams'][token].append(token_prob)
                    self.ngram_probs['unigrams'][token].append(token_log_prob)
                elif ngram_len == 2:
                    self.bigrams.append(token)
                    self.ngram_probs['bigrams'][token] = []
                    self.ngram_probs['bigrams'][token].append(token_freq)
                    self.ngram_probs['bigrams'][token].append(token_prob)
                    self.ngram_probs['bigrams'][token].append(token_log_prob)
                else:
                    self.trigrams.append(token)
                    self.ngram_probs['trigrams'][token] = []
                    self.ngram_probs['trigrams'][token].append(token_freq)
                    self.ngram_probs['trigrams'][token].append(token_prob)
                    self.ngram_probs['trigrams'][token].append(token_log_prob)

        if self.calculate_probs:
            # Token Size
            self.num_tokens['unigrams'] = sum(self.ngram_probs['unigrams'].values())
            self.num_tokens['bigrams'] = sum(self.ngram_probs['bigrams'].values())
            self.num_tokens['trigrams'] = sum(self.ngram_probs['trigrams'].values())

            # Type Size
            self.num_types['unigrams'] = len(self.ngram_probs['unigrams'])
            self.num_types['bigrams'] = len(self.ngram_probs['bigrams'])
            self.num_types['trigrams'] = len(self.ngram_probs['trigrams'])

            self.ngram_probs['unigrams'] = {
                k: [v, (v / self.num_tokens['unigrams']), math.log(v / self.num_tokens['unigrams'], 10)] for
                k, v in self.ngram_probs['unigrams'].items()}

            self.ngram_probs['bigrams'] = {k: [v] for k, v in self.ngram_probs['bigrams'].items()}
            self.ngram_probs['trigrams'] = {k: [v] for k, v in self.ngram_probs['trigrams'].items()}

            for bigram, count_list in self.ngram_probs['bigrams'].items():
                bigram_split = bigram.split(" ")
                word_1 = bigram_split[0]
                count = count_list[0]
                count_word1 = self.ngram_probs['unigrams'][word_1][0]
                prob_bigram = count / count_word1
                log_prob_bigram = math.log(prob_bigram, 10)
                self.ngram_probs['bigrams'][bigram].append(prob_bigram)
                self.ngram_probs['bigrams'][bigram].append(log_prob_bigram)

            for trigram, count_list in self.ngram_probs['trigrams'].items():
                trigram_split = trigram.split(" ")
                word_1 = trigram_split[0]
                word_2 = trigram_split[1]
                w1w2 = str(word_1) + " " + str(word_2)
                count = count_list[0]
                count_w1w2 = self.ngram_probs['bigrams'][w1w2][0]
                prob_trigram = count / count_w1w2
                log_prob_trigram = math.log(prob_trigram, 10)
                self.ngram_probs['trigrams'][trigram].append(prob_trigram)
                self.ngram_probs['trigrams'][trigram].append(log_prob_trigram)

    def ngrams_count_to_file(self, output_path):
        output_tokens = [self.unigrams, self.bigrams, self.trigrams]
        counts_out = []
        for i in output_tokens:
            count_dict = Counter(i)
            for key, value in count_dict.most_common():
                token_count = str(value) + "\t" + str(key)
                counts_out.append(token_count)

        with open(output_path, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in counts_out)

    def to_arpa_file(self, output_path):
        data_str = '\\data\\'
        token_types = {'unigrams': 1, 'bigrams': 2, 'trigrams': 3}
        output_lines = [data_str]

        for type, display_value in token_types.items():
            freq_summary = "ngram " + str(display_value) + ": type=" + str(self.num_types[type]) + " token=" + str(
                self.num_tokens[type])
            output_lines.append(freq_summary)

        for type, display_value in token_types.items():
            header_str = '\n\\' + str(display_value) + '-grams:'
            output_lines.append(header_str)

            for token, probs in self.ngram_probs[type].items():
                count = probs[0]
                probability = probs[1]
                log_probability = probs[2]
                token_string = str(count) + '\t' + str(probability) + '\t' + str(log_probability) + '\t' + token
                output_lines.append(token_string)
            end_str = '\\end\\'
            output_lines.append(end_str)

        with open(output_path, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in output_lines)

    def calculate_perplexity(self, test_data_path, lambda1, lambda2, lambda3):
        bos_symbol = '<s>'
        eos_symbol = '</s>'

        with open(test_data_path, 'r', encoding='utf8') as f:
            test_corpus = f.readlines()
        test_corpus = [sent.replace('\n', " " + eos_symbol) for sent in test_corpus]
        test_corpus = [re.split(r"\s+", sent) for sent in test_corpus]

        # sets for speed
        unique_unigrams = set(self.unigrams)
        unique_bigrams = set(self.bigrams)
        unique_trigrams = set(self.trigrams)

        prob_sum = 0
        word_num = 0
        oov_num = 0
        sent_num = 0

        for sentence in test_corpus:
            sentence.insert(0, bos_symbol)
            this_sentence = Sentence(sentence, sent_num + 1)
            sent_prob_sub = 0
            sent_oov_num = 0
            sentence_length = len(sentence) - 2
            word_num += sentence_length
            for idx, word in enumerate(sentence[1:]):
                word_1_idx = idx
                word_2_idx = idx - 1
                word_1 = sentence[word_1_idx]
                bigram = word_1 + " " + word

                if word_2_idx >= 0:
                    word_2 = sentence[word_2_idx]
                    trigram = word_2 + ' ' + word_1 + ' ' + word
                    ngram = trigram
                else:
                    trigram = None
                    ngram = bigram

                this_ngram = Ngram(ngram)

                if word in unique_unigrams:
                    prob_unigram = float(self.ngram_probs['unigrams'][word][1])
                    if bigram in unique_bigrams:
                        prob_bigram = float(self.ngram_probs['bigrams'][bigram][1])
                    else:
                        prob_bigram = float(0.0)
                    if trigram is not None and trigram in unique_trigrams:
                        prob_trigram = float(self.ngram_probs['trigrams'][trigram][1])
                    else:
                        prob_trigram = float(0.0)

                    prob_ngram = (lambda1 * prob_unigram) + (lambda2 * prob_bigram) + (lambda3 * prob_trigram)
                    log_prob_ngram = math.log(prob_ngram, 10)

                    this_ngram.lgprob = log_prob_ngram

                    if prob_trigram == 0 or prob_bigram == 0:
                        this_ngram.comments = '(unseen ngrams)'
                    else:
                        this_ngram.comments = None
                    prob_sum += log_prob_ngram
                    sent_prob_sub += log_prob_ngram
                else:
                    this_ngram.lgprob = '-inf'
                    this_ngram.comments = '(unknown word)'
                    sent_oov_num += 1
                    oov_num += 1

                this_ngram.format_output()
                this_sentence.ngrams.append(this_ngram)

            sent_count = sentence_length + 1 - sent_oov_num
            if sent_count > 0:
                sent_total = -sent_prob_sub / sent_count
                sent_ppl = 10 ** sent_total
            else:
                sent_ppl = "-inf"

            this_sentence.num_sents = 1
            this_sentence.num_words = sentence_length
            this_sentence.num_oov = sent_oov_num
            this_sentence.lgprob = sent_prob_sub
            this_sentence.ppl = sent_ppl

            self.test_probs.append(this_sentence)
            sent_num += 1

        count = word_num + sent_num - oov_num
        total = -prob_sum / count
        ppl = 10 ** total

        self.test_summary['lgprob'] = prob_sum
        self.test_summary['ave_lgprob'] = total
        self.test_summary['ppl'] = ppl
        self.test_summary['sent_num'] = sent_num
        self.test_summary['word_num'] = word_num
        self.test_summary['oov_num'] = oov_num
