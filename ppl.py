import sys
import re
from collections import Counter
import math
import language
import time


def process_file(lm_path, lamb1, lamb2, lamb3, test_data, ppl_out_path):
    lang_model = language.LanguageModel()
    # startTime = time.time()
    lang_model.build_model(lm_path, how='lm')
    # executionTime = (time.time() - startTime)
    # print('Training time in seconds: ' + str(executionTime))
    # startTime = time.time()
    lamb1fl = float(lamb1)
    lamb2fl = float(lamb2)
    lamb3fl = float(lamb3)

    lang_model.calculate_perplexity(test_data, lamb1fl, lamb2fl, lamb3fl)
    # executionTime = (time.time() - startTime)
    # print('Ppl calc time in seconds: ' + str(executionTime))


    output_lines = []
    for sent in lang_model.test_probs:
        sent_line = "Sent #" + str(sent.number) + ": " + str(sent.text)
        output_lines.append(sent_line)

        for idx, ngram in enumerate(sent.ngrams):
            ngram_line = str(idx + 1) + ': ' + str(ngram.output)
            output_lines.append(ngram_line)

        sent_summary_line = str(sent.num_sents) + " sentence, " + str(sent.num_words) + " words, " + str(
            sent.num_oov) + " OOVs"
        output_lines.append(sent_summary_line)

        sent_evaluation_line = 'lgprob=' + str(sent.lgprob) + " ppl=" + str(sent.ppl)
        output_lines.append(sent_evaluation_line)

        output_lines.append("")

    output_lines.append("\n")
    output_lines.append('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    sent_num = lang_model.test_summary['sent_num']
    word_num = lang_model.test_summary['word_num']
    oov_num = lang_model.test_summary['oov_num']
    lgprob = lang_model.test_summary['lgprob']
    ave_lgprob = lang_model.test_summary['ave_lgprob']
    ppl = lang_model.test_summary['ppl']
    corpus_summary_line = "sent_num=" + str(sent_num) + " word_num=" + str(word_num) + " oov_num=" + str(oov_num)
    output_lines.append(corpus_summary_line)
    corpus_evaluation_line = "lgprob=" + str(lgprob) + " ave_lgprob=" + str(ave_lgprob) + " ppl=" + str(ppl)
    output_lines.append(corpus_evaluation_line)

    with open(ppl_out_path, 'w', encoding='utf8') as f:
        f.writelines("%s\n" % line for line in output_lines)


if __name__ == "__main__":
    TEST = False
    LM_IN = "wsj_sec0_19.lm"
    TEST_DATA = "examples/wsj_sec22.word"
    PPL_OUT = "ppl_0.05_0.15_0.8"

    if TEST:
        process_file(LM_IN, 0.2, 0.7, 0.1, TEST_DATA, PPL_OUT)
    else:
        process_file(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
