import sys
import re
from collections import Counter
import math
import language


def process_file(ngram_count_path, lm_out_path):
    lang_model = language.LanguageModel()
    lang_model.build_model(ngram_count_path, how='ngram_counts')
    lang_model.to_arpa_file(lm_out_path)




if __name__ == "__main__":
    TEST = True
    DATA_TRAIN = "examples/wsj_sec0_19.word"
    NGRAM_COUNT_IN = "wsj_sec0_19.ngram_count"
    LM_OUT = "wsj_sec0_19.lm"

    if TEST:
        process_file(NGRAM_COUNT_IN, LM_OUT)
    else:
        process_file(sys.argv[1], sys.argv[2])
