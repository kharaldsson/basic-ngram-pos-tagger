import sys
import re
from collections import Counter
import math
import language
import time

def process_file(training_data_path, ngram_count_out_path):
    lang_model = language.LanguageModel()
    # startTime = time.time()
    lang_model.build_model(training_data_path, how='corpus')
    # executionTime = (time.time() - startTime)
    # # print('Execution time in seconds: ' + str(executionTime))
    lang_model.ngrams_count_to_file(ngram_count_out_path)
    # print(lang_model.unigrams[0:20])


if __name__ == "__main__":
    TEST = False
    DATA_TRAIN = "examples/wsj_sec0_19.word"
    # DATA_TRAIN = "examples/test_data_ex"
    NGRAM_COUNT_OUT = "wsj_sec0_19.ngram_count"

    if TEST:
        process_file(DATA_TRAIN, NGRAM_COUNT_OUT)
    else:
        process_file(sys.argv[1], sys.argv[2])
