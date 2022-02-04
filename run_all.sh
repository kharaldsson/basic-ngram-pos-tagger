#!/bin/sh

./ppl.sh wsj_sec0_19.lm 0.05 0.15 0.8 ~/dropbox/21-22/570/hw6/examples/wsj_sec22.word ppl_0.05_0.15_0.8  &&
./ppl.sh wsj_sec0_19.lm 0.1 0.1 0.8 ~/dropbox/21-22/570/hw6/examples/wsj_sec22.word ppl_0.1_0.1_0.8  &&
./ppl.sh wsj_sec0_19.lm 0.2 0.3 0.5 ~/dropbox/21-22/570/hw6/examples/wsj_sec22.word ppl_0.2_0.3_0.5  &&
./ppl.sh wsj_sec0_19.lm 0.2 0.5 0.3 ~/dropbox/21-22/570/hw6/examples/wsj_sec22.word ppl_0.2_0.5_0.3  &&
./ppl.sh wsj_sec0_19.lm 0.2 0.7 0.1 ~/dropbox/21-22/570/hw6/examples/wsj_sec22.word ppl_0.2_0.7_0.1  &&
./ppl.sh wsj_sec0_19.lm 0.2 0.8 0 ~/dropbox/21-22/570/hw6/examples/wsj_sec22.word ppl_0.2_0.8_0  &&
./ppl.sh wsj_sec0_19.lm 1.0 0 0 ~/dropbox/21-22/570/hw6/examples/wsj_sec22.word ppl_1.0_0_0