#!/bin/bash

./jobs/summarization/eval/primer.sh out/T_bart_8_256_128.params out/T_bart_8_256_128.pt E_bart_8_256_128
./jobs/summarization/eval/primer.sh out/T_t5_8_256_128.params out/T_t5_8_256_128.pt E_t5_8_256_128
./jobs/summarization/eval/primer.sh out/T_pegasus_8_256_128.params out/T_pegasus_8_256_128.pt E_pegasus_8_256_128

./jobs/summarization/eval/primer.sh out/T_bart_8_512_128.params out/T_bart_8_512_128.pt E_bart_8_512_128
./jobs/summarization/eval/primer.sh out/T_t5_8_512_128.params out/T_t5_8_512_128.pt E_t5_8_512_128
