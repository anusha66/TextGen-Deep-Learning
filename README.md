# TextGen-Deep-Learning
Generation of Text from Structured Data using Generative Models


# Commands for generating BLEU
    cd TextGen-Deep-Learning
    ./scripts/multi-bleu.pl data/gold/test.txt < data/generated_text/lin_test.txt

# Commands for evaluation
    python3 data_utils.py -mode make_ie_data -input_path "../../data/rotowire" -output_fi "roto-ie.h5"

    python3 data_utils.py -mode prep_gen_data -gen_fi ../../data/generated_text/rnn_test.txt -dict_pfx "roto-ie" -output_fi rnn_gen.h5 -input_path "../../data/rotowire" -test

    luajit extractor.lua -gpuid 1 -datafile roto-ie.h5 -preddata rnn_gen.h5 -dict_pfx "roto-ie" -just_eval

    python non_rg_metrics.py roto-gold-test.h5-tuples.txt rnn_gen.h5-tuples.txt -test


    

