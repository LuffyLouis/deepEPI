#### Training model with simpleCNN + dna2vec + SGD + bayes
python main.py Train \
--input output/prep_datasets_dna2vec_4kb_2kb_train.h5 --test_input output/prep_datasets_dna2vec_4kb_2kb_test.h5 \
--model simpleCNN --encode_method dna2vec \
--device gpu --epochs 10 --batch_size 32 \
--optimizer Adam --workers 2 \
--k_fold 10 --rerun --save_fig ROC_10fold_dna2vec_SGD_bayes 