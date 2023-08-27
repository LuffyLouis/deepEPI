#### Preprocess the valid pairs generated from HiC-Pro

python main.py Preprocess --hic_file ./SRR400264_00.allValidPairs.hic \
--input ./SRR400264_00.allValidPairs \
--juicer_tools_path juicer_tools_1.22.01.jar --threads 2 \
--chrom_size_file ./hg38.chrom_size \
--method KR --flag BP --bin_size 500000 --output_dir ./output --output_prefix eigen_ \
--distance 1e6 --within_compartment 2 --promoter -a ./hg38.refGene.gtf \
--temp_dir ./temp --output output/filt_interactions.txt \
--output_raw output/filt_interactions_raw.txt

python main.py Preprocess --hic_file /home/luffy/liuhongfei/deepEPI/SRR400264_00.allValidPairs.hic \
--input /home/luffy/liuhongfei/deepEPI/SRR400264_00.allValidPairs \
--juicer_tools_path juicser_tools_1.22.01.jar --threads 2 \
--chrom_size_file /home/luffy/liuhongfei/deepEPI/hg38.chrom_size \
--method KR --flag BP --bin_size 500000 --output_dir ./output --output_prefix eigen_ \
--distance 1e6 --within_compartment 2 --promoter -a ./hg38.refGene.gtf \
--temp_dir ./temp --output output/filt_interactions.txt \
--output_raw output/filt_interactions_raw.txt

python main.py Preprocess --input /home/hfliu/liuhongfei/deepEPI/SRR400264_00.allValidPairs \
--eigenvec_dir ./ -s generate_interactions --temp_dir ./temp --output output/annotated_interactions.txt \
--hic_file /home/hfliu/liuhongfei/HTS-snakemake/HiC-seq/output/HiC-Pro/hic_results/juicerbox/SRR400264_00/SRR400264_00.allValidPairs.hic \
--juicer_tools_path /home/hfliu/liuhongfei/software/juicer_tools_1.22.01.jar --threads 20 \
--chrom_size_file /home/hfliu/liuhongfei/HTS-snakemake/ChIP-seq/reference_genome/hg38.chrom_size \
--method KR --flag BP --bin_size 500000 --output_dir ./ --output_prefix eigen_


python main.py Preprocess --input /home/hfliu/liuhongfei/duck_eQTL/output/HiC_seq/HiC-Pro/hic_results/data/SRR11910010/SRR11910010.allValidPairs \
--eigenvec_dir ./ -s generate_interactions --temp_dir ./temp --output output/annotated_interactions.txt \
--hic_file /home/hfliu/liuhongfei/HTS-snakemake/HiC-seq/output/HiC-Pro/hic_results/juicerbox/SRR400264_00/SRR400264_00.allValidPairs.hic \
--juicer_tools_path /home/hfliu/liuhongfei/software/juicer_tools_1.22.01.jar --threads 20 \
--chrom_size_file /home/hfliu/liuhongfei/HTS-snakemake/ChIP-seq/reference_genome/hg38.chrom_size \
--method KR --flag BP --bin_size 500000 --output_dir ./ --output_prefix eigen_

python main.py Preprocess -s filtration --input ./annotated_interactions.txt \
--temp_dir ./temp --output output/filt_interactions.txt \
--output_raw output/filt_interactions_raw.txt \
--chrom_size_file ./hg38.chrom_size \
--threads 16 --distance 1e6 --within_compartment 2 --promoter -a ./hg38.refGene.gtf

## balance datasets
python main.py Preprocess -s balance --input output/filt_interactions.txt \
--raw_interactions_file output/filt_interactions_raw.txt \
--reference_genome ./hg38.fa \
--temp_dir ./temp --output output/balanced_interactions.txt \
--threads 16

## extract_dataset
python main.py Preprocess -s extract_dataset --input output/balanced_interactions.txt \
--raw_interaction_fasta_file output/temp_balanced/raw_interactions.fasta \
--encode_method onehot \
--concat_epi \
--temp_dir ./temp --output output/final_interactions_onehot_.h5 \
--threads 16 --compression 5

# generate enhancer-promoter with unequal length 
python main.py Preprocess -s extract_dataset --input output/balanced_interactions.txt \
--raw_interaction_fasta_file output/temp_balanced/raw_interactions.fasta \
--encode_method onehot \
--concat_epi \
--temp_dir ./temp --output output/final_interactions_onehot_4kb_2kb.h5 \
--threads 8 --compression 5
# python main.py Preprocess -s extract_dataset --input output/balanced_interactions.txt \
# --raw_interaction_fasta_file output/temp_balanced/raw_interactions.fasta \
# --encode_method onehot \
# --temp_dir ./temp --output output/final_interactions_onehot_split.h5 \
# --threads 16 --compression 5

python main.py Preprocess -s extract_dataset --input output/balanced_interactions.txt \
--raw_interaction_fasta_file output/temp_balanced/raw_interactions.fasta \
--pretrained_vec_file input/dna2vec-20230705-0901-k3to8-100d-10c-32730Mbp-sliding-k0s.w2v \
--concat_epi \
--temp_dir ./temp --output output/final_interactions_dna2vec.h5 \
--threads 16 --compression 5

python main.py Preprocess -s extract_dataset --input output/balanced_interactions.txt \
--raw_interaction_fasta_file output/temp_balanced/raw_interactions.fasta \
--pretrained_vec_file input/dna2vec-20230705-0901-k3to8-100d-10c-32730Mbp-sliding-k0s.w2v \
--encode_method dna2vec \
--concat_epi \
--temp_dir ./temp --output output/final_interactions_dna2vec_4kb_2kb.h5 \
--threads 8 --compression 5

# python main.py Preprocess -s extract_dataset --input output/balanced_interactions.txt \
# --raw_interaction_fasta_file output/temp_balanced/raw_interactions.fasta \
# --pretrained_vec_file input/dna2vec-20230705-0901-k3to8-100d-10c-32730Mbp-sliding-k0s.w2v \
# --temp_dir ./temp --output output/final_interactions_dna2vec_split.h5 \
# --threads 16 --compression 5
###
python main.py Datasets --input output/final_interactions_dna2vec.h5 \
--key raw \
--output_prefix output/prep_datasets_dna2vec \
--compression 5

python main.py Datasets --input output/final_interactions_dna2vec_split.h5 \
--key raw \
--output_prefix output/prep_datasets \
--compression 5

python main.py Datasets --input output/final_interactions_onehot_4kb_2kb.h5 \
--key raw \
--output_prefix output/prep_datasets_onehot_4kb_2kb \
--compression 5

python main.py Datasets --input output/final_interactions_dna2vec_4kb_2kb.h5 \
--key raw \
--output_prefix output/prep_datasets_dna2vec_4kb_2kb \
--compression 5

## Training model with simpleCNN
python main.py Train --input output/prep_datasets_onehot_train.h5 \
--model simpleCNN \
--device cuda --epochs 10 --batch_size 32

## Training model with simpleCNN to search super parameters
python main.py Train --input output/prep_datasets_onehot_train.h5 \
--model simpleCNN \
--device cuda --epochs 10 --batch_size 32 --k_fold 10 --is_param_optim --rerun

## Parameters optimization
python main.py Train --input output/prep_datasets_onehot_train.h5 \
--test_input output/prep_datasets_onehot_test.h5 \
--model simpleCNN --device cuda --epochs 10 --batch_size 32 --k_fold 10 \
--is_param_optim --param_optim_strategy grid --rerun --save_fig ROC_10fold 

## Parameters optimization
python main.py Train \
--input output/prep_datasets_onehot_train.h5 --test_input output/prep_datasets_onehot_test.h5 \
--model simpleCNN --device cuda --epochs 10 --batch_size 32 \
--optimizer Adam \
--k_fold 10 --is_param_optim --param_optim_strategy grid --rerun --save_fig ROC_10fold_adam

python main.py Train \
--input output/prep_datasets_onehot_train.h5 --test_input output/prep_datasets_onehot_test.h5 \
--model simpleCNN --device cuda --epochs 10 --batch_size 32 \
--optimizer SGD \
--k_fold 10 --is_param_optim --param_optim_strategy grid --rerun --save_fig ROC_10fold_SGD

# optimization with bayes
python main.py Train \
--input output/prep_datasets_onehot_train.h5 --test_input output/prep_datasets_onehot_test.h5 \
--model simpleCNN --device cuda --epochs 10 --batch_size 32 \
--optimizer SGD --workers 2 \
--k_fold 10 --is_param_optim --param_optim_strategy bayes --rerun --save_fig ROC_10fold_SGD_bayes
# {'target': 0.7616666666666667, 'params': {'batch_size': 53.891976203938626, 'learning_rate': 0.32029159502896104}}

## Training model with simpleCNN + dna2vec 
python main.py Train --input output/prep_datasets_dna2vec_train.h5 \
--test_input output/prep_datasets_dna2vec_test.h5 \
--model simpleCNN --encode_method dna2vec \
--device cuda --epochs 10 --batch_size 64 --rerun
##
python main.py Train --input output/prep_datasets_dna2vec_train.h5 \
--test_input output/prep_datasets_dna2vec_test.h5 \
--model simpleCNN --encode_method dna2vec \
--device cuda --epochs 10 --batch_size 32 --rerun \
--workers 0

## Optimze parameters for simpleCNN + dna2vec
python main.py Train \
--input output/prep_datasets_dna2vec_train.h5 --test_input output/prep_datasets_dna2vec_test.h5 \
--model simpleCNN --encode_method dna2vec \
--device cuda --epochs 10 --batch_size 32 \
--optimizer SGD --workers 2 \
--k_fold 10 --is_param_optim --param_optim_strategy grid --rerun --save_fig ROC_10fold_dna2vec_SGD

for ((i=1;i<4;i++))
do
echo $i
python main.py Train \
--input output/prep_datasets_dna2vec_train.h5 --test_input output/prep_datasets_dna2vec_test.h5 \
--model simpleCNN --encode_method dna2vec \
--device gpu --epochs 10 --batch_size 32 \
--optimizer Adam --workers 2 \
--k_fold 10 --is_param_optim --param_optim_strategy grid --rerun --save_fig ROC_10fold_dna2vec_Adam_${i}
done
##
python main.py Train \
--input output/prep_datasets_dna2vec_4kb_2kb_train.h5 --test_input output/prep_datasets_dna2vec_4kb_2kb_test.h5 \
--model simpleCNN --encode_method dna2vec \
--device cuda --epochs 10 --batch_size 32 \
--optimizer Adam --workers 2 \
--k_fold 10 --rerun --save_fig ROC_10fold_dna2vec_SGD_bayes 
#### Training model with simpleCNN + dna2vec + Adam + bayes
for ((i=1;i<4;i++))
do
echo $i
python main.py Train \
--input output/prep_datasets_dna2vec_4kb_2kb_train.h5 --test_input output/prep_datasets_dna2vec_4kb_2kb_test.h5 \
--model simpleCNN --encode_method dna2vec \
--device cuda --epochs 10 --batch_size 32 \
--optimizer Adam --workers 2 \
--params_config config_bayes.txt \
--k_fold 10 --is_param_optim --param_optim_strategy bayes --rerun --save_fig ROC_10fold_dna2vec_Adam_bayes_${i} >log/log_dna2vec_Adam_bayes_${i}.log
done
#### Training model with simpleCNN + dna2vec + SGD + bayes
for ((i=1;i<4;i++))
do
echo $i
python main.py Train \
--input output/prep_datasets_dna2vec_4kb_2kb_train.h5 --test_input output/prep_datasets_dna2vec_4kb_2kb_test.h5 \
--model simpleCNN --encode_method dna2vec \
--device cuda --epochs 10 --batch_size 32 \
--optimizer SGD --workers 2 \
--params_config config_bayes.txt \
--k_fold 10 --is_param_optim --param_optim_strategy bayes --rerun --save_fig ROC_10fold_dna2vec_SGD_bayes_${i}  >log/log_dna2vec_SGD_bayes_${i}.log
done

## Training model with EPI-Mind + dna2vec 
python main.py Train \
--input output/prep_datasets_dna2vec_4kb_2kb_train.h5 --test_input output/prep_datasets_dna2vec_4kb_2kb_test.h5 \
--model EPIMind --encode_method dna2vec \
--optimizer Adam \
--workers 2 \
--heads 8 --num_layers 4 --num_hiddens 72 --ffn_num_hiddens 256 \
--device cuda --epochs 10 --batch_size 64 --rerun

python main.py Train \
--input output/prep_datasets_dna2vec_4kb_2kb_train.h5 --test_input output/prep_datasets_dna2vec_4kb_2kb_test.h5 \
--model EPIMind --encode_method dna2vec \
--optimizer Adam \
--workers 2 \
--heads 8 --num_layers 4 --num_hiddens 72 --ffn_num_hiddens 256 \
--device gpu --epochs 10 --batch_size 64 --rerun --verbose

##
for ((i=1;i<4;i++))
do
echo $i
python main.py Train \
--input output/prep_datasets_dna2vec_4kb_2kb_train.h5 --test_input output/prep_datasets_dna2vec_4kb_2kb_test.h5 \
--model EPIMind --encode_method dna2vec \
--optimizer Adam --k_fold 10 --is_param_optim --param_optim_strategy bayes \
--workers 2 \
--params_config config_bayes.txt \
--heads 8 --num_layers 4 --num_hiddens 72 --ffn_num_hiddens 256 \
--device cuda --epochs 10 --batch_size 64 --rerun --verbose --save_fig ROC_10fold_EPIMind_dna2vec_SGD_bayes_${i} >log/log_EPIMind_dna2vec_SGD_bayes_${i}.log
done

################################
## Preprocess
java -jar /home/hfliu/liuhongfei/software/juicer_tools_1.22.01.jar eigenvector \
-p KR SRR11910010.allValidPairs.hic NC_051772.1 BP 10000 sd

python main.py Preprocess --hic_file /home/hfliu/liuhongfei/duck_eQTL/output/HiC_seq/HiC-Pro/hic_results/juicerbox/SRR11910010/SRR11910010.allValidPairs.hic \
--input /home/hfliu/liuhongfei/duck_eQTL/output/HiC_seq/HiC-Pro/hic_results/data/SRR11910010/SRR11910010.allValidPairs \
--juicer_tools_path /home/hfliu/liuhongfei/software/juicer_tools_1.22.01.jar --threads 2 \
--chrom_size_file /home/hfliu/liuhongfei/duck_eQTL/reference/ZJU.v1.0.chrom_size \
--method SCALE --flag BP --bin_size 10000 --output_dir ./output_duck --output_prefix eigen_ \
--distance 1e6 --within_compartment 2 --promoter -a /home/hfliu/liuhongfei/duck_eQTL/reference/GCF_015476345.1_ZJU1.0_genomic.gtf \
--temp_dir ./temp_duck --output output_duck/filt_interactions.txt \
--output_raw output_duck/filt_interactions_raw.txt