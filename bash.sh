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
--pretrained_vec_file input/dna2vec-20230705-0901-k3to8-100d-10c-32730Mbp-sliding-k0s.w2v \
--temp_dir ./temp --output output/final_interactions_dna2vec.h5 \
--threads 16 --compression 5

python main.py Preprocess -s extract_dataset --input output/balanced_interactions.txt \
--raw_interaction_fasta_file output/temp_balanced/raw_interactions.fasta \
--pretrained_vec_file input/dna2vec-20230705-0901-k3to8-100d-10c-32730Mbp-sliding-k0s.w2v \
--temp_dir ./temp --output output/final_interactions_dna2vec_split.h5 \
--threads 16 --compression 5