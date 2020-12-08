#!/bin/sh

PATH=$PATH:/mnt/big/kwc/pubmed/pubtator5/ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/titles_and_abstracts/:/mnt/home/kwc/suffix

untagged=/mnt/big/kwc/pubmed/pubtator5/ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/titles_and_abstracts
cd /mnt/big/kwc/pubmed/Boxiang/abbrev/processed_data/preprocess/bioc/propose_qa_on_bioc

cat *fasta | awk -F'|' '$1 ~ /^  / {sf=substr($1,3); lf=$2; print lf " (" sf}' | sort -u |
    freqs.sh $untagged |
    awk -F'\t' '{x[$2] += $1}; END {for(i in x) print x[i] "\t" i}' > suffix_freqs.txt