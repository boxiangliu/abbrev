MED1250_labeled="/mnt/scratch/boxiang/projects/abbrev/data/MED1250/MED1250_labeled"

# How many short-form-long-form pairs were extracted? 
grep "^  " $MED1250_labeled | wc -l
# 1221


# How many paragraphs were in the dataset? 
grep -e "^$" -e "^  " $MED1250_labeled
# 1258