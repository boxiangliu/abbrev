cd ../data/BioC/
for f in `find . -name '*.xml'`; do
    out=`dirname $f`/`basename $f .xml`.txt
    preprocess/bioc/xml_to_txt.py < $f > $out
done 