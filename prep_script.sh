##download Dataset
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
mkdir -p data
unzip LA.zip >> /dev/null
mv LA data/LA
rm LA.zip
