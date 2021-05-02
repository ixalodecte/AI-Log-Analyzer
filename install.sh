echo "installation des packages python..."
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
echo "telechargement du fichier 'word_to_vec'"
cd data
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
cd ..

