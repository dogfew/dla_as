pip install -r requirements.txt
pip install gdown>4.7
mkdir -p default_test_model
cd default_test_model
gdown 1UBJKIAn_ibl9UH7WmKzYpEphFkdz6V0z -O lcnn-lfcc-10.pth
gdown 191cPrwnaGGu6Vz8-fEjr3U8NO3wkbSCt -O rawnet2-s3-50.pth
gdown 1acqZie5JlQuJr7axjwzbC2Q8o-yRgYED -O rawnet2-s1.50.pth
cd ..
