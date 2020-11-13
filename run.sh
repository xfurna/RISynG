#!/bin/bash
mkdir labels
mkdir Gmat
mkdir intWA
echo "OV"
python3 generate_G.py -d OV -m 1 -b 1
python3 generate_G.py -d OV -m 2 -b 1
python3 integrate.py -d OV -o 12

echo "================================="
echo "================================="
echo "STAD"
python3 generate_G.py -d STAD -m 1 -b 1
python3 generate_G.py -d STAD -m 2 -b 0.9
python3 integrate.py -d STAD -o 12

echo "================================="
echo "================================="
echo "LGG"
python3 generate_G.py -d LGG -m 1 -b 0.6
python3 generate_G.py -d LGG -m 3 -b 0.9
python3 generate_G.py -d LGG -m 4 -b 0.8
python3 generate_G.py -d LGG -m 2 -b 0.8
python3 integrate.py -d LGG -o 1342

echo "================================="
echo "================================="
echo "BRCA"
python3 generate_G.py -d BRCA -m 3 -b 1
python3 generate_G.py -d BRCA -m 2 -b 0.7
python3 generate_G.py -d BRCA -m 1 -b 1
python3 generate_G.py -d BRCA -m 4 -b 0.6
python3 integrate.py -d BRCA -o 3214

echo "================================="
echo "================================="
echo "CESC"
python3 generate_G.py -d CESC -m 3 -b 0.8
python3 generate_G.py -d CESC -m 1 -b 0.3
python3 generate_G.py -d CESC -m 2 -b 0.9
python3 generate_G.py -d CESC -m 4 -b 1
python3 integrate.py -d CESC -o 3124