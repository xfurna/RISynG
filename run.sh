#!/bin/bash
mkdir ../labels
mkdir ../Gmat
mkdir ../intWA

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    # VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            python3 risyng -h
            exit
            ;;
        OV)
            
            echo "OV"
            python3 risyng -d OV -m 1 -b 1
            python3 risyng -d OV -m 2 -b 1
            python3 risyng --fuse y -d OV -o 12
            echo "================================="
            echo "================================="
            ;;
        STAD)
            
            echo "STAD"
            python3 risyng -d STAD -m 1 -b 1
            python3 risyng -d STAD -m 2 -b 0.9
            python3 risyng --fuse y -d STAD -o 12

            echo "================================="
            echo "================================="
            ;;
        LGG)
            
            echo "LGG"
            python3 risyng -d LGG -m 1 -b 0.6
            python3 risyng -d LGG -m 3 -b 0.9
            python3 risyng -d LGG -m 4 -b 0.8
            python3 risyng -d LGG -m 2 -b 0.8
            python3 risyng --fuse y -d LGG -o 1342

            echo "================================="
            echo "================================="
            ;;
        BRCA)
            
            echo "BRCA"
            python3 risyng -d BRCA -m 3 -b 1
            python3 risyng -d BRCA -m 2 -b 0.7
            python3 risyng -d BRCA -m 1 -b 1
            python3 risyng -d BRCA -m 4 -b 0.6
            python3 risyng --fuse y -d BRCA -o 3214

            echo "================================="
            echo "================================="
            ;;
        CESC)
                            
            echo "CESC"
            python3 risyng -d CESC -m 3 -b 0.8
            python3 risyng -d CESC -m 1 -b 0.3
            python3 risyng -d CESC -m 2 -b 0.9
            python3 risyng -d CESC -m 4 -b 1
            python3 risyng --fuse y -d CESC -o 3124
            echo "================================="
            echo "================================="
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done
