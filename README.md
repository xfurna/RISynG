# RISynG

Recursive Integration of Synergised Graph Representations

## Installation

### Clone the repository 

```
$ git clone https://github.com/xfurna/RISynG.git
$ cd RISynG
```
Alternatively, download the package and change your working directory to `RISynG/`
### Install dependencies

```
$ pip3 install pipenv
$ pipenv install
```

### Activate the virtual environment

```
$ pipenv shell
```

## Usage
Add executable permission to the provided `run.sh` script.
```
$ chmod +x run.sh
```
There are five dataset in the `Datasets` folder- OV, STAD, CESC, LGG and BRCA. The algorithm can be executed for these dataset through the provided shell script `run.sh` as illustrated below.

```
$ ./run.sh dataset # replace 'dataset' with the name of the dataset
```
Execution can be done for multiple dataset just by cascading the names one after the other. 
