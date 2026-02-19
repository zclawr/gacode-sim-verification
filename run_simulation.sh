#!/bin/bash
# run_simulation.sh script parameters: 
# $1 : simulation type [tglf, cgyro]
# $2 : path to input folder
. /etc/environment
. $GACODE_ROOT/shared/bin/gacode_setup
# . ./.venv/bin/activate
# function parameters:
# $1 : simulation type [tglf, cgyro]
# $2 : simulation directory (must contain input.tglf or input.cgyro)
function run_simulation {
    if [[ $1 == "tglf" ]]; then
        echo "Beginning TGLF at $2"
        tglf -i $2
        tglf -e $2
        echo "Finished TGLF at $2"
    elif [[ $1 == "cgyro" ]]; then
        echo "Beginning CGYRO at $2"
        cgyro -i $2
        cgyro -e $2 -n 8 -nomp 4
        echo "Finished CGYRO at $2"
    fi
}

for dir in $(find $2 -type d) # expects input directory to contain subdirs for each input
do
  if [[ "$dir" == "$2" ]]; then
    continue # skip parent directory
  fi
  echo "Processing directory: $dir"
  run_simulation $1 $dir # outputs are stored in subdirs per input
done

# Upload results to S3
make upload file=$2