#!/bin/bash

# this script is used to build and run the flight computer software for the SIL

# build the software

cd ../2024_C_AV_RPI

cmake .
make clean
make

#run the server with info

cd ../PythonTest

python3 FeederServer.py &

echo "Server is running"

### Need to add the command to run the flight computer software
# the issue is that we need the board or to have a mock of the signals that the board would send




