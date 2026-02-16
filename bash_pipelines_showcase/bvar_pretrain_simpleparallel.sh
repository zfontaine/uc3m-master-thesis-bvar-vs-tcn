#!/bin/bash
#!/usr/bin/env bash
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study
# on Forecasting Accuracy and Uncertainty Calibration"
# Universidad Carlos III de Madrid (UC3M), 2026
#
# FILENAME: bvar_pretrain_simpleparallel.sh
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-07
#
# LICENSE: GNU GPLv3 for code
#          Creative Commons for the Master thesis
#
# NOTE:
# The script allows to run two bvar pipelines in parallel using 
# naive linux parallelisation via "&"
#
# MUST CHANGE DIRECTORY AND MODEL NUMBER FOR EACH RUN!
#===================================================================

# Run first script from chosen directory
(cd 08 && ./pipe-bvar.sh) > log_08.txt 2>&1 &
PID1=$!

# Run second script from chosen directory
(cd 09 && ./pipe-bvar.sh) > log_09.txt 2>&1 &
PID2=$!

echo "Both scripts running in background!"
echo "PIDs: $PID1 (lag 08) and $PID2 (lag 09)"
echo ""
echo "Monitor with:"
echo "  tail -f log_08.txt"
echo "  tail -f log_09.txt"
echo ""
echo "Check progress:"
echo "  ls -1 08/RESULTS/*.csv 2>/dev/null | wc -l"
echo "  ls -1 09/RESULTS/*.csv 2>/dev/null | wc -l"
echo ""
echo "Check if still running:"
echo "  jobs"
echo "  ps -p $PID1 $PID2"

# Wait for both to finish
wait

echo ""
echo "Both completed!"
echo "Results in:"
echo "  08/RESULTS/"
echo "  09/RESULTS/"
