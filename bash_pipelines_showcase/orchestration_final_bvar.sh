#!/bin/bash
#!/usr/bin/env bash
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study
# on Forecasting Accuracy and Uncertainty Calibration"
# Universidad Carlos III de Madrid (UC3M), 2026
#
# FILENAME: orchestration_final_bvar.sh
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-07
#
# LICENSE: GNU GPLv3 for code
#          Creative Commons for the Master thesis
#
#
# NOTE: Run FINAL BVAR MODEL on all excitement levels in parallel

#===============================================================

echo "------------------------------------------"
echo "BVAR FINAL MODEL FORECASTING RUN FOR LOG 5"
echo "------------------------------------------"
echo "Begin run for excitement 0.25 and 0.50"

(cd excitement_0.25 && ./pipe-bvar.sh) > log_excitement0p25.txt 2>&1 &
PID1=$!

(cd excitement_0.50 && ./pipe-bvar.sh) > log_excitement0p50.txt 2>&1 &
PID2=$!

echo "PIDs: $PID1 (0.25) and $PID2 (0.50)"

wait

echo "Begin run for 0.75"

(cd excitement_0.75 && ./pipe-bvar.sh) > log_excitement0p75.txt 2>&1 &
PID3=$!

echo "PID: $PID3 (0.75)"

wait

echo ""
echo ""
echo "DONE!"
echo "BVAR trained sucessfully!"

