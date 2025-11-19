#!/bin/bash
# Monitor all running jobs on VSC

echo "========================================="
echo "VSC Job Monitoring - EPIC-KITCHENS"
echo "========================================="
echo ""

echo "Current Job Queue:"
echo "-----------------"
ssh vsc "squeue -u vsc38064 --clusters=wice -o '%.10i %.12P %.25j %.8u %.2t %.10M %.6D %R'"

echo ""
echo "========================================="
echo "Validation Job (65466221):"
echo "========================================="
ssh vsc "tail -50 /data/leuven/380/vsc38064/epic_kitchens/validation_output_65466221.txt 2>/dev/null || echo 'Not started yet or no output'"

echo ""
echo "========================================="
echo "Recent Errors (if any):"
echo "========================================="
ssh vsc "tail -20 /data/leuven/380/vsc38064/epic_kitchens/validation_error_65466221.txt 2>/dev/null || echo 'No errors yet'"

echo ""
echo "========================================="
echo "To check results when complete:"
echo "ssh vsc 'cat /data/leuven/380/vsc38064/epic_kitchens/validation_results.json'"
echo "========================================="
