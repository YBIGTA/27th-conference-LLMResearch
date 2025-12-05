#!/bin/bash
echo "Finding and killing all Python processes..."
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9
echo "All Python processes killed."
