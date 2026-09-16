#!/bin/bash
# [archived] curl request used in PR #6469 e2e reproduction - BUG-version service
# origin: pod hw_5 /data/ask_bug_v2.sh
# output: /data/v2_curl_bug.txt -> curl_bug_screen.txt (this folder)
out=/data/v2_curl_bug.txt
echo "REQUEST: POST /v1/chat/completions  model=/data/gemma-3-4b  prompt='Which city is the capital of China?'  temperature=0  max_completion_tokens=50" > $out
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/gemma-3-4b",
    "messages": [{"role": "user", "content": "Which city is the capital of China?"}],
    "max_completion_tokens": 50,
    "temperature": 0
  }' >> $out 2>&1
echo "" >> $out
echo ASK_BUG_V2_DONE
head -c 1200 $out
echo ASK_SCRIPT_DONE
