#!/bin/bash

. path.sh
counter=1
while [ $counter -le 10 ]
do
python3 ./scripts/dicewars-ai-only.py -d --logdir logs -r --ai gf.xgeffe00 dt.wpm_c dt.stei dt.ste
((counter++))
done
counter=1
while [ $counter -le 10 ]
do
python3 ./scripts/dicewars-ai-only.py -d --logdir logs -r --ai gf.xgeffe00 dt.wpm_c dt.stei dt.wpm_d
((counter++))
done
counter=1
while [ $counter -le 10 ]
do
python3 ./scripts/dicewars-ai-only.py -d --logdir logs -r --ai gf.xgeffe00 dt.wpm_c dt.wpm_s dt.wpm_d
((counter++))
done
counter=1
while [ $counter -le 10 ]
do
python3 ./scripts/dicewars-ai-only.py -d --logdir logs -r --ai gf.xgeffe00 dt.wpm_c dt.ste dt.sdc
((counter++))
done
counter=1
while [ $counter -le 10 ]
do
python3 ./scripts/dicewars-ai-only.py -d --logdir logs -r --ai gf.xgeffe00 dt.wpm_c dt.stei dt.wpm_d
((counter++))
done