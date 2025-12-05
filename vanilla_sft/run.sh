SESSION_NAME="session"
CMD1="python train_eval.py --config=configs/arc_challenge.json --seed=10"
CMD2="python train_eval.py --config=configs/cqa.json --seed=10"
CMD3="python train_eval.py --config=configs/gsm8k.json --seed=10"

tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "$CMD1 && $CMD2 && $CMD3" C-m
tmux attach-session -t $SESSION_NAME
