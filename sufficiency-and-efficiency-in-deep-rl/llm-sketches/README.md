
To create an initial dataset, run `python3 chat_pack.py`. This'll load a blank GPT2 instance and you'll converse with it, generating data. Data will be written to the `data/` directory upon `Ctrl-C`. 

To perform an initial fit to data, again load a GPT2 instance and apply the dataset you've generated. Use interactive Python to achieve this like follows. Notice `mv1` is a prefix, as several files must be created per model. 

```
python3 chat_pack.py
python3 initial_fit.py --data-path data/chat-game-data-1728172436.pt 
python3 chat_pack.py --path models/mv1
python3 add_data.py --load-model-path models/mv1 --save-model-path models/mv2 --load-data-path data/chat-game-data-1728188488.pt --pi-min .3 --pi-max .7
python3 chat_pack.py --path models/mv2 
python3 add_data.py --load-model-path models/mv2 --save-model-path models/mv3 --load-data-path data/chat-game-data-1728267177.pt --pi-min .3 --pi-max .7 
python3 chat_pack.py --path models/mv3
python3 add_data.py --load-model-path models/mv3 --save-model-path models/mv4 --load-data-path data/chat-game-data-1728840605.pt --pi-min .3 --pi-max .7 --p-gpt2-loss .98
python3 chat_pack.py --path models/mv4 
```

