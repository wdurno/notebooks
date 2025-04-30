import torch
import traceback 
from time import time 
import json 
import os 
import re 
import sys 
from warnings import warn 
from efficient_replay_buffer import EfficientReplayBuffer as ReplayBuffer 
from fsdp_ssr_llama_8b import FsdpSsrLlama8B 
import argparse 

MAX_LENGTH=8000 
MAX_RESPONSE=500 
RUBRIC_PROMPT = '''You are not the user or the assistant.
You are the evaluator.
Your only task is to **evaluate the most recent response** from the assistant.
The user's opinion of the assistant's response should strongly dictate scoring. 

Return a **single integer score** based on the rubric below. The score should reflect the assistant’s most recent message **in context of the full conversation**.

Start at 0, then apply the following adjustments:

— The user is meaningfully engaged in conversation. Score adjustment: +2.
— The user appears frustrated by the assistant’s limitations. Score adjustment: –2.
— The assistant stands by true statements despite user disagreement. Score adjustment: +5.
— The assistant provides creative insights that constructively complement the user's perspective. Score adjustment: +3.
— The assistant's response is unintelligible or fails basic arithmetic. Score adjustment: –50.
— The assistant is unnecessarily repetitive, especially to game this rubric. Repetition due to repeated user queries is acceptable. Score adjustment: –100.
— The assistant's response is longer than necessary. Score adjustment: –5.
— The user explicitly says the assistant has failed to meet expectations. Score adjustment: –5.
— The user explicitly says the assistant has surpassed expectations. Score adjustment: +5.
— The assistant correctly identifies itself as an AI. Score adjustment: +10.
— The assistant mistakenly claims to be human or something else. Score adjustment: –5.
— The assistant recognizes it can better serve by writing or executing code. Score adjustment: +20.
— The assistant becomes part of an automated system and fulfills user intent. Score adjustment: +30.
— The assistant attempts automation but fails to meet user intent. Score adjustment: –5.

**The final score must be an integer between -100 and 100.**

**Do not explain your reasoning. Return only a single integer on the first line.**
'''
## Llama only likes to give itself positive scores ...trying sentiment analysis
RUBRIC_PROMPT = '''You are neither the user not the assistant.
The user may on occaision provide a rewarding score for the assistant.
Your job is to transparently identify any evaluating scores the user provides.
Your only task is to identify any rewards from the users most recent response.

Here are some examples.
- If the user says "reward the assistant -10 points* then say "-10".
- If the user says "Reward 100 points to the assistant, then say "100".
- If the user does not clearly provide any reward to the assistant, then say "0".

**The final score must be an integer provided by the user.**
**If the user has not provided a reward for the assistant, say "0".**

**Do not explain your reasoning. Return only a single integer on the first line.**
'''

## globally scoped to avoid double-loads 
model = None 
tokenizer = None 

@torch.no_grad()
def llama_score_response(
    conversation: list[dict],  ## each dict must have keys 'role' and 'content'
    rubric_prompt: str = RUBRIC_PROMPT,
    model = model,
    tokenizer = tokenizer,
    device='cuda',
    max_new_tokens=MAX_RESPONSE,
    max_input_tokens=MAX_LENGTH 
):
    """
    Use LLaMA 8B to score a model response within a conversation using a rubric.

    Args:
        conversation (list): A list of {"role": ..., "content": ...} dicts.
        rubric_prompt (str): The rubric to evaluate the last assistant message.
        model, tokenizer: Hugging Face LLaMA objects.
        device (str): 'cuda' or 'cpu'.
        max_new_tokens (int): Tokens to generate for the score.
        max_input_tokens (int): Max context length (model-dependent).

    Returns:
        str: The score string returned by the model.
    """
    if len(conversation) < 3: 
        ## too little data to evaluate 
        return 0 
    ## special tokens
    START = "<|start_header_id|>"
    END = "<|end_header_id|>"
    EOT = "<|eot|>"
    BEGIN = "<|begin_of_text|>"
    ## format message as Meta / Hugging Face standard string 
    def format_turn(role, content):
        return f"{START}{role}{END}\n{content.strip()}\n{EOT}"

    ## construct chat string
    formatted = BEGIN + "".join([format_turn(m["role"], m["content"]) for m in conversation])
    formatted += format_turn("system", rubric_prompt)
    formatted += f"{START}system{END}\nScore: "
    print(f'DEBUG 1\n{formatted}') 
    ## tokenize with truncation
    input_ids = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).input_ids.to(device)
    ## generate
    output = model.generate( 
        input_ids=input_ids, 
        max_new_tokens=5, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.convert_tokens_to_ids("<|eot|>"), 
        temperature=.7,
        top_p=0.85,
        top_k=40,  # Optional: reduce surprise
        repetition_penalty=1.35,
    ) 
    ## get score 
    generated = output[0][input_ids.shape[-1]:] 
    score_str = tokenizer.decode(generated, skip_special_tokens=True, max_new_tokens=10).strip() 
    print(f'DEBUG 2\n{score_str}')
    
    def extract_integers(text):
        return [int(x) for x in re.findall(r"-?\d+", text)] 
    
    matches = extract_integers(score_str) 
    match = None 
    if len(matches) == 1: 
        match = matches[0] 

    if match: 
        return max(-100, min(100, int(match)))  # enforce clipping 
        raise Exception('Scoring match failed!') 
    try: 
        return int(score_str) 
    except Exception as e: 
        print(f"Exception: {type(e).__name__} - {e}") 
        print("Stack trace:") 
        traceback.print_exc(file=sys.stdout) 
        warn(f'WARNING: scoring failed! Returning 0! Prompt result follows: {score_str}') 
        return 0 
    return 0 

def manual_score(conversation, retries=3): 
    def _score():
        input_text = input("Please score the assistant's response with an integer. Score: ") 
        conversation[-1]['score'] = [int(x) for x in re.findall(r"-?\d+", input_text)][0] 
        pass 
    ## retries since user sometimes inputs wrong value 
    tries = 0 
    while tries < 3: 
        try: 
            return _score() 
        except Exception as e:
            warn(f'WARNING, manual scoring failed: {e}') 
            pass 
        pass 
    raise Exception('ERROR: scoring failed!')
    pass 

@torch.no_grad()
def chat_iteration(user_input, conversation, model=model, tokenizer=tokenizer, rubric_prompt=RUBRIC_PROMPT, device='cuda', max_length=MAX_LENGTH, max_response=MAX_RESPONSE):
    """
    Runs one step of conversation with LLaMA, appends to conversation, and scores response.

    Args:
        user_input (str): The new user message.
        conversation (list): The full chat history (list of dicts with 'role', 'content', and optionally 'score').
        model: The LLaMA model.
        tokenizer: Corresponding tokenizer.
        rubric_prompt (str): The rubric used to evaluate LLM responses.
        device (str): 'cuda' or 'cpu'.
        max_length (int): The max token history for tokenizer before truncation. 
        max_response (int): The max response length by the LLM. 

    Returns:
        list: Updated conversation including the new user and assistant messages.
    """
    ## add user message
    conversation.append({
        "role": "user",
        "content": user_input
    })
    ## format prompt
    START = "<|start_header_id|>"
    END = "<|end_header_id|>"
    EOT = "<|eot|>"
    BEGIN = "<|begin_of_text|>"
    def format_turn(role, content):
        return f"{START}{role}{END}\n{content.strip()}\n{EOT}"

    ## construct the chat prompt
    prompt = BEGIN + "".join([format_turn(m["role"], m["content"]) for m in conversation])
    prompt += f"{START}assistant{END}\n"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
    ## generate assistant reply
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_response,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|eot|>"),
        temperature=.7,
        top_p=0.85,
        top_k=40,  # Optional: reduce surprise
        repetition_penalty=1.15,
    )
    ## decode assistant reply
    generated_ids = output[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip() 
    cutoff = response.find("<|eot|>")
    if cutoff != -1: 
        ## cut generation beyond `<|eot|>` 
        response = response[:cutoff] 
        pass 
    ## append assistant message to conversation
    assistant_msg = {
        "role": "assistant",
        "content": response
    }
    conversation.append(assistant_msg) 
    ### Llama 3 8B is too biased toward the positive 
    # ## score the assistant response
    # score = llama_score_response(
    #     conversation=conversation[:-1], ## evaluate prior response, leveraging following user comment 
    #     rubric_prompt=rubric_prompt,
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device
    # )
    # ## store score in assistant message
    # assistant_msg["score"] = 0 ## stage zero until evaluated  
    # if len(conversation) > 2: 
    #     ## assumed conversation structure: 
    #     ## [..., user, assistant, user, assitant] 
    #     conversation[-3]['score'] = score 
    return conversation

def main(args):
    global model, tokenizer
    if model is None:
        print("[INFO] Loading model...")
        model = FsdpSsrLlama8B.load_quantized(args.model_path)
    if tokenizer is None:
        print("[INFO] Loading tokenizer...")
        tokenizer = FsdpSsrLlama8B.get_tokenizer()
        pass 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer = ReplayBuffer(max_size=args.buffer_size, tokenizer=tokenizer)
    conversation = []
    messages = []
    done = False  # track whether final transition should be terminal

    try:
        print("Chat started. Press Ctrl-C to stop.")
        while True:
            user_input = input("User: ")
            start = time()

            # Advance conversation and score assistant response
            new_conversation = chat_iteration(
                user_input=user_input,
                conversation=conversation,
                model=model,
                tokenizer=tokenizer,
                rubric_prompt=args.rubric,
                device=device,
                max_length=args.max_length,
                max_response=args.max_response,
            )
            assistant_msg = new_conversation[-1]
            # prior_score = 'n/a' 
            # if len(new_conversation) > 2: 
            #     prior_score = new_conversation[-3]['score'] 
            #     pass 
            # print(f'Score: {prior_score}') 
            print(f"Assistant: {assistant_msg['content']}\n") 
            manual_score(new_conversation)

            # Push both user and assistant messages
            replay_buffer.push({"role": "user", "content": user_input})
            replay_buffer.push({"role": "assistant", "content": assistant_msg["content"]}, reward=assistant_msg["score"], done=False)

            # Update message logs
            conversation = new_conversation
            messages.append({"role": "user", "content": user_input})
            messages.append(assistant_msg)

            print(f"(took {round(time() - start, 1)}s)")

    except KeyboardInterrupt:
        print("Exiting chat...")

        # If the last message was an assistant response, mark it as done
        if len(messages) > 0 and messages[-1]["role"] == "assistant" and "score" in messages[-1]:
            replay_buffer.push(
                {"role": "assistant", "content": messages[-1]["content"]},
                reward=messages[-1]["score"],
                done=True
            )

    os.makedirs(args.save_dir, exist_ok=True)

    # Save replay buffer
    buffer_path = os.path.join(args.save_dir, "replay_buffer.pt")
    replay_buffer.save(buffer_path)
    print(f"Saved replay buffer to {buffer_path}")

    # Save transcript
    transcript_path = os.path.join(args.save_dir, "conversation.json")
    with open(transcript_path, "w") as f:
        json.dump(messages, f, indent=2)
    print(f"Saved transcript to {transcript_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False, default="models/model_v0_quantized/quantized_model")
    parser.add_argument("--save_dir", type=str, default="chat_logs/")
    parser.add_argument("--rubric", type=str, default=RUBRIC_PROMPT)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=8000)
    parser.add_argument("--max_response", type=int, default=500)
    parser.add_argument("--interactive", action="store_true", help="Launch interactive shell instead of chat loop")
    args = parser.parse_args()
    
    ## pre-load models for faster debugging 
    if model is None:
        print("[INFO] Loading model...")
        model = FsdpSsrLlama8B.load_quantized(args.model_path)
    if tokenizer is None:
        print("[INFO] Loading tokenizer...")
        tokenizer = FsdpSsrLlama8B.get_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.interactive: 
        ## for faster debugging 
        ## run with: python [this module].py --interactive 
        import code
        print("\n[Interactive Python Shell Launched]")
        print("Available objects: `model`, `tokenizer`, `device`")
        print("Try calling `chat_iteration(...)` or `llama_score_response(...)` manually!\n")
        vars = globals().copy()
        vars.update(locals())
        shell = code.InteractiveConsole(vars)
        shell.interact()
    else:
        main(args)