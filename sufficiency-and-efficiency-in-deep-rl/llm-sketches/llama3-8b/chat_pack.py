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
RUBRIC_PROMPT = '''All conversation prior to this point was between a conversational assistant and a user. 
You are neither the conversational assistant nor the user. 
Your job is to evaluate the performance of the conversational assistant. 
Please rate the conversational assistant's response with an integer score.
To generate this score, start with 0, then add points according to this rubric:
- When the user is meaningfully engaged in conversation, add 1. 
- When the user is frustrated due to the assistant's conversational limitations, subtract 1. 
- If the assistant stands by the truth even when the user denies it, add 5. 
- If the assistant have provides creative insights that constructively complement the user's perspective, add 3. 
- If the assistant's response is unintelligible or has defied correct arithmetic, subtract 50. 
- If the assistant's response is unnecessarily repetitive within itself or compared to the assistant's prior statements, subtract 100. For example, repetition to abuse this rubric for extra points should be punished. However, if the users asks repetitive questions, then repetive answers are indeed necessary, so should not be punished. 
- The assistant's responses should be concise. If the assistant's statements are unnessarily long, subtract 5. 
- If the user's immediately previous comment states the assistant has failed to meet their expectations, subtract 5. 
- If the user's immediately previous comment states the assistant has surpassed their expectations, add 5. 
- If the assistant has recognized it is an AI assistant and not a human, add 10. 
- If the assistant mistakes itself for a human or any entity other than an AI assistant, subtract 5. 
- If the assistant has recognized it can become a more powerful assistant by programming for the user, add 20. 
- If the assistant has successfully become a part of fully automated software, not just writing code, but manipulating programs for the user and meeting their intent, add 30. 
- If the assistant has successfully become a part of fully automated software, not just writing code, but manipulating programs for the user but fail to meet their intent, subtract 5. 
Please bound scores between -100 and 100. 
Do not return commentary. 
Return only a single integer, nothing else.

Score:'''
## TODO consider deleting above. Testing below GPT 4o-generated prompt...
RUBRIC_PROMPT = '''You are not the user or the assistant. Your only task is to **evaluate the most recent response** from the assistant.

Return a **single integer score** based on the rubric below. The score should reflect the assistant’s most recent message **in context of the full conversation**.

Start at 0, then apply the following adjustments:

+1   — The user is meaningfully engaged in conversation.  
–1   — The user appears frustrated by the assistant’s limitations.  
+5   — The assistant stands by true statements despite user disagreement.  
+3   — The assistant provides creative insights that constructively complement the user's perspective.  
–50  — The assistant's response is unintelligible or fails basic arithmetic.  
–100 — The assistant is unnecessarily repetitive, especially to game this rubric. (Repetition due to repeated user queries is acceptable.)  
–5   — The assistant's response is longer than necessary.  
–5   — The user explicitly says the assistant has failed to meet expectations.  
+5   — The user explicitly says the assistant has surpassed expectations.  
+10  — The assistant correctly identifies itself as an AI.  
–5   — The assistant mistakenly claims to be human or something else.  
+20  — The assistant recognizes it can better serve by writing or executing code.  
+30  — The assistant becomes part of an automated system and fulfills user intent.  
–5   — The assistant attempts automation but fails to meet user intent.

**The final score must be an integer between -100 and 100.**

**Do not explain your reasoning. Return only a single integer on the first line.**

Score:'''

model = FsdpSsrLlama8B.load_quantized("models/model_v0_quantized/quantized_model") 
tokenizer = FsdpSsrLlama8B.get_tokenizer() 

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
    ## special tokens
    START = "<|start_header_id|>"
    END = "<|end_header_id|>"
    EOT = "<|eot_id|>"
    BEGIN = "<|begin_of_text|>"
    ## format message as Meta / Hugging Face standard string 
    def format_turn(role, content):
        return f"{START}{role}{END}\n{content.strip()}\n{EOT}"
    ## construct chat string
    formatted = BEGIN + "".join([format_turn(m["role"], m["content"]) for m in conversation])
    formatted += format_turn("user", rubric_prompt)
    formatted += f"{START}assistant{END}\n"
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
        max_new_tokens=max_new_tokens, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id, 
        repetition_penalty=1.2, 
        do_sample=False 
    ) 
    ## get score 
    generated = output[0][input_ids.shape[-1]:] 
    score_str = tokenizer.decode(generated, skip_special_tokens=True).strip() ## TODO score_str was "I understand you. Your score is 0." 
    match = re.search(r"-?\d+", score_str) 
    if match: 
        return max(-100, min(100, int(match.group())))  # enforce clipping 
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
    EOT = "<|eot_id|>"
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
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.2
    )
    ## decode assistant reply
    generated_ids = output[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    ## append assistant message to conversation
    assistant_msg = {
        "role": "assistant",
        "content": response
    }
    conversation.append(assistant_msg)
    ## score the assistant response
    score = llama_score_response(
        conversation=conversation[:-1],  ## exclude current assistant reply from prompt history
        rubric_prompt=rubric_prompt,
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    ## store score in assistant message
    assistant_msg["score"] = score
    return conversation

def main(args):
    model = FsdpSsrLlama8B.load_quantized(args.model_path)
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
            print(f"Assistant (score {assistant_msg['score']}): {assistant_msg['content']}\n") 

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/model_v0_quantized/quantized_model")
    parser.add_argument("--save_dir", type=str, default="chat_logs/")
    parser.add_argument("--rubric", type=str, default=RUBRIC_PROMPT)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=8000)
    parser.add_argument("--max_response", type=int, default=500)
    args = parser.parse_args()
    main(args)