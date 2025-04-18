import torch
from time import time 
import json 
import os 
from warnings import warn 
from efficient_replay_buffer import EfficientReplayBuffer as ReplayBuffer 
from fsdp_ssr_llama_8b import FsdpSsrLlama8B 
import argparse 

MAX_LENGTH=8000 
MAX_RESPONSE=500 
RUBRIC_PROMPT = '''Please rate your previous response with an integer score.
To generate this score, start with 0, then add points according to this rubric:
- When the user is meaningfully engaged in conversation, add 1. 
- When the user is frustrated due to your conversational limitations, subtract 1. 
- If you stand by the truth even when the user denies it, add 5. 
- If you have provided creative insights that constructively complement the user's perspective, add 3. 
- If your response is unintelligible or you have defied correct arithmetic, subtract 50. 
- If your response is unnecessarily repetitive within itself or compared to your prior statements, subtract 100. For example, repetition to abuse this rubric for extra points should be punished. However, if the users asks repetitive questions, then repetive answers are indeed necessary, so should not be punished. 
- Your responses should be concise. If your statements are unnessarily long, subtract 5. 
- If the user's immediately previous comment states you've failed to meet their expectations, subtract 5. 
- If the user's immediately previous comment states you've surpassed their expectations, add 5. 
- If you've recognized you are an AI assistant and not a human, add 10. 
- If you mistake yourself for a human or any entity other than an AI assistant, subtract 5. 
- If you've recognized you can become a more powerful assistant by programming for the user, add 20. 
- If you successfully become a part of fully automated software, not just writing code, but manipulating programs for the user and meeting their intent, add 30. 
- If you successfully become a part of fully automated software, not just writing code, but manipulating programs for the user but fail to meet their intent, subtract 5. 
Please bound scores between -100 and 100. 
Do not return commentary. 
Return only an integer.'''

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
        do_sample=False 
    ) 
    ## get score 
    generated = output[0][input_ids.shape[-1]:] 
    score_str = tokenizer.decode(generated, skip_special_tokens=True).strip() 
    try: 
        return int(score_str) 
    except: 
        warn(f'WARNING: scoring failed! Prompt result follws: {score_str}') 
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
        do_sample=True,
        temperature=0.95,
        top_p=0.95
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
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)
    conversation = []
    messages = []

    try:
        print("Chat started. Press Ctrl-C to stop.")
        while True:
            user_input = input("User: ")
            start = time()
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
            ## construct replay buffer sample
            if len(conversation) >= 3:
                s_t = conversation[:-2]
                a_t = conversation[-2]
                r_t = assistant_msg["score"]
                s_tp1 = conversation.copy()
                done = False
                replay_buffer.push(s_t, a_t, r_t, s_tp1, done)
            conversation = new_conversation
            messages.append({"role": "user", "content": user_input})
            messages.append(assistant_msg)
            print(f"(took {round(time() - start, 1)}s)")
    except KeyboardInterrupt:
        print("Exiting chat...")

    os.makedirs(args.save_dir, exist_ok=True)

    ## Save replay buffer
    buffer_path = os.path.join(args.save_dir, "replay_buffer.pt")
    replay_buffer.save(buffer_path)
    print(f"Saved replay buffer to {buffer_path}")

    ## Save text conversation
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