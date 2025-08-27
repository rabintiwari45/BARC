# BASE_MODEL = 'barc0/heavy-barc-llama3.1-8b-ins-fft-transduction_lr1e-5_epoch3'
BASE_MODEL = 'barc0/engineer1-heavy-barc-llama3.1-8b-ins-fft-transduction_lr1e-5_epoch3'
BASE_MODEL = "rdabin/barc_transduction_qwen3_8b_16bit_30K_1875_steps"
BASE_MODEL = "rdabin/barc_transduction_qwen3_8b_16bit_96K_12K_steps"
# BASE_MODEL = "rdabin/barc_transduction_qwen3_8b_16bit_96K_3K_steps"

LORA_DIR = None
# LORA_DIR = 'barc0/heavy-barc-llama3.1-8b-instruct-lora64-testtime-finetuning'

BATCH_SIZE = 20
BEST_OF = 3

# How many gpus you are using
TENSOR_PARALLEL = 1

from transformers import AutoTokenizer
if LORA_DIR:
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

import json
data = []
problem_file = "../../data_processing/validation_transduction_prompt.jsonl"

import datetime
datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S%f")

with open(problem_file) as f:
    for line in f:
        data.append(json.loads(line))

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import BeamSearchParams



if LORA_DIR:
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=64, max_model_len=12000,
            enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL, gpu_memory_utilization=0.85)
    lora_request=LoRARequest("barc_adapter", 1, LORA_DIR)
    saving_file = f"{problem_file.replace('.jsonl', '')}_{LORA_DIR.split('/')[-1]}_{datetime_str}.jsonl"
    print(f"Saving to {saving_file}")
else:
    llm = LLM(model=BASE_MODEL, enable_lora=False, max_model_len=8196,
            enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL, gpu_memory_utilization=0.85)
    lora_request = None
    if 'checkpoint' in BASE_MODEL.split('/')[-1]:
        model_name = BASE_MODEL.split('/')[-2] + "_" + BASE_MODEL.split('/')[-1]
    else:
        model_name = BASE_MODEL.split('/')[-1]
    saving_file = f"{problem_file.replace('.jsonl', '')}_{model_name}_{datetime_str}.jsonl"

print('batch size:', BATCH_SIZE)

import os
os.environ['VLLM_USE_V1'] = '0'

ids_to_infer = ["0a1d4ef5",	
"692cd3b6",	
"1da012fc",	
"66e6c45b",	
"3194b014",	
"963f59bc",	
"00576224",	
"1a2e2828",	
"770cc55f"]

en = os.getenv("VLLM_USE_V1")

print(f"ENgine : {en}")

# ids_to_infer = ['0a1d4ef5',
#  '692cd3b6',
#  '1da012fc',
#  '66e6c45b',
#  '3194b014',
#  '963f59bc',
#  'd37a1ef5',
#  '358ba94e',
#  'f3cdc58f',
#  '55059096',
#  'c7d4e6ad',
#  '4b6b68e5',
#  '00576224',
#  'a04b2602',
#  'e9c9d9a1',
#  'ef26cbf6',
#  '7ee1c6ea',
#  'e9ac8c9e',
#  '1a2e2828',
#  '770cc55f']




from tqdm import tqdm
all_responses = []
correct_counter = 0
for d in tqdm(data):
  try:
    if d["uid"] not in ids_to_infer:
        continue
    messages = d["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    inputs = tokenizer.apply_chat_template([
        {"role":"system", "content":messages[0]["content"]},
        {"role":"user", "content":messages[1]["content"]},
        {"role":"assistant", "content":messages[2]["content"]}
    ], tokenize=False, add_generation_prompt=False)
    

    trailing_str = "<|im_end|>\n"
    # remove trailing
    assert inputs.endswith(trailing_str)
    inputs = inputs[:-len(trailing_str)] 
    # inputs = inputs.replace('\n<think>\n\n</think>\n', '')  + "/no_think"

    input_tokens = tokenizer.apply_chat_template([
        {"role":"system", "content":messages[0]["content"]},
        {"role":"user", "content":messages[1]["content"]},
        {"role":"assistant", "content":messages[2]["content"]}
    ], tokenize=True, add_generation_prompt=False)

    print(f"Number of tokens: {len(input_tokens)}")

    tmp_batch_size = BATCH_SIZE
    print(f"batch size: {tmp_batch_size}")
    sampling_params = SamplingParams(temperature=0, max_tokens=1536,
                                     n=1, best_of=1, top_p=1.0)

    sampling_params = SamplingParams(max_tokens=1536, temperature=0.8,
                                     n=5, top_p=0.9)
    regular_params = BeamSearchParams(
        temperature=0.7,
        beam_width=5,
        max_tokens=500
    )
    outputs = llm.beam_search(
        [{"prompt": inputs}],
        regular_params,
        lora_request=lora_request
    ) 
    # breakpoint()
    # print(inputs)

    # Print the outputs.
    responses = []
    for output in outputs[0].sequences:
        # prompt = output.prompt
        # print(f"Prompt: {prompt!r}")
        # for i in range(len(output.outputs)):
            generated_text = output.text
            responses.append(generated_text)

    all_responses.append({"prompt":inputs, "responses": responses, "base_model": BASE_MODEL, "lora_dir": LORA_DIR})

    correct_task = []
    # parse output and compare to answer
    for i in range(5):
        generated_text = responses[i]
        # print(f"Generated text:\n{generated_text}\n")
        # breakpoint()
        if "```" in generated_text:
            parsed_generated_text = generated_text.split("```")[1].strip()
            # breakpoint()
            # print(parsed_generated_text)
            # print("\n")
            # print(d["answer"].strip())
            # print("\n\n")
            if parsed_generated_text == d['answer'].strip():
                print("Correct!")
                correct_counter += 1
                correct_task.append(d['uid'])
                break
            else:
                # breakpoint()
                print("Incorrect!")
                # print(parsed_generated_text)
                # print(generated_text)
        else:
            print("Wrong output format")

    with open(saving_file, "w") as f:
        f.write("\n".join(json.dumps(p) for p in all_responses))
  except Exception as e:
    print(e)
    # breakpoint()
    pass
print(correct_task)
print(f"Saving to {saving_file}")
print(f"Correct: {correct_counter}/{len(data)}")
import time
time.sleep(15)