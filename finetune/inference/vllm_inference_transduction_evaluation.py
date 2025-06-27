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

# ids_to_infer = ["0a1d4ef5",	
# "692cd3b6",	
# "1da012fc",	
# "66e6c45b",	
# "3194b014",	
# "963f59bc",	
# "00576224",	
# "1a2e2828",	
# "770cc55f"]

ids_to_infer = ['0a1d4ef5',
 '692cd3b6',
 '1da012fc',
 '66e6c45b',
 '3194b014',
 '963f59bc',
 'd37a1ef5',
 '358ba94e',
 'f3cdc58f',
 '55059096',
 'c7d4e6ad',
 '4b6b68e5',
 '00576224',
 'a04b2602',
 'e9c9d9a1',
 'ef26cbf6',
 '7ee1c6ea',
 'e9ac8c9e',
 '1a2e2828',
 '770cc55f']

ids_to_infer_ = ['19bb5feb',
  '33b52de3',
  'ba9d41b8',
  'b15fca0b',
  '45737921',
  '5b526a93',
  'ca8f78db',
  '0a1d4ef5',
  '81c0276b',
  '1e97544e',
  '5a5a2103',
  '5783df64',
  'f45f5ca7',
  '3979b1a8',
  '05a7bcf2',
  '692cd3b6',
  '8cb8642d',
  'aa300dc3',
  'd4c90558',
  'bf699163',
  '292dd178',
  '9110e3c5',
  '705a3229',
  '1990f7a8',
  '332efdb3',
  '4f537728',
  '67636eac',
  '00dbd492',
  '67c52801',
  '1da012fc',
  '575b1a71',
  '66e6c45b',
  'fafd9572',
  'ea959feb',
  'cd3c21df',
  '8597cfd7',
  '3194b014',
  '2546ccf6',
  '45bbe264',
  'da2b0fe3',
  'e1baa8a4',
  '833dafe3',
  '4cd1b7b2',
  '7d1f7ee8',
  'af24b4cc',
  '963f59bc',
  'd37a1ef5',
  '358ba94e',
  'c3202e5a',
  '2753e76c',
  'f3cdc58f',
  '55059096',
  '845d6e51',
  '3b4c2228',
  'c7d4e6ad',
  'e7dd8335',
  '94be5b80',
  'bf89d739',
  '516b51b7',
  'c87289bb',
  '94414823',
  'dc2e9a9d',
  'e57337a4',
  '4aab4007',
  '4b6b68e5',
  '8a371977',
  '1c0d0a4b',
  '759f3fd3',
  '8dae5dfc',
  'a934301b',
  '4e469f39',
  '6df30ad6',
  'f5aa3634',
  'fc754716',
  '92e50de0',
  '31adaf00',
  '7d18a6fb',
  '551d5bf1',
  '73182012',
  '5ffb2104',
  '62ab2642',
  '42a15761',
  '68b67ca3',
  '903d1b4a',
  '642d658d',
  '00576224',
  '9b4c17c4',
  '21f83797',
  '0b17323b',
  '6ea4a07e',
  'a04b2602',
  '59341089',
  'e7a25a18',
  'e88171ec',
  'e9c9d9a1',
  '09c534e7',
  '88207623',
  'a406ac07',
  'c8b7cc0f',
  'ac0c5833',
  'd931c21c',
  '7c9b52a0',
  '95a58926',
  'a680ac02',
  'cad67732',
  'ef26cbf6',
  'f823c43c',
  '72a961c9',
  'e760a62e',
  'b7cb93ac',
  '3ee1011a',
  '50aad11f',
  '1d398264',
  'c64f1187',
  'fb791726',
  '8fbca751',
  'b7999b51',
  '3490cc26',
  '7ee1c6ea',
  '2b01abd0',
  'aa4ec2a5',
  'b7fb29bc',
  'f0df5ff0',
  '414297c0',
  '58743b76',
  '6f473927',
  'c074846d',
  '2072aba6',
  '319f2597',
  '84f2aca1',
  'cf133acc',
  '009d5c81',
  '2f0c5170',
  'c6e1b8da',
  'e4075551',
  '917bccba',
  'e66aafb8',
  'f83cb3f6',
  '0becf7df',
  '62b74c02',
  '642248e4',
  '477d2879',
  '137f0df0',
  '93b4f4b3',
  'f21745ec',
  '29700607',
  'b0f4d537',
  'e9ac8c9e',
  '1a2e2828',
  '52fd389e',
  '770cc55f',
  '782b5218',
  '712bf12e',
  'd304284e',
  'f0afb749',
  '3a301edc',
  'e681b708',
  '3391f8c0',
  'e74e1818',
  '070dd51e',
  '50a16a69',
  '0607ce86',
  '1c02dbbe']

ids_to_infer_ = ['9c1e755f',
  'ac3e2b04',
  '0bb8deee',
  '762cd429',
  'e7639916',
  '42918530',
  'ca8de6ea',
  'e1d2900e',
  '5289ad53',
  '8ee62060',
  'aee291af',
  '2697da3f',
  '351d6448',
  '7bb29440',
  '97239e3d',
  'bcb3040b',
  'f3e62deb',
  '54db823b',
  '64a7c07e',
  'e95e3d8e',
  '99306f82',
  'e0fb7511',
  'ae58858e',
  '992798f6',
  '9a4bb226',
  '15113be4',
  '93c31fbe',
  '9f27f097',
  'e9b4f6fc',
  '20818e16',
  'be03b35f',
  '03560426',
  'bbb1b8b6',
  '27a77e38',
  '9bebae7a',
  '281123b4',
  '3f23242b',
  '626c0bcc',
  '896d5239',
  '9ddd00f0',
  'e69241bd',
  'fd096ab6',
  'fe9372f3',
  '69889d6e',
  'df8cc377']

ids_to_infer_medium = ['e41c6fd3',
  '0a2355a6',
  '60a26a3e',
  'baf41dbf',
  '17b80ad2',
  'f4081712',
  'dd2401ed',
  'f9d67f8b',
  '140c817e',
  '11e1fe23',
  '7e02026e',
  'ff72ca3e',
  '4e45f183',
  'bb52a14b',
  '15696249',
  '73ccf9c2',
  '32e9702f',
  '72207abc',
  '0f63c0b9',
  'b9630600',
  '4acc7107',
  'aa18de87',
  'c658a4bd',
  '12997ef3',
  'e9bb6954',
  '40f6cd08',
  'b457fec5',
  '639f5a19',
  '212895b5',
  'ce8d95cc',
  '6a11f6da',
  'd282b262',
  '4c177718',
  'a096bf4d',
  'ed74f2f2',
  '9b2a60aa',
  '9b365c51',
  'cb227835',
  'bd14c3bf',
  'af22c60d',
  '2c737e39',
  '8ba14f53',
  '2685904e',
  '4852f2fa',
  '256b0a75',
  '5b6cbef5',
  '5207a7b5',
  'd47aa2ff',
  '67b4a34d',
  '0d87d2a6',
  '22a4bbc2',
  'd492a647',
  '7c8af763',
  'e21a174a',
  'b20f7c8b',
  '103eff5b',
  'de493100',
  '9772c176',
  '0e671a1a',
  '15663ba9',
  '9caba7c3',
  '7039b2d7',
  '13713586',
  '2c0b0aff',
  '16b78196',
  'd4b1c2b1',
  '456873bc',
  '18419cfa',
  'd56f2372',
  'a3f84088',
  'e133d23d',
  '6ad5bdfd',
  '9356391f',
  '505fff84',
  'ecaa0ec1',
  '48131b3c',
  '981571dc',
  '20981f0e',
  'e5790162',
  'f8be4b64',
  'd94c3b52',
  'ac2e8ecf',
  'e345f17b',
  '17cae0c1',
  'e633a9e5',
  'e872b94a',
  'b1fc8b8e',
  '25094a63',
  '695367ec',
  'a59b95c0',
  '1d0a4b61',
  '12422b43',
  'c62e2108',
  'c97c0139',
  'e619ca6e',
  '0c786b71',
  '1a6449f1',
  '60c09cac',
  'd2acf2cb',
  '9c56f360',
  '184a9768',
  '5af49b42',
  'd017b73f',
  '96a8c0cd']

ids_to_infer_hard = ['e5c44e8f',
  '604001fa',
  '1e81d6f9',
  'e78887d1',
  '4364c1c4',
  '14754a24',
  'c1990cce',
  '85fa5666',
  'a57f2f04',
  '9def23fe',
  'c35c1b4c',
  '506d28a5',
  '79369cc6',
  'c48954c1',
  'cfb2ce5a',
  '2037f2c7',
  '85b81ff1',
  '55783887',
  '696d4842',
  'da515329',
  'd5c634a2',
  'bc4146bd',
  'ac605cbb',
  '27f8ce4f',
  '2a5f8217',
  'bf32578f',
  '817e6c09',
  'ccd554ac',
  'fd4b2b02',
  '195ba7dc',
  '73c3b0d8',
  'f3b10344',
  '48f8583b',
  'c663677b',
  'e7b06bea',
  '12eac192',
  '423a55dc',
  'aab50785',
  '66f2d22f',
  '0692e18c',
  'c92b942c',
  '3ed85e70',
  '84db8fc4',
  '8b28cd80',
  'ce039d91',
  'd19f7514',
  'dc2aa30b',
  'b7f8a4d8',
  'ed98d772',
  'f5c89df1',
  'b0722778',
  '37d3e8b2',
  '50f325b5',
  '08573cc6',
  '310f3251',
  '7d419a02',
  '3d31c5b3',
  '7953d61e',
  '8e2edd66',
  'b942fd60',
  '929ab4e9',
  '94133066',
  '136b0064',
  '90347967',
  '5d2a5c43']

ids_to_infer = ['e99362f0',
  '1acc24af',
  'f9a67cb5',
  'ad7e01d0',
  'ea9794b1',
  '58e15b12',
  '891232d6',
  '5833af48',
  '4ff4c9da',
  '5b692c0f',
  'e2092e0c',
  '0934a4d8',
  '47996f11',
  '0c9aba6e',
  '34b99a2b',
  '1c56ad9f',
  'e6de6e8f',
  'fea12743',
  '31d5ba1a',
  '79fb03f4',
  '8719f442',
  'a8610ef7',
  'b4a43f3b']


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
    outputs = llm.generate(
        inputs,
        sampling_params,
        lora_request=lora_request
    ) 
    # breakpoint()
    # print(inputs)

    # Print the outputs.
    responses = []
    for output in outputs:
        prompt = output.prompt
        # print(f"Prompt: {prompt!r}")
        for i in range(len(output.outputs)):
            generated_text = output.outputs[i].text
            responses.append(generated_text)

    all_responses.append({"prompt":inputs, "responses": responses, "base_model": BASE_MODEL, "lora_dir": LORA_DIR})

    correct_task = []
    # parse output and compare to answer
    for i in range(5):
        generated_text = responses[i]
        # print(f"Generated text:\n{generated_text}\n")
        # breakpoint()
        if "```" in generated_text:
            parsed_generated_text = generated_text.split("```")[0].strip()
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
        else:
            print("Wrong output format")

    with open(saving_file, "w") as f:
        f.write("\n".join(json.dumps(p) for p in all_responses))
  except:
    pass
print(correct_task)
print(f"Saving to {saving_file}")
print(f"Correct: {correct_counter}/{len(data)}")
import time
time.sleep(15)