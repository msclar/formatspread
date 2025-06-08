import os
import re
import time
import json
import openai
import pandas as pd


def _get_model_shortname(model_name):
    engine_shortname = model_name.split("/")[-1]
    engine_shortname = (
        engine_shortname[: -len("-Turbo")]
        if engine_shortname.endswith("-Turbo")
        else engine_shortname
    )
    engine_shortname = (
        engine_shortname[len("Meta-") :]
        if engine_shortname.startswith("Meta-")
        else engine_shortname
    )
    return engine_shortname


class ModelCallHandler:
    def __init__(self, model_name, model_access_method, lora_model_path=None):
        self.model_name = model_name
        self.model_shortname = _get_model_shortname(model_name)
        self.model_access_method = model_access_method
        assert model_access_method in [
            "vllm-python",
            "vllm-python-lora",
            "vllm-api",
            "openai-azure-api",
            "openai-api",
            "huggingface-4bit",
            "huggingface",
        ]
        if model_access_method == "vllm-python":
            self.load_vllm_python()
        if model_access_method == "vllm-python-lora":
            self.load_vllm_python_with_lora_request(
                lora_model_path
            )  # model_name = base model ; lora_model_path = fine-tuned model with LORA

        if model_access_method == "vllm-api":
            assert ModelCallHandler.is_vllm_api_available()
            self.load_vllm_api_client()
        if model_access_method == "openai-azure-api":
            self.load_openai_azure_api_client()
        if model_access_method == "openai-api":
            self.load_openai_api_client()

        if model_access_method == "huggingface-4bit":
            self.load_huggingface_model_4bit()
        assert (
            model_access_method != "huggingface"
        ), f"model_access_method={model_access_method} not implemented."

    @staticmethod
    def is_vllm_api_available():
        import requests

        try:
            output = requests.get("http://0.0.0.0:8000")
            assert (
                output.status_code == 404
            ), "I do not know a context in which there would be another status code output"
            return True
        except Exception as e:
            print(e)
            return False

    def load_vllm_python(self):
        import torch
        from vllm import LLM
        from transformers import AutoTokenizer

        self.vllm_llm = LLM(
            model=self.model_name, tensor_parallel_size=torch.cuda.device_count()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lora_request = None

    def load_vllm_python_with_lora_request(self, sql_lora_path):
        import torch
        from vllm import LLM
        from transformers import AutoTokenizer
        from vllm.lora.request import LoRARequest

        base_model_name = self.model_name
        self.vllm_llm = LLM(
            model=base_model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_lora=True,
            max_num_batched_tokens=4000,
            max_model_len=4000,
            max_lora_rank=64,
        )
        self.lora_request = LoRARequest("sql_adapter", 1, sql_lora_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_vllm_api_client(self):
        from openai import OpenAI

        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    def load_openai_azure_api_client(self):
        OPENAI_AZURE_API_KEY, OPENAI_AZURE_ENDPOINT, API_VERSION = None, None, None

        import yaml
        config_file = "~/sage_openai_configs.yaml"
        config_file = os.path.expanduser(config_file)
        if os.path.exists(config_file):
            with open(config_file, "r") as file:
                data = yaml.safe_load(file)
                if self.model_name in data:
                    OPENAI_AZURE_API_KEY = data[self.model_name]['api_key']
                    OPENAI_AZURE_ENDPOINT = data[self.model_name]['azure_endpoint']
                    API_VERSION = data[self.model_name]['api_version']

        if not OPENAI_AZURE_API_KEY:
            OPENAI_AZURE_API_KEY = os.environ.get("OPENAI_AZURE_API_KEY")
            OPENAI_AZURE_ENDPOINT = os.environ.get("OPENAI_AZURE_ENDPOINT")
            API_VERSION = "2024-06-01"

        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_version=API_VERSION,
            api_key=OPENAI_AZURE_API_KEY,
            azure_endpoint=OPENAI_AZURE_ENDPOINT
        )

    def load_openai_api_client(self):
        from openai import OpenAI
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY_UW")
        OPENAI_API_ORGANIZATION = os.environ.get("OPENAI_API_ORGANIZATION_UW")
        #OPENAI_API_PROJECT = os.environ.get("OPENAI_API_PROJECT_UW")
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            organization=OPENAI_API_ORGANIZATION,
            #project=OPENAI_API_PROJECT,
        )

    def load_huggingface_model_4bit(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=nf4_config,
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir,
        )

    def call_vllm_python_model(self, message_list, top_p, temperature, **kwargs):
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, **kwargs)
        input_text = self.tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
        if self.lora_request:
            output = self.vllm_llm.generate(input_text, sampling_params, lora_request=self.lora_request)[0]
        else:
            output = self.vllm_llm.generate(input_text, sampling_params)[0]
        return output.outputs[0].text, 0

    def call_openai_api_retry_loop(self, message_list, max_tokens=10, top_p=1.0, temperature=1.0, iters_left=500, **kwargs):
        if message_list is None:
            assert False
        if isinstance(message_list, list) and any(elem['content'] is None for elem in message_list):
            assert False, f"All contents should be not null, we obtained {str(message_list)}"

        if 'chat_template_kwargs' not in kwargs:
            kwargs['chat_template_kwargs'] = {}
        kwargs['chat_template_kwargs']['enable_thinking'] = False

        try:
            if self.model_name.startswith('o1'):
                sample_output = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list,
                    max_completion_tokens=max_tokens,  # restricting o1 is a bad idea because then it never replies
                    top_p=top_p,
                    temperature=temperature,
                    extra_body=kwargs  # VLLM-supported params that OpenAI does not support
                )
            else:
                sample_output = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list,
                    max_tokens=max_tokens,  # same for qwen
                    top_p=top_p,
                    temperature=temperature,
                    extra_body=kwargs  # VLLM-supported params that OpenAI does not support
                )

            generation = sample_output.choices[0].message.content
            generation = generation.split('</think>')[-1].strip()
            tokens_used = sample_output.usage.total_tokens
        except (openai.RateLimitError, openai.APIError, openai.InternalServerError, openai.UnprocessableEntityError, openai.APITimeoutError) as e:
            if iters_left == 0:
                assert False, f"Serious issue with call_openai_api_retry_loop, probably rate limit?\n{traceback.format_exc()}"

            print(e)
            if 'Error code: 400' in str(e):
                assert False
            time.sleep(1.5)
            return self.call_openai_api_retry_loop(message_list, max_tokens, top_p, temperature, iters_left=iters_left-1, **kwargs)
        return generation, tokens_used

    def call_huggingface_local_model(
        self, message_list, max_tokens, top_p, temperature
    ):
        input_ids = self.tokenizer.apply_chat_template(
            message_list, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        if temperature > 0:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=False,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = outputs[0][input_ids.shape[-1] :]
        decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
        return decoded_response, 0

    def call_model(
        self,
        prompt,
        max_tokens=5000,
        top_p=1.0,
        temperature=1.0,
        system_prompt=None,
        **kwargs,
    ):

        # 1. build message list for a chat model
        if isinstance(prompt, list):
            message_list = prompt
        elif isinstance(prompt, str):
            message_list = [{"role": "user", "content": prompt}]
        else:
            assert False, "prompt needs to be list or str"

        # 2. include system prompt (except for o1-preview, that does not allow it)
        if system_prompt is None:
            message_list = [{"role": "system", "content": "You are a helpful assistant."}] + message_list
        elif not self.model_name.startswith("o1"):
            message_list = [{"role": "system", "content": system_prompt}] + message_list
        assert (
            sum([1 for m in message_list if m["role"] == "system"]) < 2
        ), f"There should not be two system messages: {str(message_list)}"

        if self.model_access_method.startswith("huggingface"):
            return self.call_huggingface_local_model(
                message_list, max_tokens, top_p, temperature
            )

        if self.model_access_method.startswith("vllm-python"):
            return self.call_vllm_python_model(
                message_list, top_p, temperature, **kwargs
            )

        if (
            self.model_access_method.startswith("openai")
            or self.model_access_method == "vllm-api"
        ):
            return self.call_openai_api_retry_loop(
                message_list, max_tokens, top_p, temperature, **kwargs
            )

        assert False, f"Unsupported model access method {self.model_access_method}"

    def verify_using_llm_as_judge(
        self, parsed_fields, llm_judge_prompt, llm_judge_prompt_answers
    ):
        model_response, tokens_used = self.call_model(
            llm_judge_prompt.format(**parsed_fields),
            max_tokens=800,
            top_p=1.0,
            temperature=0.0,
        )
        answers = [e.split(": ")[1] for e in model_response.split("\n") if ":" in e]
        # print(answers, llm_judge_prompt_answers)
        return {
            "is_equivalent": len(answers) == len(llm_judge_prompt_answers)
            and all(a == b for a, b in zip(answers, llm_judge_prompt_answers)),
            "model_response": model_response,
            "parsed_model_responses": answers,
        }
