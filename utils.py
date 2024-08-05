import copy
import math
import os
import psutil

import bert_score
import openai
from rouge_score import rouge_scorer
import torch
import torch.nn.functional as F

from grammar_definition import apply_prompt_format, flatten
from parsing_supernatural_instructions_tasks import OPEN_GENERATION_SUPERNATURAL_INSTRUCTIONS_TASKS

openai.api_key = os.getenv("OPENAI_API_KEY")
PRINT_HIDDEN_STATE = False


def call_openai_api_with_retry(args, prompt, max_tokens=10):
    try:
        if args.gpt3_engine in ['gpt-4', 'gpt-3.5-turbo']:
            sample_output = openai.ChatCompletion.create(
                model=args.gpt3_engine,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                top_p=1.0,
                temperature=1.0
            )
            generation = sample_output['choices'][0]['message']['content'].strip()
        else:
            sample_output = openai.Completion.create(
                engine=args.gpt3_engine,
                prompt=prompt,
                max_tokens=max_tokens,
                top_p=1.0,
                temperature=1.0
            )
            generation = sample_output['choices'][0]['text'].strip()
        tokens_used = sample_output['usage']['total_tokens']
    except (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError,
            openai.error.Timeout) as e:
        print(e)
        return call_openai_api_with_retry(args, prompt)
    return generation, tokens_used


def query_model_parallelized(model, tokenizer, prompt_list, max_tokens, top_p, temperature):
    if tokenizer.chat_template is None:
        inputs = tokenizer(prompt_list, padding=True, return_tensors='pt', return_token_type_ids=False).to('cuda')
    else:
        turns = [[{"role": "user", "content": prompt}] for prompt in prompt_list]
        inputs = tokenizer.apply_chat_template(turns, padding=True, return_tensors='pt', return_dict=True, add_generation_prompt=True, return_token_type_ids=False).to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            **inputs, pad_token_id=tokenizer.pad_token_id, top_p=top_p, temperature=temperature, max_new_tokens=max_tokens,
            return_dict_in_generate=True, output_hidden_states=True, output_attentions=False, output_scores=True
        )

        logits_list = [[] for _ in range(len(prompt_list))]

        # we do not print hidden state and scores because it is too much memory spenditure
        if PRINT_HIDDEN_STATE:
            # take the first (0th) inference. Its last layer (-1) will have shape [1, prompt_size, 4096]. Take last one.
            final_prompt_hidden_state_list = [
                outputs['hidden_states'][0][-1][i, -1, :].tolist() for i in range(len(prompt_list))]
        else:
            for new_token_idx in range(len(outputs['scores'])):
                for i in range(len(prompt_list)):
                    logits = torch.topk(outputs['scores'][new_token_idx][i, :], k=100)
                    logits = [(value, index) for value, index in zip(logits.values.tolist(), logits.indices.tolist())]
                    logits_list[i].append(logits)
            final_prompt_hidden_state_list = [None for _ in range(len(prompt_list))]

    sequences = outputs['sequences']
    if tokenizer.chat_template is not None:
        sequences = [seq[len(input_ids):] for seq, input_ids in zip(sequences, inputs["input_ids"], strict=True)]
    generated_answer_list = [s for s in tokenizer.batch_decode(sequences, skip_special_tokens=True)]
    return generated_answer_list, logits_list, final_prompt_hidden_state_list


def _apply_prompt_format_to_extracted_fields(
        structured_prompt_format, input_fields_list, regex_key_idx_list, output_fields_list=None):
    # Precompute all format options
    prompt = {}
    for key in set(regex_key_idx_list):
        prompt[key] = flatten(structured_prompt_format.solve(
            {'enumeration_length': key,
             'print_output_fields': True,
             'exclude_text_field_for_output_fields': output_fields_list is None})
        ).replace('<|text|>', '{}')

    # add empty default values if no output will be printed. It has to be a tuple to be able to concat with input_fields
    if output_fields_list is None:
        output_fields_list = [() for _ in input_fields_list]
    else:
        output_fields_list = [(output_field,) for output_field in output_fields_list]

    formatted_inputs = []
    for input_fields, regex_key_idx, output_field in zip(input_fields_list, regex_key_idx_list, output_fields_list):
        tmp = apply_prompt_format(prompt[regex_key_idx], input_fields + output_field)
        formatted_inputs.append(tmp)

    return formatted_inputs


def _setup_formatted_demonstrations_with_definition(
        structured_prompt_format, demonstration_definition, demonstrations_outputs,
        original_to_current_multiple_choice_classes, demos_fields_list, demos_regex_key_idx_list):
    # 1. replace the variables in the demonstration definition. Used when the instruction mentions
    # multiple choice options, which need to change when the format changes
    demonstration_definition = demonstration_definition.format(
        **structured_prompt_format.find_all_formatted_field_values()
    )

    demonstrations_outputs = [demo[0] if isinstance(demo, list) else demo for demo in demonstrations_outputs]
    if original_to_current_multiple_choice_classes:
        demonstrations_outputs = [original_to_current_multiple_choice_classes[d] for d in demonstrations_outputs]

    all_demonstrations = _apply_prompt_format_to_extracted_fields(
        structured_prompt_format, demos_fields_list, demos_regex_key_idx_list, demonstrations_outputs)
    demonstration_string = demonstration_definition + "\n\n" + "\n\n".join(all_demonstrations)
    return demonstration_string


def _setup_full_prompts_to_test_on(input_fields_list, regex_key_idx_list, selected_dataset_ids,
                                   demos_fields_list, demos_regex_key_idx_list, demonstrations_outputs,
                                   demonstration_definition,
                                   structured_prompt_format, original_to_current_multiple_choice_classes,
                                   interval_ids_to_test, n_shot):
    """
    This function creates the full prompt string to be tested. This requires:

    - Formatting the demonstrations with its definition, which may require
        replacing some variables referring to multiple choice options.
    - Apply prompt format to the desired set of examples to be tested (determined by interval_ids_to_test).
    """
    if n_shot == 0:
        assert len(demos_fields_list) == 0 and len(demos_regex_key_idx_list) == 0 and len(demonstrations_outputs) == 0
    demonstration_string = _setup_formatted_demonstrations_with_definition(
        structured_prompt_format, demonstration_definition, demonstrations_outputs,
        original_to_current_multiple_choice_classes, demos_fields_list, demos_regex_key_idx_list
    )

    # filter to keep desired interval
    inputs = _apply_prompt_format_to_extracted_fields(
        structured_prompt_format,
        input_fields_list[interval_ids_to_test[0]:interval_ids_to_test[1]],
        regex_key_idx_list[interval_ids_to_test[0]:interval_ids_to_test[1]]
    )
    selected_dataset_ids = selected_dataset_ids[interval_ids_to_test[0]:interval_ids_to_test[1]]

    full_prompt_string_list = []
    for input_element, idx in zip(inputs, selected_dataset_ids):
        full_prompt_string_list.append(demonstration_string + "\n\n" + input_element)

    return full_prompt_string_list, selected_dataset_ids


# layzily initiailize a bertscore model
bert_score_model = None


def get_bert_score_model():
    global bert_score_model
    if bert_score_model is None:
        bert_score_model = bert_score.BERTScorer(lang='en')
    return bert_score_model


def evaluate_prompt_format(
        args, dataset, input_fields_list, regex_key_idx_list, selected_dataset_ids,
        demos_fields_list, demos_regex_key_idx_list, demonstrations_outputs, demonstration_definition,
        structured_prompt_format, model, tokenizer, model_will_repeat_input,
        original_to_current_multiple_choice_classes, interval_ids_to_test=(None, None)):
    """
    Function that evaluates a prompt format (i.e. node) on a given set of samples (interval_ids_to_test).
    If interval_ids_to_test is not provided, it defaults to evaluating the whole dataset.
    """

    # 1. set up input prompts including demonstrations
    input_prompt_string_list, selected_dataset_ids = _setup_full_prompts_to_test_on(
        input_fields_list, regex_key_idx_list, selected_dataset_ids,
        demos_fields_list, demos_regex_key_idx_list, demonstrations_outputs, demonstration_definition,
        structured_prompt_format, original_to_current_multiple_choice_classes, interval_ids_to_test, args.n_shot)

    # 2. update the output values if needed, i.e. if the multiple choice classes now have different names
    assert all(len(dataset[idx]['output']) >= 1 for idx in selected_dataset_ids)
    for idx in selected_dataset_ids:
        if len(dataset[idx]['output']) > 1:
            dataset[idx]['output'] = [dataset[idx]['output'][0]]
    dataset_updated = copy.deepcopy(dataset)
    if original_to_current_multiple_choice_classes:
        for idx in range(len(dataset)):
            dataset_updated[idx]['output'][0] = original_to_current_multiple_choice_classes[dataset[idx]['output'][0]]
    output_classes = sorted(list(set([dataset_updated[idx]['output'][0] for idx in selected_dataset_ids])))

    # 3. evaluate
    if args.evaluation_metric == 'probability_ranking':
        return solve_with_rank_based_scoring(
            dataset_updated, selected_dataset_ids, model, tokenizer, input_prompt_string_list, args.batch_size_llm)

    elif args.evaluation_metric in {'exact_prefix_matching', 'rouge', 'bertscore'}:
        logs = generate_text_with_metadata(
            args, input_prompt_string_list, model, tokenizer, model_will_repeat_input,
            dataset_updated, selected_dataset_ids, output_classes)
        if args.evaluation_metric == 'exact_prefix_matching':
            return exact_prefix_matching_scoring(logs)
        elif args.evaluation_metric == 'rouge':
            scorer = rouge_scorer.RougeScorer(['rougeL'])
            scores = [scorer.score(entry['generation'], entry['answer'])["rougeL"].fmeasure for entry in logs]
            avg_score = sum(scores) / len(scores)
            return (avg_score, None, None), (None, logs)
        elif args.evaluation_metric == 'bertscore':
            references = [entry['answer'] for entry in logs]
            candidates = [entry['generation'] for entry in logs]
            scorer = get_bert_score_model()
            P, R, F1 = scorer.score(candidates, references)
            return (F1.mean().item(), None, None), (None, logs)




def generate_text_with_metadata(args, input_prompt_string_list, model, tokenizer, model_will_repeat_input, dataset,
                                selected_dataset_ids, output_classes):
    logs = []
    all_tokens_used = 0
    for batch_idx in range(math.ceil(len(input_prompt_string_list) / args.batch_size_llm)):
        batch_range = [batch_idx * args.batch_size_llm, (batch_idx + 1) * args.batch_size_llm]  # [) range

        full_prompt_string_list = input_prompt_string_list[batch_range[0]:batch_range[1]]
        if args.use_gpt3:
            generation_list = []
            for prompt in full_prompt_string_list:
                generation, tokens_used = call_openai_api_with_retry(args, prompt)
                generation_list.append(generation)
                all_tokens_used += tokens_used

            score_list = [None for _ in range(len(generation_list))]
            final_prompt_hidden_state_list = [None for _ in range(len(generation_list))]
        else:
            generation_list, score_list, final_prompt_hidden_state_list = query_model_parallelized(
                model, tokenizer, full_prompt_string_list, max_tokens=args.max_new_tokens, top_p=1.0, temperature=1.0,
            )
            if args.task_filename not in OPEN_GENERATION_SUPERNATURAL_INSTRUCTIONS_TASKS:
                generation_list = [gen.lower() for gen in generation_list]
            if model_will_repeat_input and tokenizer.chat_template is None:  # for chat models, truncation happens in query_model_parallelized
                generation_list = [generation[len(full_prompt_string):]
                                   for generation, full_prompt_string in zip(generation_list, full_prompt_string_list)]

        selected_dataset_ids_list = [idx for idx in selected_dataset_ids[batch_range[0]:batch_range[1]]]
        assert len(generation_list) == len(selected_dataset_ids_list) == len(score_list) == len(
            final_prompt_hidden_state_list) == len(full_prompt_string_list)
        for generation, scores, idx, final_prompt_hidden_state, full_prompt_string in \
                zip(generation_list, score_list, selected_dataset_ids_list, final_prompt_hidden_state_list,
                    full_prompt_string_list):
            expected_output = dataset[idx]['output'][0]
            # 'entry' and 'output_classes' are needed for score generations

            current_log = {
                'entry': dataset[idx],
                'dataset_idx': idx,
                'generation': generation,
                'answer': expected_output,
                'output_classes': output_classes,
                'full_prompt_string': full_prompt_string,
                'eval_type': 'exact_prefix_matching',
                'scores': scores,
            }
            if PRINT_HIDDEN_STATE:
                current_log['final_prompt_hidden_state'] = final_prompt_hidden_state
            logs.append(current_log)

    print('Total tokens used:', all_tokens_used)
    return logs


def match_robust_to_multiple_choice(generation, answer_to_compare):
    """
    We return whether the generation matched with the expected answer.

    This function assumes clean_text has already been run.
    """
    # likewise, if the response says "article" and the right answer is "a"
    if not generation.startswith(answer_to_compare):
        return False

    # if generation starts with answer and they are the same length, they are the same string
    if len(generation) == len(answer_to_compare):
        return True

    # if the generation starts with the correct text, make sure the next char is not text or number
    # otherwise it might be just the first part of a random word (e.g. "a" with "article")
    # or if correct answer is ii, and all answers are i, ii, iii, iv, avoid being overly optimistic!
    return not generation[len(answer_to_compare)].isalpha() and not generation[len(answer_to_compare)].isdigit()


def exact_prefix_matching_scoring(logs):
    accuracy = {
        'right': [],
        'wrong': [],
        'other': [],
        'total': 0
    }
    for entry in logs:
        clean_text = lambda x: x.strip(' .,()\n-><').lower()

        right_answer = entry['entry']['output'][0]
        wrong_answers = [e for e in entry['output_classes'] if e != right_answer]

        entry['right_answer_formatted'] = right_answer
        entry['wrong_answers_formatted'] = wrong_answers

        right_answer = clean_text(right_answer)
        wrong_answers = [clean_text(e) for e in wrong_answers]
        generation = entry['generation']

        clean_generation = clean_text(generation)
        is_right = match_robust_to_multiple_choice(clean_generation, right_answer)
        is_wrong = any(
            match_robust_to_multiple_choice(clean_generation, wrong_answer) for wrong_answer in wrong_answers)

        accuracy['right'].append(is_right)
        accuracy['wrong'].append(is_wrong)
        accuracy['other'].append(not is_wrong and not is_right)
        accuracy['total'] += 1

        if 'output_classes' in entry and len(entry['output_classes']) > 50:
            del entry['output_classes']

    # not changing this since it's called from many classes
    return (sum(accuracy['right']) * 1.0 / max(accuracy['total'], 1),
            sum(accuracy['wrong']) * 1.0 / max(accuracy['total'], 1),
            accuracy['total']), (accuracy, logs)


def solve_with_rank_based_scoring(
        dataset, selected_dataset_ids, model, tokenizer, input_prompt_string_list, batch_size_llm):
    output_classes = sorted(list(set([dataset[idx]['output'][0] for idx in selected_dataset_ids])))
    assert len(output_classes) < 100
    assert tokenizer is not None and model is not None

    # if all output values are only one token, then we can just look at the output probabilities
    # instead of computing perplexity for all possible prompt+outputs!
    # also if all output values share the same prefix. E.g. ['0', '1'] tokenizes to [[1, 29871, 29900], [1, 29871, 29896]]
    # the first token id is always '1', so we ignore it
    output_classes_tokens = [t for t in tokenizer(output_classes, return_token_type_ids=False)['input_ids']]
    single_token_classes = all([len(t) == 2 for t in output_classes_tokens])
    all_classes_share_common_prefix = len(set([tuple(t[:-1]) for t in output_classes_tokens])) == 1

    accuracy = {
        'right': [],
        'wrong': [],
        'other': [],
        'total': 0
    }
    logs = []

    if single_token_classes or all_classes_share_common_prefix:
        # batching happens across inputs
        for batch_idx in range(math.ceil(len(input_prompt_string_list) / batch_size_llm)):
            print("Memory usage:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

            batch_range = [batch_idx * batch_size_llm, (batch_idx + 1) * batch_size_llm]  # [) range

            full_prompt_string_list = input_prompt_string_list[batch_range[0]:batch_range[1]]
            generation_list = get_ranking_based_generation_single_token_output_classes(
                full_prompt_string_list, output_classes, tokenizer, model)

            selected_dataset_ids_list = [idx for idx in selected_dataset_ids[batch_range[0]:batch_range[1]]]
            assert len(generation_list) == len(selected_dataset_ids_list), f"{len(generation_list)} generations, {len(selected_dataset_ids_list)} selected ids"
            assert len(generation_list) == len(full_prompt_string_list)
            for generation, idx, full_prompt_string in zip(generation_list, selected_dataset_ids_list, full_prompt_string_list):
                expected_output = dataset[idx]['output'][0]

                assert expected_output in output_classes, f"expected_output={expected_output}, output_classes={output_classes}"

                accuracy['right'].append((generation == expected_output))
                accuracy['wrong'].append((generation != expected_output and generation in output_classes))
                accuracy['other'].append((generation not in output_classes))
                accuracy['total'] += 1
                logs.append(
                    {
                        'entry': dataset[idx],
                        'dataset_idx': idx,
                        'generation': generation,
                        'answer': expected_output,
                        'output_classes': output_classes,
                        'full_prompt_string': full_prompt_string,
                        'eval_type': 'ranking_single_token',
                        'scores': None,
                    }
                )

    else:
        # batching happens inside each input, since we need to do inference for each prompt+possible_output
        for i in range(len(input_prompt_string_list)):
            idx = selected_dataset_ids[i]
            full_prompt_string = input_prompt_string_list[i]

            generation = get_ranking_based_generation_multiple_token_output_classes(
                full_prompt_string, output_classes, tokenizer, model, batch_size_llm,
            )
            expected_output = dataset[idx]['output'][0]
            assert expected_output in output_classes, f"expected_output={expected_output}, output_classes={output_classes}"

            accuracy['right'].append((generation == expected_output))
            accuracy['wrong'].append((generation != expected_output and generation in output_classes))
            accuracy['other'].append((generation not in output_classes))
            accuracy['total'] += 1

            logs.append(
                {
                    'entry': dataset[idx],
                    'dataset_idx': idx,
                    'generation': generation,
                    'answer': expected_output,
                    'output_classes': output_classes,
                    'full_prompt_string': full_prompt_string,
                    'eval_type': 'ranking_multiple_token',
                    'scores': None,
                }
            )

    return (sum(accuracy['right']) * 1.0 / max(accuracy['total'], 1),
            sum(accuracy['wrong']) * 1.0 / max(accuracy['total'], 1),
            accuracy['total']), (accuracy, logs)


def get_ranking_based_generation_single_token_output_classes(prompts, output_classes, tokenizer, model):
    top_p = 1.0
    temperature = 1.0

    # if all output values are only one token, then we can just look at the output probabilities!
    # also if all output values share the same prefix. E.g. ['0', '1'] tokenizes to [[1, 29871, 29900], [1, 29871, 29896]]
    # the first token id is always '1', so we ignore it
    output_classes_tokens = [t for t in tokenizer(output_classes, return_token_type_ids=False)['input_ids']]
    all_classes_share_common_prefix = len(set([tuple(t[:-1]) for t in output_classes_tokens])) == 1

    if tokenizer.chat_template is None:
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False)
    else:
        turns = [[{"role": "user", "content": prompt}] for prompt in prompts]
        tokenized = tokenizer.apply_chat_template(turns, return_tensors="pt", padding=True, return_dict=True, add_generation_prompt=True, return_token_type_ids=False)
    tokenized_inputs_list = tokenized["input_ids"].tolist()
    if all_classes_share_common_prefix:
        for i in range(len(tokenized_inputs_list)):
            # if the tokenized element is [1, 29871, 29900], get [29871]
            tokenized_inputs_list[i] += output_classes_tokens[0][1:-1]
    tokenized_inputs = torch.tensor(tokenized_inputs_list).to('cuda')

    with torch.no_grad():
        outputs = model.generate(input_ids=tokenized_inputs, pad_token_id=tokenizer.pad_token_id,
                                 top_p=top_p, temperature=temperature, max_new_tokens=1,
                                 return_dict_in_generate=True, output_scores=True)

    scores = outputs["scores"][0]  # first dimension = 1 since we only generate one token
    generations = []
    for i in range(len(prompts)):
        all_logits = scores[i, :].squeeze().tolist()
        all_logits_sorted = sorted([(all_logits[t[-1]], i) for i, t in enumerate(output_classes_tokens)], reverse=True)
        generations.append(output_classes[all_logits_sorted[0][1]])
    return generations


def get_ranking_based_generation_multiple_token_output_classes(prompt, output_classes, tokenizer, model,
                                                               batch_size_llm):
    output_classes_tokens = [t for t in tokenizer(output_classes, return_token_type_ids=False)['input_ids']]

    prompts = [prompt + class_seq for class_seq in output_classes]
    all_logits_list, all_tokens_list = [], []
    for batch_idx in range(math.ceil(len(prompts) / batch_size_llm)):
        batch_range = [batch_idx * batch_size_llm, (batch_idx + 1) * batch_size_llm]  # [) range
        all_logits, all_tokens = _get_input_logits_and_tokens(prompts[batch_range[0]:batch_range[1]], tokenizer, model)
        all_logits_list.extend(all_logits)
        all_tokens_list.extend(all_tokens)

    n_classes = len(output_classes)
    class_logprobs = []
    for class_index in range(n_classes):
        class_logits = all_logits_list[class_index]

        # the lengths of each class sequence in tokens
        target_token_length = (len(output_classes_tokens[class_index]))
        # we only need the logits for the end sequence
        tokens = all_tokens_list[class_index]
        # we have to go back by one because we don't care about the logits for the predicted token
        sequence_logits = class_logits[-target_token_length - 1: -1]
        sequence_tokens = tokens[-target_token_length:]
        # we take a log_softmax over all token logits for each position in the class sequence to
        #  get log probabilities, and then sum the logprobs for the tokens actually chosen
        logprobs = F.log_softmax(sequence_logits, dim=-1).to('cpu')
        class_logprob = sum(
            [logprobs[i, token] for i, token in enumerate(sequence_tokens)]
        )
        class_logprobs.append(class_logprob.item())

    return output_classes[torch.tensor(class_logprobs).argmax(dim=-1).item()]


def _get_input_logits_and_tokens(inputs, tokenizer, model):
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, return_token_type_ids=False).to('cuda')
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
    return logits, tokenized_inputs["input_ids"]
