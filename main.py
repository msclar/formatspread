import argparse
import copy
import itertools
import json
import os
import random

from data_loading import load_supernatural_instructions_task, load_instruction_induction_task
from format_evaluation import GeneticAlgorithmAmongPrompts, value_assignment_str_to_indices, \
    ThompsonSamplingAlgorithmAmongPrompts
from grammar_definition import pointers_to_all_objects, create_pointer_action_type_pairs, MAPPING_ALL_CATEGORIES, \
    holistic_node_format_sanity_checks

random.seed(0)


def _load_model(args):
    model, tokenizer, model_will_repeat_input = None, None, False

    if args.model_name and not args.use_gpt3:
        import torch
        cache_dir = args.cache_dir

        if 'Llama-2-70b-hf' in args.model_name or args.use_4bit:
            # assert args.batch_size_llm == 1
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            # torch_dtype=torch.float16 is incompatible with batching
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, use_fast=True, cache_dir=cache_dir, return_token_type_ids=False)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, cache_dir=cache_dir, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            )
            model_will_repeat_input = True

            # Add special padding token
            special_tokens_dict = {'pad_token': '<pad>'}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            tokenizer.padding_side = "left"
            print('We have added', num_added_toks, 'tokens')

            # Resize the token embeddings
            model.resize_token_embeddings(len(tokenizer))

            # Set `pad_token_id` in model's configuration
            model.config.pad_token_id = tokenizer.pad_token_id

        elif any(t in args.model_name.lower() for t in ['llama', 'falcon', 'mistral', 'mixtral']) \
                and args.batch_size_llm is not None:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # torch_dtype=torch.float16 is incompatible with batching
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=cache_dir,
                                                      return_token_type_ids=False)
            model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=cache_dir, trust_remote_code=True)
            model = model.to('cuda')
            model_will_repeat_input = True

            # Add special padding token
            special_tokens_dict = {'pad_token': '<pad>'}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            tokenizer.padding_side = "left"
            print('We have added', num_added_toks, 'tokens')

            # Resize the token embeddings
            model.resize_token_embeddings(len(tokenizer))

            # Set `pad_token_id` in model's configuration
            model.config.pad_token_id = tokenizer.pad_token_id

        elif not args.use_gpt3:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, use_fast=True, cache_dir=cache_dir, return_token_type_ids=False)
            model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=cache_dir, trust_remote_code=True)
            model = model.to('cuda')
            model_will_repeat_input = True

        model.tie_weights()
        model.eval()
        model.tie_weights()

    return model, tokenizer, model_will_repeat_input


def _load_task(args):
    if args.dataset_name == 'natural-instructions':
        from parsing_supernatural_instructions_tasks import OPEN_GENERATION_SUPERNATURAL_INSTRUCTIONS_TASKS
        args.max_new_tokens = 50 if args.task_filename in OPEN_GENERATION_SUPERNATURAL_INSTRUCTIONS_TASKS else 10
        structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
        original_multiple_choice_output_format, args_compute_node_score, raw_dataset_size = load_supernatural_instructions_task(
            args)
    elif args.dataset_name == 'instruction-induction':
        structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
        original_multiple_choice_output_format, args_compute_node_score, raw_dataset_size = load_instruction_induction_task(
            args)
        args.max_new_tokens = 15
    else:
        assert False, "No custom loading function found for this dataset."

    return structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
           original_multiple_choice_output_format, args_compute_node_score, raw_dataset_size


def _value_assignment_is_valid(structured_prompt_format, global_constraints, value_assignment, allow_text_action_type):
    # A. copy structured_prompt_format to avoid modifying the original
    new_structured_prompt_format, new_global_constraints = \
        copy.deepcopy((structured_prompt_format, global_constraints))
    all_pointers = pointers_to_all_objects(new_structured_prompt_format) + new_global_constraints
    all_pointers_enumerated = [(e, i) for i, e in enumerate(all_pointers)]
    pointer_action_pairs = create_pointer_action_type_pairs(
        all_pointers_enumerated, allow_text_action_type=allow_text_action_type)

    # B. apply the value assignment
    value_assignments_ids = value_assignment_str_to_indices([value_assignment], pointer_action_pairs)[0]
    for (element, element_id, action_type), action_value_id in zip(pointer_action_pairs, value_assignments_ids):
        action_value, action_value_name = MAPPING_ALL_CATEGORIES[action_type][int(action_value_id)]
        element.update_field(action_type, action_value)

    # C. evaluate new node holistically
    return holistic_node_format_sanity_checks(new_structured_prompt_format)


def _sample_value_assignments(args):
    # load task [we might do it twice, but this first time is to load the structured_prompt_format]
    structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
        original_multiple_choice_output_format, args_compute_node_score, raw_dataset_size = _load_task(args)

    # sample nodes to evaluate if file has not been passed
    all_pointers = pointers_to_all_objects(structured_prompt_format) + global_constraints
    all_pointers_enumerated = [(e, i) for i, e in enumerate(all_pointers)]
    pointer_action_pairs = create_pointer_action_type_pairs(
        all_pointers_enumerated, allow_text_action_type=args.allow_text_action_type)

    action_value_options = []
    for a, b, action_type in pointer_action_pairs:
        action_value_options.append([f_name for f_value, f_name in MAPPING_ALL_CATEGORIES[action_type]])

    num_combinations = 1
    for e in action_value_options:
        num_combinations *= len(e)

    if num_combinations <= args.num_formats_to_analyze:
        valid_value_assignments = []
        for value_assignment in itertools.product(*action_value_options):
            if _value_assignment_is_valid(
                    structured_prompt_format, global_constraints, value_assignment, args.allow_text_action_type):
                valid_value_assignments.append(value_assignment)
    else:
        valid_value_assignments = set()
        while len(valid_value_assignments) < args.num_formats_to_analyze:
            value_assignment = [random.choice(sublist) for sublist in action_value_options]
            if _value_assignment_is_valid(
                    structured_prompt_format, global_constraints, value_assignment, args.allow_text_action_type):
                valid_value_assignments.add(tuple(value_assignment))
        valid_value_assignments = [list(e) for e in valid_value_assignments]

    # set an order in which to shuffle the whole dataset (including demonstrations)
    dataset_ordered_ids = list(range(raw_dataset_size))
    random.shuffle(dataset_ordered_ids)

    return valid_value_assignments, dataset_ordered_ids


def _generate_neighbor_value_assignment(value_assignment, idx_to_change, action_types):
    action_type_to_change = action_types[idx_to_change]
    neighbor_value_assignment = copy.copy(value_assignment)

    cur_value = value_assignment[idx_to_change]
    new_value = cur_value
    while new_value == cur_value:
        new_value = random.choice(MAPPING_ALL_CATEGORIES[action_type_to_change])[1]
    neighbor_value_assignment[idx_to_change] = new_value
    return neighbor_value_assignment


def _sample_value_assignments_edges(args):
    # load task [we might do it twice, but this first time is to load the structured_prompt_format]
    structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
    original_multiple_choice_output_format, args_compute_node_score, raw_dataset_size = _load_task(args)

    # sample nodes to evaluate if file has not been passed
    all_pointers = pointers_to_all_objects(structured_prompt_format) + global_constraints
    all_pointers_enumerated = [(e, i) for i, e in enumerate(all_pointers)]
    pointer_action_pairs = create_pointer_action_type_pairs(
        all_pointers_enumerated, allow_text_action_type=args.allow_text_action_type)

    action_value_options = []
    action_types = []
    for a, b, action_type in pointer_action_pairs:
        action_value_options.append([f_name for f_value, f_name in MAPPING_ALL_CATEGORIES[action_type]])
        action_types.append(action_type)

    valid_value_assignments = []
    while len(valid_value_assignments) < args.num_edges_to_analyze * 2:
        value_assignment = [random.choice(sublist) for sublist in action_value_options]

        # generate value assignment with only one difference w.r.t. the current one (an "edge")
        # we decide which one to change using round robin
        idx_to_change = (len(valid_value_assignments) // 2) % len(action_types)
        neighbor_value_assignment = _generate_neighbor_value_assignment(value_assignment, idx_to_change, action_types)
        if tuple(value_assignment) in valid_value_assignments or \
                tuple(neighbor_value_assignment) in valid_value_assignments:
            continue

        if _value_assignment_is_valid(structured_prompt_format, global_constraints, value_assignment,
                                      args.allow_text_action_type) and \
                _value_assignment_is_valid(structured_prompt_format, global_constraints, neighbor_value_assignment,
                                           args.allow_text_action_type):
            valid_value_assignments.append(tuple(value_assignment))
            valid_value_assignments.append(tuple(neighbor_value_assignment))

    valid_value_assignments = [list(e) for e in valid_value_assignments]

    # set an order in which to shuffle the whole dataset (including demonstrations)
    dataset_ordered_ids = list(range(raw_dataset_size))
    random.shuffle(dataset_ordered_ids)

    return valid_value_assignments, dataset_ordered_ids


def _sample_value_assignment_paths(args, existing_value_assignments):
    # load task [we might do it twice, but this first time is to load the structured_prompt_format]
    structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
    original_multiple_choice_output_format, args_compute_node_score, raw_dataset_size = _load_task(args)

    # sample nodes to evaluate if file has not been passed
    all_pointers = pointers_to_all_objects(structured_prompt_format) + global_constraints
    all_pointers_enumerated = [(e, i) for i, e in enumerate(all_pointers)]
    pointer_action_pairs = create_pointer_action_type_pairs(
        all_pointers_enumerated, allow_text_action_type=args.allow_text_action_type)

    action_value_options = []
    action_types = []
    for a, b, action_type in pointer_action_pairs:
        action_value_options.append([f_name for f_value, f_name in MAPPING_ALL_CATEGORIES[action_type]])
        action_types.append(action_type)

    valid_value_assignments = []
    for value_assignment_0 in existing_value_assignments:
        found_valid_path = False
        while not found_valid_path:
            idx_to_change_1 = random.randrange(len(action_types))
            value_assignment_1 = _generate_neighbor_value_assignment(value_assignment_0, idx_to_change_1, action_types)

            idx_to_change_2 = random.randrange(len(action_types))
            value_assignment_2 = _generate_neighbor_value_assignment(value_assignment_1, idx_to_change_2, action_types)

            if len({tuple(value_assignment_0), tuple(value_assignment_1), tuple(value_assignment_2)}) != 3:
                continue

            if _value_assignment_is_valid(structured_prompt_format, global_constraints, value_assignment_1,
                                          args.allow_text_action_type) and \
                    _value_assignment_is_valid(structured_prompt_format, global_constraints, value_assignment_2,
                                               args.allow_text_action_type):
                valid_value_assignments.append(tuple(value_assignment_1))
                valid_value_assignments.append(tuple(value_assignment_2))
                found_valid_path = True
                print([tuple(value_assignment_0), tuple(value_assignment_1), tuple(value_assignment_2)])

    return valid_value_assignments


def _get_task_filename_to_print(args):
    if args.dataset_name == 'natural-instructions':
        task_filename = args.task_filename
        to_print = task_filename.split("_")[0]
        to_print = to_print[:-5] if to_print.endswith('.json') else to_print
    elif args.dataset_name == 'instruction-induction':
        task_filename = args.task_filename.replace('_', '-')
        to_print = task_filename[:-5] if task_filename.endswith('.json') else task_filename
    else:
        assert False, "Dataset not supported."
    return to_print


def _get_output_filename(args):
    scoring_type = 'rankscore' if args.evaluation_metric == 'probability_ranking' else 'genscore'
    use_4bit_str = '_4bit' if args.use_4bit else ''
    if args.evaluation_type == 'format_spread':
        filename = f'metadataholistic_{disable_text_action_type}_{scoring_type}_{task_filename_to_print}_search_model_{args.model_name.split("/")[-1]}_nshot_{args.n_shot}_numnodes_{args.num_formats_to_analyze}_numsamples_{args.num_samples}_thompson_numformats_{args.num_formats_format_spread}_batch_{args.batch_size_format_spread}_maxcalls_{args.budget_format_spread}{use_4bit_str}'
    elif args.num_formats_to_analyze:
        filename = f'metadataholistic_{disable_text_action_type}_{scoring_type}_{task_filename_to_print}_search_model_{args.model_name.split("/")[-1]}_nshot_{args.n_shot}_numnodes_{args.num_formats_to_analyze}_numsamples_{args.num_samples}{use_4bit_str}'
    elif args.num_edges_to_analyze:
        filename = f'metadataholistic_{disable_text_action_type}_{scoring_type}_{task_filename_to_print}_search_model_{args.model_name.split("/")[-1]}_nshot_{args.n_shot}_numedges_{args.num_edges_to_analyze}_numsamples_{args.num_samples}{use_4bit_str}'
    elif args.extend_graph_paths_from_file:
        # it is exactly like args.num_formats_to_analyze, but from a specific file
        filename = f'metadataholistic_{disable_text_action_type}_{scoring_type}_{task_filename_to_print}_search_model_{args.model_name.split("/")[-1]}_nshot_{args.n_shot}_numnodes-extension_{num_new_paths}_numsamples_{args.num_samples}{use_4bit_str}'
    else:
        assert False, "No output file format defined."

    return filename


if __name__ == "__main__":
    # python main.py --task_filename singular_to_plural.json --num_formats_to_analyze 5 --batch_size_llm 10 --model_name "meta-llama/Llama-2-7b-hf" --n_shot 5

    parser = argparse.ArgumentParser()

    # params to load a task
    parser.add_argument('--task_filename', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, choices=['natural-instructions', 'instruction-induction'])

    # params to create or load a set of formats to evaluate
    parser.add_argument('--num_formats_to_analyze', type=int, default=None)
    parser.add_argument('--num_edges_to_analyze', type=int, default=None, help='Use for atomic changes experiment.')
    parser.add_argument('--extend_graph_paths_from_file', type=str, default=None,
                        help='Use solely for non-monotonic paths experiment. Only include filename of old 499 samples file.')
    parser.add_argument('--nodes_to_evaluate_filepath', type=str, default=None,
                        help='Filepath containing the formats to evaluate. If no file is passed, '
                             'the script loads the default file if available, or creates it if it does not exist.')

    # params to set up evaluation settings
    parser.add_argument('--num_samples', type=int, default=1000, help='Maximum number of samples to evaluate for each format.')
    parser.add_argument('--evaluation_metric', choices=['exact_prefix_matching', 'probability_ranking', 'rouge', 'bertscore'])
    parser.add_argument('--evaluation_type', type=str, choices=['full', 'format_spread'],
                        help='Determines how to evaluate the array of formats defined. '
                             'Options are full evaluation of each node, or use Thompson Sampling to quickly find the format spread.')
    parser.add_argument('--n_shot', type=int, default=1)

    # params to load models and how to use them
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--batch_size_llm', type=int, default=2, help='Batch size to call the LLM.')
    parser.add_argument('--use_4bit', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='/gscratch/xlab/msclar/.cache')

    # FormatSpread-specific parameters, corresponding to Thompson Sampling
    parser.add_argument('--num_formats_format_spread', type=int, default=320, help='Number of formats to sample.')
    parser.add_argument('--batch_size_format_spread', type=int, default=20, help='Batch size used by FormatSpread when running Thompson Sampling. Only used with `--evaluation_type format_spread`')
    parser.add_argument('--budget_format_spread', type=int, default=40000, help='Maximum number of model calls allowed when exploring best and worst formats, i.e. budget for thompson sampling. Only used with `--evaluation_type format_spread`')

    # saving parameters
    parser.add_argument('--output_dir', type=str, default='data')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # note: earlier version of the code allowed to vary the text for synonyms, but that has been deprecated
    args.disable_text_action_type = True
    args.allow_text_action_type = not args.disable_text_action_type
    disable_text_action_type = 'textdisabled'

    if args.model_name in ['gpt-5', 'gpt-3.5-turbo']:
        args.use_gpt3 = True
        args.gpt3_engine = args.model_name
    else:
        args.use_gpt3 = False
        args.gpt3_engine = None

    assert args.num_samples % args.batch_size_llm == 0  # for simplicity
    assert args.batch_size_format_spread % args.batch_size_llm == 0 if args.evaluation_type == 'format_spread' else True   # for simplicity
    assert len(
        [e for e in [args.num_formats_to_analyze, args.num_edges_to_analyze, args.extend_graph_paths_from_file] if
         e is not None]) == 1
    if args.extend_graph_paths_from_file is not None:
        assert args.task_filename in args.extend_graph_paths_from_file

    demonstrations_filename_suffix = ''

    # 0. load sampled formats (or sample formats if they are not available)
    task_filename_to_print = _get_task_filename_to_print(args)
    if args.num_formats_to_analyze:
        filepath = os.path.join(args.output_dir,
                                f'holistic_random_sample_{task_filename_to_print}_nodes_{args.num_formats_to_analyze}_{disable_text_action_type}.json')
        if args.nodes_to_evaluate_filepath:
            tmp = json.load(open(args.nodes_to_evaluate_filepath, 'r'))
            valid_value_assignments = tmp['valid_value_assignments']
            dataset_ordered_ids = tmp['dataset_ordered_ids']
        elif os.path.exists(filepath):
            tmp = json.load(open(filepath, 'r'))
            valid_value_assignments = tmp['valid_value_assignments']
            dataset_ordered_ids = tmp['dataset_ordered_ids']
        else:
            valid_value_assignments, dataset_ordered_ids = _sample_value_assignments(args)
            json.dump({'valid_value_assignments': valid_value_assignments,
                       'dataset_ordered_ids': dataset_ordered_ids}, open(filepath, 'w'))
            print('Created sample and stored it in', filepath)

        args.dataset_ordered_ids = dataset_ordered_ids  # used in data loading
    elif args.num_edges_to_analyze:
        filepath = os.path.join(args.output_dir,
                                f'holistic_random_sample_{task_filename_to_print}_edges_{args.num_edges_to_analyze}_{disable_text_action_type}.json')
        if args.nodes_to_evaluate_filepath:
            tmp = json.load(open(args.nodes_to_evaluate_filepath, 'r'))
            valid_value_assignments = tmp['valid_value_assignments']
            dataset_ordered_ids = tmp['dataset_ordered_ids']
        elif os.path.exists(filepath):
            tmp = json.load(open(filepath, 'r'))
            valid_value_assignments = tmp['valid_value_assignments']
            dataset_ordered_ids = tmp['dataset_ordered_ids']
        else:
            valid_value_assignments, dataset_ordered_ids = _sample_value_assignments_edges(args)
            json.dump({'valid_value_assignments': valid_value_assignments,
                       'dataset_ordered_ids': dataset_ordered_ids}, open(filepath, 'w'))
            print('Created sample and stored it in', filepath)

        args.dataset_ordered_ids = dataset_ordered_ids  # used in data loading
    elif args.extend_graph_paths_from_file:
        """
        We have a file with already analyzed nodes (~499) and we want to sample a bunch of paths v_1->v_2->v_3.
        We cap it to 300*2 new nodes to analyze.
        """
        num_new_paths = 300
        filepath = os.path.join(
            args.output_dir,
            f'extension_{num_new_paths}_paths_from_{args.extend_graph_paths_from_file}'
        )

        if os.path.exists(filepath):
            tmp = json.load(open(filepath, 'r'))
            valid_value_assignments = tmp['valid_value_assignments']
            dataset_ordered_ids = tmp['dataset_ordered_ids']
        else:
            assert os.path.exists(os.path.join(args.output_dir, args.extend_graph_paths_from_file))
            tmp = json.load(open(os.path.join(args.output_dir, args.extend_graph_paths_from_file), 'r'))
            existing_value_assignments = tmp['valid_value_assignments']
            dataset_ordered_ids = tmp['dataset_ordered_ids']

            assert len(existing_value_assignments) >= num_new_paths
            valid_value_assignments = _sample_value_assignment_paths(args, existing_value_assignments[:num_new_paths])
            json.dump({'valid_value_assignments': valid_value_assignments,
                       'dataset_ordered_ids': dataset_ordered_ids}, open(filepath, 'w'))

    # 1. load task
    structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
    original_multiple_choice_output_format, args_compute_node_score, _ = _load_task(args)
    print('Task loaded.')

    # 1.b. check that the evaluation metric is reasonable
    # Specifically, we can compute probability ranking metric only if the task is a classification task
    output_options_size = len(set([e for d in args_compute_node_score['dataset'] for e in d['output']]))
    assert output_options_size < 10 if args.evaluation_metric == 'probability_ranking' else True

    # 2. load model
    model, tokenizer, model_will_repeat_input = _load_model(args)
    print('Model loaded.')

    args_compute_node_score['model'] = model
    args_compute_node_score['tokenizer'] = tokenizer
    args_compute_node_score['model_will_repeat_input'] = model_will_repeat_input
    args_compute_node_score['args'].use_gpt3 = args.use_gpt3
    args_compute_node_score['args'].gpt3_engine = args.gpt3_engine

    # 3. evaluate formats
    print('Start evaluation of formats.')
    if args.evaluation_type == 'format_spread':
        search_tree = ThompsonSamplingAlgorithmAmongPrompts(
            structured_prompt_format,
            global_constraints,
            extra_params_structured_prompt_format,
            args_compute_node_score=args_compute_node_score,
            objective='lowest_accuracy',  # dummy in this mode
            allow_text_action_type=args.allow_text_action_type,
            original_multiple_choice_output_format=original_multiple_choice_output_format
        )

        search_tree.main(
            value_assignments=valid_value_assignments[:args.num_formats_format_spread + 1],
            batch_size=args.batch_size_format_spread,
            num_formats=args.num_formats_format_spread,
            max_allowed_number_of_model_calls=args.budget_format_spread
        )

    elif args.evaluation_type == 'full':
        # exhaustive node evaluation
        search_tree = GeneticAlgorithmAmongPrompts(
            structured_prompt_format,
            global_constraints,
            extra_params_structured_prompt_format,
            args_compute_node_score=args_compute_node_score,
            objective='lowest_accuracy',  # dummy in this mode
            allow_text_action_type=args.allow_text_action_type,
            original_multiple_choice_output_format=original_multiple_choice_output_format
        )

        print('valid_value_assignments', valid_value_assignments)
        search_tree.main(
            value_assignments=valid_value_assignments,
            num_samples_to_test=args.num_samples
        )

        acc = search_tree.list_node_accuracies()
        print('Best Node:', acc[0])
        print('Worst Node:', acc[-1])

    search_tree.save(os.path.join(args.output_dir, f'{_get_output_filename(args)}.json'))
