import json
import os
import random
import re

from grammar_definition import flatten, _one_text_field
from parsing_supernatural_instructions_tasks import SUPERNATURAL_INSTRUCTIONS_TASKS_WITH_NO_FORMAT, \
    create_initial_structured_prompt_format

SUPERNATURAL_INSTRUCTIONS_DIRECTORY = '../natural-instructions/tasks'
INSTRUCTION_INDUCTION_DIRECTORY = '../instruction-induction'

STRING_ALL_CHARACTERS_FOR_REGEX_MATCHING = "([A-Za-z0-9α-ωΑ-Ω“”‘’′`,.…'-–—−:∶\(\)\[\]{}/%\?\!\\\" ;$≤≥≠†€₹→≡~∨⊃·°•∃∀ʻ&⁄\_#\\n𝑆𝑚√𝑠𝑁𝐴𝑒𝑅𝑇ι⟩⟨›‹ου‖♥‰�龍►➥™,‚∼⋅]+)"
random.seed(0)


def extract_regex(prompt_format):
    prompt_format_original = prompt_format.replace('<|text|>', '<text>')  # pipe cannot be used for regex
    regex_sentence_extractor_str = re.escape(prompt_format_original).replace(
        '<text>', STRING_ALL_CHARACTERS_FOR_REGEX_MATCHING)
    regex_sentence_extractor_str = '^' + regex_sentence_extractor_str + '$'
    regex_sentence_extractor = re.compile(regex_sentence_extractor_str)
    return regex_sentence_extractor


def _extract_fields_from_dataset(regex_sentence_extractor_dict, dataset, num_samples):
    input_fields_list = []
    outputs_list = []

    # tells us which key in regex_sentence_extractor_dict matched, useful for knowing
    # which format version (with number of enumerations) to apply later
    regex_key_idx_list = []
    selected_ids = []

    for i, entry in enumerate(dataset):
        if len(input_fields_list) == num_samples:
            break

        # we skip data points that we could not parse:
        # sometimes even in the same task, the spacing is not respected (probably due to manual errors)
        # note: we process possible regexes from longest to shortest, because often a template with two fields would
        # match a string that actually has five fields
        input_fields, regex_key_idx = None, None
        for regex_key_idx, regex_sentence_extractor in sorted(regex_sentence_extractor_dict.items(), reverse=True):
            input_fields = re.search(regex_sentence_extractor, entry['input'])
            if input_fields:
                break
        if not input_fields:
            print(f"WARNING: data point {i} ({entry['input']}) was not able to be processed.")
            print('CHARACTERS USED:', [e for e in set(entry['input']) if not re.match(STRING_ALL_CHARACTERS_FOR_REGEX_MATCHING, e)])
            continue

        input_fields = input_fields.groups()
        input_fields_list.append(input_fields)
        regex_key_idx_list.append(regex_key_idx)

        outputs_list.append(entry['output'])
        selected_ids.append(i)

    return input_fields_list, outputs_list, regex_key_idx_list, selected_ids


def _load_raw_dataset_supernatural_instructions(task_filename):
    # find filename based on task_filename
    task_filenames = [f for f in os.listdir(SUPERNATURAL_INSTRUCTIONS_DIRECTORY) if task_filename in f]
    assert len(task_filenames) == 1, f"{task_filenames} should be length = 1"
    task_filename = task_filenames[0]

    filepath = os.path.join(SUPERNATURAL_INSTRUCTIONS_DIRECTORY, task_filename)
    raw_dataset = json.load(open(filepath, 'r'))
    return raw_dataset


def set_up_prompt_variation_exploration_without_extra_files(
        args,
        structured_prompt_format,
        extra_params_structured_prompt_format,
        instruction=None
):
    """
    Mel notes: currently
        choosing demonstrations;
        loading dataset;
        potentially adding "answer" field; create
        regex extracting fields
    """

    raw_dataset = _load_raw_dataset_supernatural_instructions(args.task_filename)
    demonstration_definition = raw_dataset['Definition'][0] if instruction is None else instruction

    raw_dataset = raw_dataset['Instances']
    if hasattr(args, 'dataset_ordered_ids') and args.dataset_ordered_ids:
        assert len(args.dataset_ordered_ids) == len(raw_dataset)
        print('args.dataset_ordered_ids!')
        raw_dataset = [raw_dataset[i] for i in args.dataset_ordered_ids]
    else:
        random.shuffle(raw_dataset)

    demonstrations = raw_dataset[:10]
    dataset = [entry for entry in raw_dataset[10:]]

    if extra_params_structured_prompt_format and extra_params_structured_prompt_format.get('enumeration_length_range'):
        regex_sentence_extractor_dict = {}
        for e in range(*extra_params_structured_prompt_format.get('enumeration_length_range')):
            prompt_format_original = flatten(structured_prompt_format.solve({'enumeration_length': e}))
            regex_sentence_extractor_dict[e] = extract_regex(prompt_format_original)
    else:
        regex_sentence_extractor = extract_regex(flatten(structured_prompt_format.solve()))
        regex_sentence_extractor_dict = {None: regex_sentence_extractor}  # None because there is no length

    return demonstration_definition, dataset, regex_sentence_extractor_dict, demonstrations, len(raw_dataset)


def setup_demonstrations(args, regex_sentence_extractor_dict, demonstrations):

    demos_fields_list, demonstrations_outputs, demos_regex_key_idx_list, _ = _extract_fields_from_dataset(
        regex_sentence_extractor_dict, demonstrations, num_samples=args.n_shot)

    if len(demos_fields_list) != args.n_shot:
        print("Insufficient n-shot demos.")
        print(len(demos_fields_list))
        assert False, f"{len(demos_fields_list)} != {args.n_shot}"
        exit(1)

    file_suffix = ''
    return demos_fields_list, demonstrations_outputs, demos_regex_key_idx_list, file_suffix


def load_supernatural_instructions_task(args):
    """
    All logic for loading the dataset, extracting the original formatting from the text.

    PRECOMPUTE
    1. Load model and tokenizer (OK)
    2. Detect regex to extract fields from dataset (currently from external file, but it could be from the initial structure)
    3. Extract formatting from dataset (keep a set of fields)
    4. Extract desired few shot examples and extract their formatting (keep a set of fields)

    Args params needed:

    args.task_filename
    args.num_samples
    args.n_shot
    Plus the ones needed for uses of args_compute_node_score
    """

    # SuperNaturalInstructions Tasks without a defined format
    if any(t in args.task_filename for t in SUPERNATURAL_INSTRUCTIONS_TASKS_WITH_NO_FORMAT):
        raw_dataset = _load_raw_dataset_supernatural_instructions(args.task_filename)
        demonstration_definition = raw_dataset['Definition'][0]
        raw_dataset = raw_dataset['Instances']
        return _setup_non_formatted_dataset_with_one_field_only(args, raw_dataset, demonstration_definition)

    # Parse Formatted SuperNaturalInstructions Tasks
    structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
        instruction, original_multiple_choice_output_format = create_initial_structured_prompt_format(args)
    demonstration_definition, dataset, regex_sentence_extractor_dict, demonstrations, raw_dataset_size = \
        set_up_prompt_variation_exploration_without_extra_files(
            args, structured_prompt_format, extra_params_structured_prompt_format, instruction)
    demonstration_definition = demonstration_definition if instruction is None else instruction

    input_fields_list, _, regex_key_idx_list, selected_dataset_ids = _extract_fields_from_dataset(
        regex_sentence_extractor_dict, dataset, num_samples=args.num_samples)

    demos_fields_list, demonstrations_outputs, demos_regex_key_idx_list, demonstrations_filename_suffix = \
        setup_demonstrations(args, regex_sentence_extractor_dict, demonstrations)

    args_compute_node_score = {
        'args': args,
        'dataset': dataset,
        'input_fields_list': input_fields_list,
        'regex_key_idx_list': regex_key_idx_list,  # tells us which of the options of enumeration quantities applies
        'selected_dataset_ids': selected_dataset_ids,
        'demos_fields_list': demos_fields_list,
        'demonstrations_outputs': demonstrations_outputs,
        'demos_regex_key_idx_list': demos_regex_key_idx_list,
        # tells us which of the options of enumeration quantities applies
        'demonstration_definition': demonstration_definition,
    }

    return structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
        original_multiple_choice_output_format, args_compute_node_score, raw_dataset_size


def _setup_non_formatted_dataset_with_one_field_only(args, raw_dataset, demonstration_definition):
    # set up initial formatting
    structured_prompt_format, global_constraints = _one_text_field('Input', answer_field_text='Output', chosen_space='\n')
    extra_params_structured_prompt_format = None
    original_multiple_choice_output_format = None

    if hasattr(args, 'dataset_ordered_ids') and args.dataset_ordered_ids:
        assert len(args.dataset_ordered_ids) == len(raw_dataset)
        print('args.dataset_ordered_ids!')
        raw_dataset = [raw_dataset[i] for i in args.dataset_ordered_ids]
    else:
        random.shuffle(raw_dataset)

    # set up dataset & demonstrations with the same fields and formatting as SuperNatural Instructions
    demonstrations = raw_dataset[:10]
    dataset = [entry for entry in raw_dataset[10:]]

    demos_fields_list = [tuple([example['input']]) for example in demonstrations][:args.n_shot]
    demonstrations_outputs = [example['output'] for example in demonstrations][:args.n_shot]

    input_fields_list = [tuple([example['input']]) for example in dataset][:args.num_samples]
    selected_dataset_ids = list(range(len(input_fields_list)))

    args_compute_node_score = {
        'args': args,
        'dataset': dataset,
        'input_fields_list': input_fields_list,
        'regex_key_idx_list': [None] * len(input_fields_list),  # setting to None because there is only one format option (no enumeration length variation)
        'selected_dataset_ids': selected_dataset_ids,
        'demos_fields_list': demos_fields_list,
        'demonstrations_outputs': demonstrations_outputs,
        'demos_regex_key_idx_list': [None] * len(demonstrations_outputs),  # setting to None because there is only one format option (no enumeration length variation)
        'demonstration_definition': demonstration_definition,
    }

    return structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
        original_multiple_choice_output_format, args_compute_node_score, len(raw_dataset)


def load_instruction_induction_task(args):
    """
    This dataset doesn't have a pre-defined format to extract like SuperNatural Instructions.
    We will use the formatting that APE has used as a starting point.

    We'll generate the equivalent structures as the ones generated in SuperNaturalInstructions.

    Instructions: https://github.com/orhonovich/instruction-induction/blob/main/data/annotations/antonyms.json
    I-O: https://github.com/orhonovich/instruction-induction/tree/main/data/raw/induce
    """

    # load datasets
    # task_filename = f"{task_name}.json"
    instructions = json.load(open(os.path.join(INSTRUCTION_INDUCTION_DIRECTORY, 'data', 'annotations', args.task_filename), 'r'))
    instructions = instructions['annotations']
    print('instructions', instructions)

    raw_dataset = json.load(open(os.path.join(INSTRUCTION_INDUCTION_DIRECTORY, 'data', 'raw', 'induce', args.task_filename), 'r'))
    raw_dataset = list(raw_dataset['examples'].values())
    raw_dataset = [{'input': entry['input'], 'output': [entry['output']]} for entry in raw_dataset]

    # chose best instruction with some criterion (long, is properly cased to begin with)
    demonstration_definition = sorted([inst for inst in instructions if inst[0].isupper()], reverse=True, key=len)[0]

    return _setup_non_formatted_dataset_with_one_field_only(args, raw_dataset, demonstration_definition)
