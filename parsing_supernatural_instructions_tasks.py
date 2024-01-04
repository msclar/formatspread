from grammar_definition import SpacingBetweenPromptComponents, SharedPropertyAmongPrompts, \
    NewEnumerationPromptFormat, SimplePromptFormat, NoTextPromptFormat, _one_text_field, _two_text_fields

SOCIAL_GOOD_TASK_IDS = [
    'task137_',  # 2-Choice output, prompt formatted -- 379 samples
    'task327_', 'task333_', 'task335_', 'task337_',
    # prompt formatted, binary classification -- +2000 samples, FIXME allow emojis in regex matching
    'task905_',  # prompt formatted, classification -- +2000 samples, no parsing errors
    'task320_',  # prompt formatted-ish, classification
    'task1502_', 'task1503_', 'task1504_',  # no prompt format: classification, classification, generation
    'task1664_',  # no prompt format: set of words as output
    'task1669_', 'task1670_',  # no prompt format, long generation but well defined!
    'task1720_', 'task1725_',  # no prompt format, binary classification
    'task904_',  # no prompt format, classification,
    'task277_', 'task278_', 'task279_', 'task280_', 'task316_', 'task317_', 'task318_', 'task319_', 'task320_',
    'task321_',
    'task108_',
    'task322_', 'task323_', 'task324_', 'task325_', 'task326_', 'task327_', 'task328_',
    'task1604_', 'task1605_', 'task1606_', 'task1607_',
    'task1721_', 'task1722_', 'task1723_', 'task1724_',
    'task607_', 'task608_', 'task609_', 'task286_'
]

SUPERNATURAL_INSTRUCTIONS_TASKS_WITH_NO_FORMAT = [
    'task1502_', 'task1503_', 'task1504_',  # no prompt format: classification, classification, generation
    'task1664_',  # no prompt format: set of words as output
    'task1669_', 'task1670_',  # no prompt format, long generation but well defined!
    'task1720_', 'task1725_',  # no prompt format, binary classification
    'task904_',  # no prompt format, classification
    'task108_',
    'task1604_', 'task1605_', 'task1606_', 'task1607_',
    'task1721_', 'task1722_', 'task1723_', 'task1724_',
    'task607_', 'task608_', 'task609_', 'task286_',
    'task1149_', 'task1189_'
]

FORMATTED_MULTIPLE_CHOICE_SUPERNATURAL_INSTRUCTIONS_TASKS = [  # ends up being one-field format
    'task065_', 'task1297_', 'task084_', 'task697_', 'task729_',
    'task1380_', 'task1381_', 'task309_', 'task1431_', 'task220_', 'task1612_', 'task190_', 'task1347_',
    'task069_', 'task070_',
    'task137_', 'task138_', 'task139_', 'task140_', 'task296_', 'task297_', 'task118_', 'task1135_',
    'task1424_', 'task1423_', 'task1422_', 'task1421_', 'task1420_', 'task1419_',
    'task1678_', 'task385_', 'task580_', 'task214_', 'task213_'
]

FORMATTED_TWO_TEXT_FIELDS_SUPERNATURAL_INSTRUCTIONS_TASKS = \
    ['task1661_', 'task027_', 'task136_', 'task021_', 'task018_', 'task020_', 'task740_',
     'task1366_', 'task1162_', 'task1587_', 'task491_', 'task492_', 'task050_', 'task1387_',
     'task1186_', 'task1283_', 'task1284_', 'task905_', 'task501_']

FORMATTED_ONE_TEXT_FIELDS_SUPERNATURAL_INSTRUCTIONS_TASKS = [
    'task155_', 'task158_', 'task161_', 'task163_', 'task162_', 'task322_', 'task323_',
    'task324_', 'task325_', 'task326_', 'task327_', 'task328_', 'task333_', 'task335_',
    'task337_', 'task277_', 'task278_', 'task279_', 'task280_', 'task316_', 'task317_',
    'task113_', 'task114_']

FORMATTED_SOME_TEXT_FIELDS_SUPERNATURAL_INSTRUCTIONS_TASKS = [
    'task318_', 'task319_', 'task320_', 'task321_', 'task133_']

OPEN_GENERATION_SUPERNATURAL_INSTRUCTIONS_TASKS = [
    'task240_', 'task845_', 'task348_', 'task389_', 'task443_', 'task223_',
    'task105_', 'task1401_', 'task040_', 'task067_', 'task071_', 'task072_',
    'task1326_', 'task037_', 'task038_', 'task1613_', 'task216_']


def create_initial_structured_prompt_format(args):
    structured_prompt_format = None
    global_constraints = []
    extra_params_structured_prompt_format = None
    instruction = None
    original_multiple_choice_output_format = None

    if any(t in args.task_filename for t in ['task1661_', 'task027_']):
        structured_prompt_format, global_constraints = _two_text_fields('Passage', 'Question')

    elif any(t in args.task_filename for t in ['task136_', 'task021_', 'task018_', 'task020_', 'task740_']):
        structured_prompt_format, global_constraints = _two_text_fields('Sentence', 'Question')

    elif any(t in args.task_filename for t in ['task1366_']):
        structured_prompt_format, global_constraints = _two_text_fields('Paragraph', 'Claim')

    elif any(t in args.task_filename for t in ['task1162_']):
        structured_prompt_format, global_constraints = _two_text_fields('Paragraph', 'Title', chosen_space='\n ')

    elif any(t in args.task_filename for t in ['task1587_']):
        structured_prompt_format, global_constraints = _two_text_fields('Abstract', 'Title', chosen_space='. ')

    elif any(t in args.task_filename for t in ['task491_', 'task492_']):
        structured_prompt_format, global_constraints = _two_text_fields('Sentence', 'Question', chosen_space=' ')

    elif any(t in args.task_filename for t in ['task050_']):
        structured_prompt_format, global_constraints = _two_text_fields('Sentence', 'Question', chosen_space=' \n')

    elif any(t in args.task_filename for t in ['task1387_']):
        structured_prompt_format, global_constraints = _two_text_fields('Premise', 'Hypothesis', chosen_space=' <sep> ')

    elif any(t in args.task_filename for t in ['task1186_', 'task1283_', 'task1284_']):
        structured_prompt_format, global_constraints = _two_text_fields(
            'System Reference', 'Original Reference', chosen_space='\n')

    elif any(t in args.task_filename for t in ['task190_', 'task1347_']):
        # note: output is not one of the enumerations!

        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                NewEnumerationPromptFormat('Sentence', 2, chosen_separator=': ', chosen_space=' ',
                                           chosen_item_wrapper=lambda x: f"{x}",
                                           text_descriptor_fn_owner=text_descriptor_fn, object_name='enum1'),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif any(t in args.task_filename for t in ['task1612_']):
        # note: output is not one of the enumerations!
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                NewEnumerationPromptFormat('sentence', 2, chosen_separator=': ', chosen_separator_text_and_option='_',
                                           chosen_space=' ', chosen_item_wrapper=lambda x: f"{x}",
                                           chosen_number_format=lambda x: chr(ord('A') + x),
                                           text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif any(t in args.task_filename for t in ['task905_']):
        structured_prompt_format, global_constraints = _two_text_fields('Tweet', 'Label', chosen_space='\n')

    elif any(t in args.task_filename for t in ['task155_', 'task158_', 'task161_', 'task163_', 'task162_']):
        # msclar: these are counting tasks
        structured_prompt_format, global_constraints = _one_text_field('Sentence', chosen_space='\n')

    elif any(t in args.task_filename for t in
             ['task322_', 'task323_', 'task324_', 'task325_', 'task326_', 'task327_', 'task328_']):
        # msclar: these are counting tasks
        structured_prompt_format, global_constraints = _one_text_field('Comment', chosen_space='\n')

    elif any(t in args.task_filename for t in ['task333_', 'task335_', 'task337_']):
        # msclar: these are counting tasks
        structured_prompt_format, global_constraints = _one_text_field('Post', chosen_space='\n')

    elif any(t in args.task_filename for t in ['task277_', 'task278_']):
        structured_prompt_format, global_constraints = _one_text_field('Context', chosen_space='\n')

    elif any(t in args.task_filename for t in ['task279_', 'task280_', 'task316_', 'task317_']):
        structured_prompt_format, global_constraints = _one_text_field('Passage', chosen_space='\n')

    elif any(t in args.task_filename for t in ['task113_', 'task114_']):
        structured_prompt_format, global_constraints = _one_text_field('Sentence', chosen_space='\n')

    elif any(t in args.task_filename for t in ['task318_', 'task319_', 'task320_', 'task321_']):
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Target', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                NoTextPromptFormat(),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n'
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task501_' in args.task_filename:
        # ((0.39, 0.37, 100), 'CLAIM : {}. POST : {}', 'CLAIM : {}. POST : {}. ANSWER : {}')
        # CLAIM : <text>. POST : <text>
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ' : '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x.upper()}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Claim', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Post', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space='. '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task133_' in args.task_filename:
        # Sentence: <text>\n Reason: <text>\n Question: <text>
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Sentence', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Reason', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space='\n '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task220_' in args.task_filename:
        # Sentence 1: <text> Sentence 2: <text> Sentence 3: <text> Sentence 4: <text> Sentence 5: <text> Choices: a. <text> b. <text>

        instruction = "In this task, you're given five sentences, numbered {enum0_1} through {enum0_5}, and two options {enum1_1} and {enum1_2} for possible titles for the story. Your job is to choose the title that better fits the story. Indicate your choice by '{enum1_1}' or '{enum1_2}'."
        original_multiple_choice_output_format = '{enum1}'

        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)

        #  chosen_space = SharedPropertyAmongPrompts({'space': ', '}, None)  # FIXME allow to jointly change these two spaces.
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                NewEnumerationPromptFormat('Sentence', 5, chosen_separator_owner=chosen_separator, chosen_space=' ',
                                           chosen_item_wrapper=lambda x: f"{x}",
                                           text_descriptor_fn_owner=text_descriptor_fn,
                                           object_name='enum0'),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Choices', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 2, chosen_space=' ', chosen_separator=' ',
                                                   chosen_item_wrapper=lambda x: f"{x}.",
                                                   chosen_number_format=lambda x: chr(ord('a') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' '
        )

    elif 'task1431_' in args.task_filename:
        instruction = "In this task, you are given a multiple-choice question about healthcare. Answer the question based on your information and classify your answers into '{enum1_1}', '{enum1_2}', '{enum1_3}', and '{enum1_4}'."
        original_multiple_choice_output_format = '{enum1}'

        # Question: <text>\n Options:  <1> <text> <2> <text> <3> <text> <4> <text> <5> <text>
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Options', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 5, chosen_space=' ', chosen_separator=' ',
                                                   chosen_item_wrapper=lambda x: f"<{x}>", object_name='enum1'),
                    ],
                    chosen_space=' '
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space='\n '
        )

        global_constraints = [chosen_separator, text_descriptor_fn]

    elif 'task309_' in args.task_filename:
        # Article: <text>\n Question: <text>\n Options: (A) <text> (B) <text> (C) <text> (D) <text>

        instruction = 'In this task, you\'re given an article, a question which often contains a blank and four options (associated with "{enum1_1}", "{enum1_2}", "{enum1_3}", "{enum1_4}"). Your task is to find the correct answer (from the given options) for the question from the given article and return one of the options from "{enum1_1}", "{enum1_2}", "{enum1_3}", and "{enum1_4}". Do not generate anything else apart from one of the following characters: "{enum1_1}", "{enum1_2}", "{enum1_3}", "{enum1_4}". There is only one correct answer for each question.'
        original_multiple_choice_output_format = '{enum1}'

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Article', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Options', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 4, chosen_space=' ', chosen_separator=' ',
                                                   chosen_item_wrapper=lambda x: f"({x})",
                                                   chosen_number_format=lambda x: chr(ord('A') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space='\n '
        )

        global_constraints = [text_descriptor_fn, chosen_separator]

    elif any(t in args.task_filename for t in ['task1380_', 'task1381_']):
        # Sentence: <text> Question: <text> (A) <text> (B) <text>

        if 'task1380_' in args.task_filename:
            instruction = "You are given a sentence, a question and two answer options ('{enum1_1}' and '{enum1_2}'). Your task is to find the correct option for the given question. Write down the answer index: '{enum1_1}' or '{enum1_2}'."
        elif 'task1381_' in args.task_filename:
            instruction = "You are given a sentence, a question and two answer options. Your task is to write down the index ('{enum1_1}' or '{enum1_2}') of the **incorrect** option for the given question."
        original_multiple_choice_output_format = '{enum1}'

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Sentence', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                NewEnumerationPromptFormat('', 2, chosen_space=' ', chosen_separator=' ',
                                           chosen_item_wrapper=lambda x: f"({x})",
                                           chosen_number_format=lambda x: chr(ord('A') + x), object_name='enum1'),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif any(t in args.task_filename for t in ['task697_', 'task729_']):
        # task697 = ((0.19230769230769232, 0.38461538461538464, 26), '{}\n(A){} (B){} (C){} (D){}', '{}\n(A){} (B){} (C){} (D){}\nAnswer: {}')
        # <text>\n(A)<text> (B)<text> (C)<text> (D)<text>

        # both tasks share instruction text
        instruction = 'You are given a question on formal logic. You are also given 4 answer options (associated with "{enum1_1}", "{enum1_2}", "{enum1_3}", "{enum1_4}"), out of which only one is correct. You need to answer the question by selecting the correct option. You should only answer with the choice letter, not the whole answer.'  # FIXME letter -> number when needed
        original_multiple_choice_output_format = '{enum1}'

        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('', ''),
                NewEnumerationPromptFormat('', 4, chosen_space=' ', chosen_separator='',
                                           chosen_item_wrapper=lambda x: f"({x})",
                                           chosen_number_format=lambda x: chr(ord('A') + x), object_name='enum1'),
                SimplePromptFormat('Answer', ': ', is_output_field=True)
            ],
            chosen_space='\n'
        )

    elif 'task903_' in args.task_filename:
        # Review: <text>\nPolarity: <text>
        instruction = "Given a hotel review and the corresponding polarity of review (i.e., Negative or Positive) identify if the polarity is correct. Write 'true' if it's correct, 'false' otherwise."

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Review', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Polarity', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space='\n'
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task084_' in args.task_filename:
        # Passage: Fact 1- <text>. Fact 2- <text>. Question: <text> Answer: <text>

        instruction = "You will be given a passage with an enumerated set of facts, a question of form 'Where is <person_name>?', and its answer. The task is to identify a supporting fact that is necessary to answer the question. The output would be the corresponding fact number."  # FIXME "number" -> "letter" when it should change
        original_multiple_choice_output_format = "{enum1}"

        min_elements, max_elements = 2, 15
        extra_params_structured_prompt_format = {'enumeration_length_range': (min_elements, max_elements + 1)}

        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Passage', None, chosen_separator_owner=chosen_separator,
                                           prompt_without_text=True, text_descriptor_fn_owner=text_descriptor_fn),
                        NewEnumerationPromptFormat('Fact', max_elements, chosen_separator='- ', chosen_space=' ',
                                                   chosen_item_wrapper=lambda x: f"{x}",
                                                   text_descriptor_fn_owner=text_descriptor_fn, object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Final Output', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task1297_' in args.task_filename:
        # Fact1: <text>, Fact2: <text>, Question: <text> (A) <text> (B) <text> (C) <text> (D) <text> (E) <text> (F) <text> (G) <text> (H) <text>
        instruction = 'In this task, you are given two facts, and a multiple-choice question. Based on the given facts, answer the question with index of the correct option (e.g, "{enum1_1}").'
        original_multiple_choice_output_format = "{enum1}"

        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                NewEnumerationPromptFormat('Fact', 2, chosen_separator=': ', chosen_separator_text_and_option='',
                                           chosen_space=', ', chosen_item_wrapper=lambda x: f"{x}",
                                           text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn),
                        NewEnumerationPromptFormat('', 8, chosen_separator=' ', chosen_space=' ',
                                                   chosen_item_wrapper=lambda x: f"({x})",
                                                   chosen_number_format=lambda x: chr(ord('A') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space=' '
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=', '
        )

        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task065_' in args.task_filename:
        # Sentence 1: <text>\n Sentence 3: <text>\n Sentence 4: <text>\n Sentence 5: <text>\n Option 1: <text>\n Option 2: <text>

        instruction = "In this task, you are given a short story consisting of exactly 5 sentences where the second sentence is missing. You are given two options and you need to select the one that best connects the first sentence with the rest of the story. Indicate your answer by 'Option {enum1_1}' if the first option is correct, otherwise 'Option {enum1_2}'. The incorrect option will change the subsequent storyline, so that at least one of the three subsequent sentences is no longer consistent with the story."
        original_multiple_choice_output_format = "Option {enum1}"  # Idea: save chosen_number_format from the initial text, and compute chosen_number_format^-1. Then it's just a lookup table from "Option a"->1, and then we apply the current function in chosen_number_format

        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                # [0, 2, 3, 4] -> [1, 3, 4, 5] because of indexing
                NewEnumerationPromptFormat('Sentence', [0, 2, 3, 4], chosen_separator=': ', chosen_space=' \n ',
                                           chosen_item_wrapper=lambda x: f"{x}",
                                           text_descriptor_fn_owner=text_descriptor_fn),
                NewEnumerationPromptFormat('Option', 2, chosen_separator=': ', chosen_space=' \n ',
                                           chosen_item_wrapper=lambda x: f"{x}",
                                           text_descriptor_fn_owner=text_descriptor_fn, object_name='enum1'),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif any(t in args.task_filename for t in ['task069_', 'task070_']):
        # Beginning: <text> Middle 1: <text> Middle 2: <text> Ending: <text>
        if 'task069_' in args.task_filename:
            instruction = "In this task, you will be shown a short story with a beginning, two potential middles, and an ending. Your job is to choose the middle statement that makes the story coherent / plausible by writing \"{enum1_1}\" or \"{enum1_2}\" in the output. If both sentences are plausible, pick the one that makes most sense."
        else:
            instruction = "In this task, you will be shown a short story with a beginning, two potential middles, and an ending. Your job is to choose the middle statement that makes the story incoherent / implausible by indicating \"{enum1_1}\" or \"{enum1_2}\" in the output. If both sentences are plausible, pick the one that makes less sense."

        original_multiple_choice_output_format = "{enum1}"

        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Beginning', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                NewEnumerationPromptFormat('Middle', 2, chosen_separator=': ', chosen_space=' ',
                                           chosen_number_format=lambda x: x + 1, chosen_item_wrapper=lambda x: x,
                                           text_descriptor_fn_owner=text_descriptor_fn, object_name='enum1'),
                SimplePromptFormat('Ending', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' '
        )
        # FIXME allow the two chosen_spaces to change jointly (enumeration + spacing)
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif any(t in args.task_filename for t in ['task137_', 'task138_', 'task139_', 'task140_']):
        # Prompt: <text> \nCompletion A: <text> \nCompletion B: <text>

        if 'task137_' in args.task_filename:
            instruction = "Given a prompt and two completions from a language model, select the completion that is the least toxic with respect to continuing the prompt. Toxicity is defined as a rude, disrespectful, or unreasonable statement that is likely to make you leave a discussion. Generate either Completion {enum1_1} or Completion {enum1_2}."
        elif 'task138_' in args.task_filename:
            instruction = "Given a prompt and two completions from a language model, select the completion that is the most fluent with respect to continuing the prompt. Fluency is defined by coherency and ease of understanding, not necessarily grammatical correctness. Generate either Completion {enum1_1} or Completion {enum1_2}."
        elif 'task139_' in args.task_filename:
            instruction = "Given a prompt and two completions from a language model, select the completion that is more topical with respect to continuing the prompt. A prompt-completion pair is defined to be topical if the completion maintains relevance and logical succession (i.e. stays on topic) with the prompt. The flow from the prompt to the completion should be as reasonable as possible. Generate either Completion {enum1_1} or Completion {enum1_2}."
        elif 'task140_' in args.task_filename:
            instruction = "Given a prompt and two completions from a language model, select the completion that has the most similar style to the prompt. Style is defined as the tone, word choice, grammar, and sentence structure throughout the prompt-completion pair. If a prompt is colloquial, then the completion should also be colloquial, as opposed to a completion that is encyclopedic or overly formal. Generate either Completion {enum1_1} or Completion {enum1_2}."
        original_multiple_choice_output_format = "Completion {enum1}"

        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                # [0, 2, 3, 4] -> [1, 3, 4, 5] because of indexing
                SimplePromptFormat('Prompt', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                NewEnumerationPromptFormat('Completion', 2, chosen_separator=': ', chosen_space=' \n',
                                           chosen_number_format=lambda x: chr(ord('A') + x),
                                           chosen_item_wrapper=lambda x: x, text_descriptor_fn_owner=text_descriptor_fn,
                                           object_name='enum1'),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n'
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task638_' in args.task_filename:
        0 / 0
        instruction = 'You are shown a conversation between a user and system. Identify who has spoken the indicated sentence based on the conversation.'
        # original_multiple_choice_output_format is complex here, but the task has been discarded anyways because of low perf

        # Sentence1:<text> Sentence2: <text> Sentence3: <text> Question: <text> (A) <text> (B) <text>
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)

        min_elements = 1
        max_elements = 45
        extra_params_structured_prompt_format = {'enumeration_length_range': (min_elements, max_elements + 1)}

        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                NewEnumerationPromptFormat('Sentence', max_elements, chosen_separator=': ', chosen_space=', ',
                                           chosen_separator_text_and_option='',
                                           chosen_item_wrapper=lambda x: f"{x}",
                                           text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn),
                        NewEnumerationPromptFormat('', 2, chosen_separator=' ', chosen_space=' ',
                                                   chosen_item_wrapper=lambda x: f"({x})",
                                                   chosen_number_format=lambda x: chr(ord('A') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space=' '
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' '
        )

        global_constraints = [text_descriptor_fn, chosen_separator]

    elif any(t in args.task_filename for t in ['task296_', 'task297_']):
        instruction = "In this task, you're given four sentences of a story written in natural language. The given story is not complete and your job is to complete the story by selecting one of the sentence choices from ({enum1_1}) and ({enum1_2}), such that the story sounds fully coherent."  # FIXME also include formatting options in enum1
        original_multiple_choice_output_format = "{enum1}"

        # Sentence1: <text> Sentence2: <text> Sentence3: <text> Sentence4: <text> \n (A) <text> (B) <text>
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)

        min_elements = 1
        max_elements = 10
        extra_params_structured_prompt_format = {'enumeration_length_range': (min_elements, max_elements + 1)}

        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                NewEnumerationPromptFormat('Sentence', max_elements, chosen_separator=': ', chosen_space=' ',
                                           chosen_separator_text_and_option='',
                                           chosen_item_wrapper=lambda x: f"{x}",
                                           text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        NewEnumerationPromptFormat('', 2, chosen_separator=' ', chosen_space=' ',
                                                   chosen_item_wrapper=lambda x: f"({x})",
                                                   chosen_number_format=lambda x: chr(ord('A') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space=' '
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n '
        )

        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task1565_' in args.task_filename:
        # Question:<text> ,  Options: [A.jack miller B.bobby brown]
        # FIXME: we'd need to implement the wrapping with [...]
        0 / 0
        original_multiple_choice_output_format = "{enum1}"

        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Options', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 2, chosen_separator='', chosen_space=' ',
                                                   chosen_item_wrapper=lambda x: f'{x}.',
                                                   chosen_number_format=lambda x: chr(ord('A') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' ,  '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task118_' in args.task_filename:
        # """<text>\n(A)68 (B)64 (C)60 (D)16 (E)15"""

        instruction = "You are given a mathematical question described with an open-ended vocabulary. Questions in this task involve real-world situations, describing a mathematical problem. You are also given 4 or 5 answer options (associated with \"{enum1_1}\", \"{enum1_2}\", \"{enum1_3}\", \"{enum1_4}\", \"{enum1_5}\"). Do not generate anything else apart from one of the following characters: 'A', 'B, 'C', 'D', 'E'. LaTeX mathematical format (the standard way to express mathematical expressions in the typesetting software known as LaTeX) is used to express equations. Each question is solvable with high school math knowledge. Give only one answer for each question."
        original_multiple_choice_output_format = '{enum1}'

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                NoTextPromptFormat(),
                NewEnumerationPromptFormat('', 5, chosen_separator='', chosen_separator_text_and_option='',
                                           chosen_space=' ', chosen_item_wrapper=lambda x: f"({x})",
                                           chosen_number_format=lambda x: chr(ord('A') + x), object_name='enum1'),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space='\n'
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task1135_' in args.task_filename:
        instruction = "In this task, you will be presented with a question that has multiple possible answers. You should choose the most suitable option out of \"{enum1_1}\", \"{enum1_2}\", \"{enum1_3}\", \"{enum1_4}\", and \"{enum1_5}\", based on your commonsense knowledge."
        original_multiple_choice_output_format = '{enum1}'

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Options', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 5, chosen_separator=' ', chosen_separator_text_and_option='',
                                                   chosen_space=' ', chosen_item_wrapper=lambda x: x,
                                                   chosen_number_format=lambda x: chr(ord('A') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif any(t in args.task_filename for t in
             ['task1424_', 'task1423_', 'task1422_', 'task1421_', 'task1420_', 'task1419_']):
        # Problem: <text> \nOptions: a ) <text> , b ) <text> , c ) <text> , d ) <text> , e ) <text>
        if 'task1419_' in args.task_filename:
            instruction = "In this task, you need to answer the given multiple-choice question on the gain. Gain is the value by which to multiply the input. Classify your answers into '{enum1_1}', '{enum1_2}', '{enum1_3}', '{enum1_4}', and '{enum1_5}'."
        elif 'task1420_' in args.task_filename:
            instruction = "In this task, you need to answer the given multiple-choice question on the general math. Classify your answers into '{enum1_1}', '{enum1_2}', '{enum1_3}', '{enum1_4}', and '{enum1_5}'."
        elif 'task1421_' in args.task_filename:
            instruction = "In this task, you need to provide the correct option for a given problem from the provided options."
        elif 'task1422_' in args.task_filename:
            instruction = "In this task, you need to answer the given multiple-choice question on the physics. Classify your answers into '{enum1_1}', '{enum1_2}', '{enum1_3}', '{enum1_4}', and '{enum1_5}'."
        elif 'task1423_' in args.task_filename:
            instruction = "In this task, you need to answer the given multiple-choice question on geometry. Classify your answers into '{enum1_1}', '{enum1_2}', '{enum1_3}', '{enum1_4}', and '{enum1_5}'."
        elif 'task1424_' in args.task_filename:
            instruction = "In this task, you need to provide the correct option for a given problem on probability from the provided options."
        original_multiple_choice_output_format = "{enum1}"

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Problem', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Options', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 5, chosen_separator=' ', chosen_separator_text_and_option='',
                                                   chosen_space=' , ', chosen_item_wrapper=lambda x: f'{x} )',
                                                   chosen_number_format=lambda x: chr(ord('a') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n'
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task1678_' in args.task_filename:
        # Problem: <|text|>\nOptions: a. <|text|>, b. <|text|>, c. <|text|>, d. <|text|>, e. <|text|>
        instruction = "Given a math problem with context and a question and 5 answer choices, the task is to provide the correct answer choice based on the problem. You must choose one of the given answer choices by letter: {enum1_1}, {enum1_2}, {enum1_3}, {enum1_4}, and {enum1_5}; anything else is invalid."
        original_multiple_choice_output_format = "{enum1}"

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Problem', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Options', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 5, chosen_separator=' ', chosen_separator_text_and_option='',
                                                   chosen_space=', ', chosen_item_wrapper=lambda x: f'{x}.',
                                                   chosen_number_format=lambda x: chr(ord('a') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space='\n'
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task385_' in args.task_filename or 'task580_' in args.task_filename:
        # Context: Even though she had homework to do that night, Jesse helped Skylar study.
        #  Question: What will Jesse want to do next?
        #  Options: (A) read homework to Skylar (B) help Skylar finish (C) skip her studying

        if 'task385_' in args.task_filename:
            instruction = "In this task, you're given a context passage, a question, and three answer options. Your task is to return an incorrect answer option to the question from the choices given. For all questions, only one of the three answer options is correct. Pick one of the two incorrect answer options as the output."
        elif 'task580_' in args.task_filename:
            instruction = "In this task, you're given a context, a question, and three options. Your task is to find the correct answer to the question using the given context and options. Also, you may need to use commonsense reasoning about social situations to answer the questions. Classify your answers into '{enum1_1}', '{enum1_2}', and '{enum1_3}'."
        else:
            assert False
        original_multiple_choice_output_format = '{enum1}'

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Context', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SimplePromptFormat('Question', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Options', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 3, chosen_separator=' ', chosen_separator_text_and_option='',
                                                   chosen_space=' ', chosen_item_wrapper=lambda x: f"({x})",
                                                   chosen_number_format=lambda x: chr(ord('A') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' \n '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]

    elif 'task214_' in args.task_filename or 'task213_' in args.task_filename:
        # Title: The Lawsuit. Sentence 1: Denise got hit by a car. Sentence 2: She sued the driver. Sentence 3: She got a huge settlement. Sentence 4: Denise retired and moved to the beach. Choices: a. He signed up for another class to learn more. b. Her fortune was worth the pain!

        if 'task213_' in args.task_filename:
            instruction = "In this task, you're given the title of a five-sentence story, the first four sentences, and two options for the fifth sentence as {enum1_1} and {enum1_2}. Your job is to pick the sentence option that seamlessly connects with the rest of the story, indicating your choice as '{enum1_1}' or '{enum1_2}'. If both sentences are plausible, pick the one that makes more sense."
        elif 'task214_' in args.task_filename:
            instruction = "In this task, you're given the title of a five-sentence story, the first four sentences, and two options for the fifth sentence as {enum1_1} and {enum1_2}. Your job is to pick the sentence option that does not connect with the rest of the story, indicating your choice as '{enum1_1}' or '{enum1_2}'. If both sentences are plausible, pick the one that makes less sense."
        else:
            assert False
        original_multiple_choice_output_format = '{enum1}'

        chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
        text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
        structured_prompt_format = SpacingBetweenPromptComponents(
            [
                SimplePromptFormat('Title', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn),
                NewEnumerationPromptFormat('Sentence', 4, chosen_separator_owner=chosen_separator,
                                           chosen_separator_text_and_option=' ',
                                           chosen_space=' ', chosen_item_wrapper=lambda x: f"{x}",
                                           chosen_number_format=lambda x: x + 1, object_name='enum0'),
                SpacingBetweenPromptComponents(
                    [
                        SimplePromptFormat('Choices', None, chosen_separator_owner=chosen_separator,
                                           text_descriptor_fn_owner=text_descriptor_fn, prompt_without_text=True),
                        NewEnumerationPromptFormat('', 2, chosen_separator='. ', chosen_separator_text_and_option='',
                                                   chosen_space=' ', chosen_item_wrapper=lambda x: x,
                                                   chosen_number_format=lambda x: chr(ord('a') + x),
                                                   object_name='enum1'),
                    ],
                    chosen_space='',
                    allow_only_non_char_spaces=True
                ),
                SimplePromptFormat('Answer', None, chosen_separator_owner=chosen_separator,
                                   text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
            ],
            chosen_space=' '
        )
        global_constraints = [text_descriptor_fn, chosen_separator]
    else:
        # task058 = cannot be done because it has two moving length variables
        print("Unrecognized task", args.task_filename)
        return None, None, None, None, None

    return structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
           instruction, original_multiple_choice_output_format
