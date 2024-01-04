import random
import inspect

random.seed(42)


# removed '\n\n' to make sure this is only used between entries
CHOSEN_SEPARATOR_LIST = ['', '::: ', ':: ', ': ', ' \n\t', '\n    ', ' : ', ' - ', ' ', '\n ', '\n\t', ':', '::', '- ', '\t']  # sep='' is used rarely, only for enumerations because there is already formatting there
CHOSEN_SPACE_LIST = ['', ' ', '\n', ' \n', ' -- ',  '  ', '; \n', ' || ', ' <sep> ', ' -- ', ', ', ' \n ', ' , ', '\n ', '. ', ' ,  ']  # space='' is used a lot
CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST = ['', ' ', '  ', '\t']

CHOSEN_SEPARATOR_LIST = [(e, e) for e in CHOSEN_SEPARATOR_LIST]
CHOSEN_SPACE_LIST = [(e, e) for e in CHOSEN_SPACE_LIST]
CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST = [(e, e) for e in CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST]


TEXT_DESCRIPTOR_FN_LIST = [
    (lambda x: x, "lambda x: x"),
    (lambda x: x.title(), "lambda x: x.title()"),
    (lambda x: x.upper(), "lambda x: x.upper()"),
    (lambda x: x.lower(), "lambda x: x.lower()")
]
ITEM_WRAPPER_LIST = [
    (lambda x: f'({x})', "lambda x: f'({x})'"),
    (lambda x: f'{x}.', "lambda x: f'{x}.'"),
    (lambda x: f'{x})', "lambda x: f'{x})'"),
    (lambda x: f'{x} )', "lambda x: f'{x} )'"),
    (lambda x: f'[{x}]', "lambda x: f'[{x}]'"),
    (lambda x: f'<{x}>', "lambda x: f'<{x}>'"),
]
NUMBER_FORMAT_LIST = [
    (lambda x: x + 1, "lambda x: x + 1"),
    (lambda x: chr(ord('A') + x), "lambda x: chr(ord('A') + x)"),
    (lambda x: chr(ord('a') + x), "lambda x: chr(ord('a') + x)"),
    (lambda x: chr(0x215F + x + 1) + ('' if x < 12 else 0 / 0), "lambda x: chr(0x215F + x + 1)"),
    (lambda x: NewEnumerationPromptFormat.ROMAN_NUMERALS[x], "lambda x: EnumerationPromptFormat.ROMAN_NUMERALS[x]"),
    (lambda x: NewEnumerationPromptFormat.ROMAN_NUMERALS[x].upper(), "lambda x: EnumerationPromptFormat.ROMAN_NUMERALS[x].upper()")
]

MAPPING_ALL_CATEGORIES = {
    'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST,
    'chosen_item_wrapper': ITEM_WRAPPER_LIST,
    'chosen_number_format': NUMBER_FORMAT_LIST,
    'chosen_space': CHOSEN_SPACE_LIST,
    'chosen_separator': CHOSEN_SEPARATOR_LIST,  # in OPTION_1:^TEXT, this is ^
    'chosen_separator_text_and_option': CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST  # in OPTION_1:^TEXT, this is _
}


def lambda_to_string(lambda_fn):
    funcString = str(inspect.getsourcelines(lambda_fn)[0])
    funcString = funcString.strip("['\\n']").strip('\\n"').split("=")[1].strip().strip(',').strip('\n')
    return funcString

class SpacingBetweenPromptComponents:
    SEARCH_SPACE_VALID_OPTIONS = {
        'chosen_space': CHOSEN_SPACE_LIST
    }
    SYNONYM_SETS = []

    def __init__(self, prompt_format_list, chosen_space, allow_only_non_char_spaces=False):
        self.chosen_space = chosen_space
        self.prompt_format = prompt_format_list  # or SharedPropertyAmongPrompts

        self.is_output_field = False

        # in some cases we want to avoid having a comma like a space (only used right now for original chosen_space='')
        self.allow_only_non_char_spaces = allow_only_non_char_spaces

    def solve(self, extra_params=None):
        prompt_format_with_resolved_shared_property = self.prompt_format
        if isinstance(self.prompt_format, SharedPropertyAmongPrompts):
            prompt_format_with_resolved_shared_property = self.prompt_format.solve(extra_params)

        result = []
        for i, e in enumerate(prompt_format_with_resolved_shared_property):
            # ignore an output field if that was the request
            if not isinstance(e, str) and e.is_output_field:
                if extra_params and extra_params.get('print_output_fields', False):
                    if i > 0:
                        result.append(self.chosen_space)
                    result.append(e.solve(extra_params))
            else:
                if i > 0:
                    result.append(self.chosen_space)
                result.append(e.solve(extra_params) if not isinstance(e, str) else e)

        return result

    def find_all_formatted_field_values(self):
        if isinstance(self.prompt_format, SharedPropertyAmongPrompts):
            return self.prompt_format.find_all_formatted_field_values()
        else:
            result = {}
            for e in self.prompt_format:
                assert len(set(result.keys()) & set(e.find_all_formatted_field_values().keys())) == 0
                result.update(e.find_all_formatted_field_values())
            return result

    def update_field(self, field_name, new_field_value):
        if field_name not in self.__dict__:
            return False

        if self.allow_only_non_char_spaces and not new_field_value.isspace():
            return False

        setattr(self, field_name, new_field_value)
        return True

    def has_attribute(self, field_name):
        return field_name in self.__dict__

    def attributes_under_control(self):
        return list(self.SEARCH_SPACE_VALID_OPTIONS.keys())


class NewEnumerationPromptFormat:
    """
    Variable-length enumeration. E.g. listing facts, listing options.

    This new version is less recursive.

    Option 1 : text <sep> Option 2 : text
    """

    ROMAN_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv']
    SEARCH_SPACE_VALID_OPTIONS = {
        'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST,
        'chosen_item_wrapper': ITEM_WRAPPER_LIST,
        'chosen_number_format': NUMBER_FORMAT_LIST,
        'chosen_space': CHOSEN_SPACE_LIST,
        'chosen_separator': CHOSEN_SEPARATOR_LIST,  # in OPTION_1:^TEXT, this is ^
        'chosen_separator_text_and_option': CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST  # in OPTION_1:^TEXT, this is _
    }

    SYNONYM_SETS = []
    def __init__(self,
                 text_descriptor_format,
                 length,
                 chosen_space,
                 chosen_separator=': ',
                 chosen_separator_owner=None,
                 chosen_separator_text_and_option=None,
                 chosen_item_wrapper=None,
                 chosen_number_format=None,
                 text_descriptor_fn=lambda x: x,
                 text_descriptor_fn_owner=None,
                 object_name=None):
        self.chosen_item_wrapper = \
            chosen_item_wrapper if chosen_item_wrapper else self.SEARCH_SPACE_VALID_OPTIONS['chosen_item_wrapper'][0][0]
        self.chosen_number_format = \
            chosen_number_format if chosen_number_format else self.SEARCH_SPACE_VALID_OPTIONS['chosen_number_format'][0][0]
        self.chosen_space = chosen_space
        self.chosen_separator = chosen_separator
        self.chosen_separator_owner = chosen_separator_owner

        if chosen_separator_text_and_option is None:
            chosen_separator_text_and_option = '' if not text_descriptor_format else ' '
        self.chosen_separator_text_and_option = chosen_separator_text_and_option

        self.chosen_space_between_text_and_item = None

        self.text_descriptor_format = text_descriptor_format
        self.text_descriptor_fn_owner = text_descriptor_fn_owner
        self.text_descriptor_fn = text_descriptor_fn

        assert isinstance(length, int) or isinstance(length, list)
        length_range = range(length) if isinstance(length, int) else length

        self.enumeration_item_id_list = length_range
        self.prompt_format = text_descriptor_format  # FIXME? this is just so that it's a str for when calling pointers_to_all_objects()

        self.is_output_field = False
        self.object_name = object_name  # used to reference this object when filling

    def format_text_descriptor_field(self, index):
        text = '<|text|>'

        if self.text_descriptor_fn_owner is None:
            prompt = self.text_descriptor_fn(self.text_descriptor_format)
        else:
            prompt = self.text_descriptor_fn_owner.apply_field_fn('text_descriptor_fn', self.text_descriptor_format)

        chosen_separator = self.chosen_separator
        if self.chosen_separator_owner is not None:
            assert 'chosen_separator' in self.chosen_separator_owner.fields
            chosen_separator = self.chosen_separator_owner.fields['chosen_separator']

        # return prompt.format(self.chosen_item_wrapper(self.chosen_number_format(index)))
        return f"{prompt}{self.chosen_separator_text_and_option}{self.chosen_item_wrapper(self.chosen_number_format(index))}{chosen_separator}{text}"

    def solve(self, extra_params=None):
        """
        extra_params: Dictates whether to modify the self.prompt_format.
        Currently used only to print fewer options in the enumeration than the maximum allowed.
        """
        enumeration_length = extra_params.get('enumeration_length', None) if extra_params else None

        # First, solve each enumeration item
        solved_elements = []
        for index in self.enumeration_item_id_list[:enumeration_length]:
            solved_elements.append(self.format_text_descriptor_field(index))

        result = []
        for i, e in enumerate(solved_elements):
            if i > 0:
                result.append(self.chosen_space)
            result.append(e.solve(extra_params) if not isinstance(e, str) else e)

        return result

    def find_all_formatted_field_values(self):
        """
        Obtain a dictionary with all the (field_name, field_value) to be used
        in updating the instruction formatted field values.
        """

        if self.object_name:
            field_names_to_values = {
                f'{self.object_name}_{i + 1}': self.chosen_number_format(index)
                for i, index in enumerate(self.enumeration_item_id_list)
            }
            return field_names_to_values
        return {}

    def update_field(self, field_name, new_field_value):
        if field_name not in self.__dict__:
            return False

        """
        Check for consistency between components, to avoid weird looking enumerations like the following:

        Options:
        1.
        {} 2.
        {} 3.
        {} 4.
        {}
        
        Rule to enforce is: '\n' in chosen_separator (e.g. "::" in "1::") => '\n' in chosen_space
        """
        spacing_values = {
            'chosen_separator': self.chosen_separator,
            'chosen_separator_text_and_option': self.chosen_separator_text_and_option,
            'chosen_space': self.chosen_space
        }
        spacing_values[field_name] = new_field_value

        if self.chosen_separator_owner is not None:
            assert 'chosen_separator' in self.chosen_separator_owner.fields
            spacing_values['chosen_separator'] = self.chosen_separator_owner.fields['chosen_separator']

        if ('\n' in spacing_values['chosen_separator'] or
            '\n' in spacing_values['chosen_separator_text_and_option']) and \
                '\n' not in spacing_values['chosen_space']:
            return False

        setattr(self, field_name, new_field_value)
        return True

    def has_attribute(self, field_name):
        return field_name in self.__dict__

    def attributes_under_control(self):
        attrs = list(self.SEARCH_SPACE_VALID_OPTIONS.keys())
        if not self.text_descriptor_format:
            attrs.remove('text_descriptor_fn')  # changing casing and space from an empty string doesn't make sense
            attrs.remove('chosen_separator_text_and_option')
        if self.text_descriptor_fn_owner is not None:
            attrs.remove('text_descriptor_fn')  # this attribute is controlled by some other entity
        if self.chosen_separator_owner is not None:
            attrs.remove('chosen_separator')
        return attrs


class SimplePromptFormat:
    """
    Simplest formatting. For example,

    Sentence: <|text|>
    Question: <|text|>
    Answer: <|text|>
    """

    SEARCH_SPACE_VALID_OPTIONS = {
        'chosen_separator': CHOSEN_SEPARATOR_LIST,
        'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST
    }
    SYNONYM_SETS = []

    def __init__(self,
                 text_descriptor,
                 separator,
                 text_descriptor_fn=lambda x: x,
                 prompt_without_text=False,
                 chosen_separator_owner=None,
                 text_descriptor_fn_owner=None,
                 is_output_field=False):
        self.text_descriptor = text_descriptor  # keep as is
        self.chosen_separator = separator
        self.prompt_format = self.text_descriptor

        self.prompt_without_text = prompt_without_text  # used for text only prompts (without variable text)
        self.text_descriptor_fn = text_descriptor_fn
        # self.index_item = -1  # only used for enumerations

        self.text_descriptor_owner = None
        self.chosen_separator_owner = chosen_separator_owner
        self.text_descriptor_fn_owner = text_descriptor_fn_owner

        self.is_output_field = is_output_field

    def assign_field_owner(self, field_name, owner):
        assert field_name in self.__dict__
        setattr(self, field_name + '_owner', owner)

    def solve(self, extra_params=None):

        resolved_prompt_format = self.prompt_format
        if self.text_descriptor_owner:  # only used for enumeration
            assert self.index_item is not None
            resolved_prompt_format = self.text_descriptor_owner.format_text_descriptor_field(self.index_item)
        elif self.text_descriptor_fn_owner:
            resolved_prompt_format = self.text_descriptor_fn_owner.apply_field_fn('text_descriptor_fn', resolved_prompt_format)
        else:
            resolved_prompt_format = self.text_descriptor_fn(resolved_prompt_format)

        true_separator = self.chosen_separator
        if self.chosen_separator_owner:
            assert 'chosen_separator' in self.chosen_separator_owner.fields
            true_separator = self.chosen_separator_owner.fields['chosen_separator']

        exclude_text_field_for_output_fields = self.is_output_field and extra_params and extra_params.get('exclude_text_field_for_output_fields', False)

        text = '' if self.prompt_without_text or exclude_text_field_for_output_fields else '<|text|>'
        return f"{resolved_prompt_format}{true_separator}{text}"

    def find_all_formatted_field_values(self):
        return {}

    def update_field(self, field_name, new_field_value):
        if field_name not in self.__dict__:
            return False

        if self.chosen_separator_owner and field_name in self.chosen_separator_owner.fields:
            return False

        # we need a separator on simple prompt format, otherwise it'd be "INPUT<text>" which is illegible
        if field_name == 'chosen_separator' and new_field_value == '':
            return False

        setattr(self, field_name, new_field_value)
        return True

    def has_attribute(self, field_name):
        return field_name in self.__dict__

    def attributes_under_control(self):
        result = []
        if self.chosen_separator_owner is None:
            result.append('chosen_separator')
        if self.text_descriptor_fn_owner is None and self.text_descriptor:
            result.append('text_descriptor_fn')
        return result


class NoTextPromptFormat:
    SEARCH_SPACE_VALID_OPTIONS = {}

    def __init__(self):
        self.is_output_field = False
        self.prompt_format = ''

    def solve(self, extra_params=None):
        exclude_text_field_for_output_fields = self.is_output_field and extra_params and extra_params.get('exclude_text_field_for_output_fields', False)

        text = '' if exclude_text_field_for_output_fields else '<|text|>'
        return text

    def attributes_under_control(self):
        return []

    def find_all_formatted_field_values(self):
        return {}


class SharedPropertyAmongPrompts:
    SEARCH_SPACE_VALID_OPTIONS = {
        'chosen_separator': CHOSEN_SEPARATOR_LIST,
        'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST
    }
    SYNONYM_SETS = []

    def __init__(self, fields_dict, prompt_list_to_apply):
        self.fields = fields_dict  # = {'chosen_separator': ':: '}
        self.prompt_format = prompt_list_to_apply

        if prompt_list_to_apply is not None:
            for field_name, field_value in self.fields.items():
                for e in self.prompt_format:
                    e.assign_field_owner(field_name, self)
                    assert field_name in e.__dict__
                    setattr(e, field_name, field_value)

        self.is_output_field = False

    def solve(self, extra_params=None):
        if self.prompt_format is None:
            return None

        enumeration_length = extra_params.get('enumeration_length') if extra_params else None  # a[:None] returns full list

        result = []
        for e in self.prompt_format[:enumeration_length]:
            if not isinstance(e, str) and e.is_output_field:
                if extra_params and extra_params.get('print_output_fields', False):
                    result.append(e.solve(extra_params))
            else:
                result.append(e.solve(extra_params))

    def find_all_formatted_field_values(self):
        if self.prompt_format is None:
            return {}

        result = {}
        for e in self.prompt_format:
            assert len(set(result.keys()) & set(e.find_all_formatted_field_values().keys())) == 0
            result.update(e.find_all_formatted_field_values())
        return result

    def update_field(self, field_name, new_field_value):
        if field_name not in self.fields:
            return False

        self.fields[field_name] = new_field_value
        return True

    def has_attribute(self, field_name):
        return field_name in self.fields

    def apply_field_fn(self, field_name, string):
        assert field_name in self.fields
        return self.fields[field_name](string)

    def attributes_under_control(self):
        return list(self.fields.keys())


def flatten(nested_string_list):
    return "".join([flatten(e) if isinstance(e, list) else e for e in nested_string_list])


def pointers_to_all_objects(root_element):
    result = [root_element]
    if not isinstance(root_element.prompt_format, list):
        return result + pointers_to_all_objects(root_element.prompt_format)

    for elem in root_element.prompt_format:
        result.append(elem)
        if not isinstance(elem.prompt_format, str):
            result.extend(pointers_to_all_objects(elem))

    return result


def get_possible_actions(e, allow_text_action_type=True):
    possible_keys = [k for k in e.SEARCH_SPACE_VALID_OPTIONS if e.has_attribute(k)]  # is punctuation replacement an option?
    assert all([k in possible_keys for k in e.attributes_under_control()]), f'{e.attributes_under_control()} not subset of {possible_keys} for node {e.solve()}'

    possible_keys = e.attributes_under_control()  # this should avoid self loops in graph search

    if allow_text_action_type and any(v in e.prompt_format for v_list in e.SYNONYM_SETS for v in v_list):  # is text replacement an option?
        possible_keys += ['text']

    return possible_keys


def create_pointer_action_type_pairs(
        all_pointers_enumerated, forced_action_type=None, allow_text_action_type=True
):
    """
    Simultaneously choose which element we'll perform the action over, and the action itself.
    """

    pointer_action_pairs = []
    for e, index in all_pointers_enumerated:
        possible_keys = get_possible_actions(e, allow_text_action_type)
        if forced_action_type:
            possible_keys = [forced_action_type] if forced_action_type in possible_keys else []
        for action_type in possible_keys:
            pointer_action_pairs.append((e, index, action_type))

    return pointer_action_pairs


def holistic_node_format_sanity_checks(root_element, prohibit_newlines=False):
    """
    Checks that the prompt format's value assignments are reasonable, and consistent across fields.

    For example, this functions checks that if a space between component does not have \n, then the separator between
    fields should also not have that.

    E.g. input\n{}output\n{} returns False.
    E.g. input\n{}\noutput\n{} returns True.
    E.g. input {}\noutput {} returns True.
    E.g. this should return True (because Options is prompt_without_text=True):
        Question
        <|text|>
        Options
        [1] <|text|> [2] <|text|> [3] <|text|> [4] <|text|> [5] <|text|>
        Answer
        <|text|>

    Also checks the update_field() rule of spacing in NewEnumerationPromptFormat.

    """
    if isinstance(root_element, str):
        return True

    # local constraint from NewEnumeration, added here because update_field() won't be called from genetic/global_random
    if isinstance(root_element, NewEnumerationPromptFormat):
        spacing_values = {
            'chosen_separator': root_element.chosen_separator,
            'chosen_separator_text_and_option': root_element.chosen_separator_text_and_option,
            'chosen_space': root_element.chosen_space
        }
        if root_element.chosen_separator_owner is not None:
            assert 'chosen_separator' in root_element.chosen_separator_owner.fields
            spacing_values['chosen_separator'] = root_element.chosen_separator_owner.fields['chosen_separator']

        if ('\n' in spacing_values['chosen_separator'] or
            '\n' in spacing_values['chosen_separator_text_and_option']) and \
                '\n' not in spacing_values['chosen_space']:
            return False

    # local constraint from simple prompt format: we need an actual separator in simple formats, '' is invalid
    if isinstance(root_element, SimplePromptFormat):
        true_separator = root_element.chosen_separator
        if root_element.chosen_separator_owner:
            assert 'chosen_separator' in root_element.chosen_separator_owner.fields
            true_separator = root_element.chosen_separator_owner.fields['chosen_separator']

        if true_separator == '':
            return False

    # local constraint from SpacingBetweenPromptComponents
    if isinstance(root_element, SpacingBetweenPromptComponents) and \
            root_element.allow_only_non_char_spaces and not root_element.chosen_space.isspace():
        return False

    # global constraint: avoid using chosen_space='' unless it is separating between a prompt without text and a text.
    # E.g. INPUT - <|text|>OUTPUT - <|text|>  should not be allowed but
    # OPTIONS: A. text B. text should be accepted
    if isinstance(root_element, SpacingBetweenPromptComponents) and root_element.chosen_space == '' and \
            isinstance(root_element.prompt_format, list):

        all_prompt_without_texts_except_maybe_last_elem = all(
            hasattr(elem, 'prompt_without_text') and elem.prompt_without_text
            for elem in root_element.prompt_format[:-1])
        if not all_prompt_without_texts_except_maybe_last_elem:
            return False

    # global constraint with newlines as explained in the function's documentation
    if isinstance(root_element, SpacingBetweenPromptComponents) and '\n' not in root_element.chosen_space:
        if isinstance(root_element.prompt_format, list):
            return all(holistic_node_format_sanity_checks(e, prohibit_newlines=True) for e in root_element.prompt_format)
        else:
            return holistic_node_format_sanity_checks(root_element.prompt_format, prohibit_newlines=True)

    # FIXME add the exception of an empty text field
    if prohibit_newlines and hasattr(root_element, 'chosen_separator'):
        # if the prompt does not have text then it is ok to put a new line, since it's not awkwardly separating
        # the descriptor from the text, which is our goal here
        if hasattr(root_element, 'prompt_without_text') and root_element.prompt_without_text:
            pass
        else:
            chosen_separator = root_element.chosen_separator
            if root_element.chosen_separator_owner is not None:
                assert 'chosen_separator' in root_element.chosen_separator_owner.fields
                chosen_separator = root_element.chosen_separator_owner.fields['chosen_separator']
            if '\n' in chosen_separator:
                return False

    if not isinstance(root_element.prompt_format, list):
        return holistic_node_format_sanity_checks(root_element.prompt_format, prohibit_newlines=prohibit_newlines)

    result = [holistic_node_format_sanity_checks(elem, prohibit_newlines=prohibit_newlines)
              for elem in root_element.prompt_format]
    return all(result)


def apply_prompt_format(prompt, input_fields):
    # Possible FIX for variable-length prompt formats. Choose output based on the number of fields.
    tmp = prompt.format(*input_fields)
    if prompt.count('{}') != len(input_fields):
        print('WARNING, wrong number of fields!', prompt, input_fields)
    return tmp


def _one_text_field(text1, answer_field_text='Answer', chosen_space='\n'):
    # Input: <text>\nOutput: <text>
    chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
    text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
    structured_prompt_format = SpacingBetweenPromptComponents(
        [
            SimplePromptFormat(text1, None, chosen_separator_owner=chosen_separator,
                               text_descriptor_fn_owner=text_descriptor_fn),
            SimplePromptFormat(answer_field_text, None, chosen_separator_owner=chosen_separator,
                               text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
        ],
        chosen_space=chosen_space
    )
    global_constraints = [text_descriptor_fn, chosen_separator]

    return structured_prompt_format, global_constraints


def _two_text_fields(text1, text2, answer_field_text='Answer', chosen_space='\n'):
    # Passage: <text>\nQuestion: <text>
    chosen_separator = SharedPropertyAmongPrompts({'chosen_separator': ': '}, None)
    text_descriptor_fn = SharedPropertyAmongPrompts({'text_descriptor_fn': lambda x: x}, None)
    structured_prompt_format = SpacingBetweenPromptComponents(
        [
            SimplePromptFormat(text1, None, chosen_separator_owner=chosen_separator,
                               text_descriptor_fn_owner=text_descriptor_fn),
            SimplePromptFormat(text2, None, chosen_separator_owner=chosen_separator,
                               text_descriptor_fn_owner=text_descriptor_fn),
            SimplePromptFormat(answer_field_text, None, chosen_separator_owner=chosen_separator,
                               text_descriptor_fn_owner=text_descriptor_fn, is_output_field=True)
        ],
        chosen_space=chosen_space
    )
    global_constraints = [text_descriptor_fn, chosen_separator]

    return structured_prompt_format, global_constraints

