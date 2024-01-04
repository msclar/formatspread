import copy
import random
from typing import List

import numpy as np

from grammar_definition import pointers_to_all_objects, create_pointer_action_type_pairs, \
    flatten, MAPPING_ALL_CATEGORIES, holistic_node_format_sanity_checks
from utils import evaluate_prompt_format

random.seed(0)


def value_assignment_str_to_indices(value_assignments, pointer_action_pairs):
    value_assignments_ids = []
    for assignment in value_assignments:
        assert len(pointer_action_pairs) == len(assignment), f"{len(pointer_action_pairs)} != {len(assignment)}"
        assignment_ids = []
        for (_, _, action_type), assignment_value in zip(pointer_action_pairs, assignment):
            idx = [i for i, (_, v) in enumerate(MAPPING_ALL_CATEGORIES[action_type]) if v == assignment_value][0]
            assignment_ids.append(idx)
        value_assignments_ids.append(assignment_ids)
    return value_assignments_ids


class GeneticAlgorithmAmongPrompts:

    def __init__(self,
                 structured_prompt_format,
                 global_constraints,
                 extra_params_structured_prompt_format,
                 args_compute_node_score,
                 objective,
                 allow_text_action_type=True,
                 original_multiple_choice_output_format=None):
        self.args_compute_node_score = args_compute_node_score
        self.metadata = {}
        self.all_structured_prompt_formats_last_id_evaluated = {}
        self.all_structured_prompt_formats_accuracies = {}  # actually has the accuracies computed
        self.objective = objective
        self.extra_params_structured_prompt_format = extra_params_structured_prompt_format
        self.original_multiple_choice_output_format = original_multiple_choice_output_format

        # nodes (prompt formats) are represented by their solved_format
        solved_format = self._get_node_from_format(structured_prompt_format)
        self.all_structured_prompt_formats = {
            solved_format: [structured_prompt_format, global_constraints]  # nodes
        }

        # all multiple choice classes in the original format, important to know how to update them when format changes
        original_multiple_choice_classes = self.find_all_multiple_choice_output_classes(
            solved_format, original_multiple_choice_output_format)
        self.original_multiple_choice_classes = original_multiple_choice_classes

        self.generation_order = {solved_format: 0}
        self.edges = []
        self.allow_text_action_type = allow_text_action_type

        self.metadata = {}
        self.metadata['extra_params'] = {'allow_text_action_type': self.allow_text_action_type}
        self.metadata['nodes'] = {}  # used in some extensions of this class
        self.metadata['bit_representations'] = {}  # used in some extensions of this class

        self.all_structured_prompt_formats_accuracies = {
            solved_format: self._compute_node_score(structured_prompt_format, num_samples_to_test=-1)
        }
        self.metadata['bit_representations'][solved_format] = [None]  # None = no actions have been done yet
        self.metadata['extra_params']['objective'] = self.objective

        all_pointers = pointers_to_all_objects(structured_prompt_format) + global_constraints
        all_pointers_enumerated = [(e, i) for i, e in enumerate(all_pointers)]
        pointer_action_pairs = create_pointer_action_type_pairs(
            all_pointers_enumerated, allow_text_action_type=self.allow_text_action_type)
        self.initial_structured_prompt_format = structured_prompt_format
        self.initial_global_constraints = global_constraints
        self.pointer_action_pairs = pointer_action_pairs

        action_value_options = []
        for a, b, action_type in pointer_action_pairs:
            action_value_options.append(range(len(MAPPING_ALL_CATEGORIES[action_type])))
        self.action_value_options = action_value_options

    def find_all_multiple_choice_output_classes(self, resolved_node_format, output_format):
        if not output_format:
            return []

        # output_format = "Option {enum1}", where "enum1" is the object name
        object_name = output_format.split('{')[1].split('}')[0]

        structured_prompt_format, global_constraints = self.all_structured_prompt_formats[resolved_node_format]
        all_pointers = pointers_to_all_objects(structured_prompt_format) + global_constraints
        pointer_to_object_list = [pointer
                                  for pointer in all_pointers
                                  if 'object_name' in pointer.__dict__ and pointer.object_name == object_name]
        assert len(pointer_to_object_list) == 1
        pointer_to_object = pointer_to_object_list[0]
        return [output_format.format(**{object_name: pointer_to_object.chosen_number_format(idx)})
                for idx in pointer_to_object.enumeration_item_id_list]

    def _get_node_from_format(self, prompt_format):
        extra_params = {'print_output_fields': True, 'exclude_text_field_for_output_fields': False}
        return flatten(prompt_format.solve(extra_params)).replace('<|text|>', '{}')

    def _copy_objects_before_expanding_node(self, solved_format):
        # this function creates a copy of the passed format node (solved formats)
        # this prevents accidentally modifying the previous node when searching a tree of prompt formats

        structured_prompt_format, global_constraints = self.all_structured_prompt_formats[solved_format]
        structured_prompt_format, global_constraints = copy.deepcopy((structured_prompt_format, global_constraints))

        all_pointers = pointers_to_all_objects(structured_prompt_format) + global_constraints
        all_pointers_enumerated = [(e, i) for i, e in enumerate(all_pointers)]

        if 'all_pointers_enumerated' not in self.metadata:
            self.metadata['all_pointers_enumerated'] = [
                (str(type(e).__name__), self._get_node_from_format(e) if e.solve() else list(e.fields.keys())) for e, i
                in all_pointers_enumerated
            ]

        return structured_prompt_format, global_constraints, all_pointers_enumerated

    def list_node_accuracies(self):
        return sorted([(v, k,
                        flatten(self.all_structured_prompt_formats[k][0].solve({'print_output_fields': True})).replace(
                            '<|text|>', '{}'))
                       for k, v in self.all_structured_prompt_formats_accuracies.items()], reverse=True)

    def save(self, filename):
        import json
        to_dump = {
            # 'all_structured_prompt_formats': self.all_structured_prompt_formats,
            'generation_order': self.generation_order,
            'edges': self.edges,
            'all_structured_prompt_formats_accuracies': self.all_structured_prompt_formats_accuracies,
            'metadata': self.metadata
        }

        json.dump(to_dump, open(filename, 'w'))

    def _compute_node_score_from_resolved_prompt(self, resolved_prompt, num_samples_to_test=-1):
        last_id_analyzed = self.all_structured_prompt_formats_last_id_evaluated.get(resolved_prompt, 0)
        interval_ids_to_test = (last_id_analyzed, last_id_analyzed + num_samples_to_test) \
            if num_samples_to_test != -1 and last_id_analyzed is not None \
            else (None, None)

        # transform the multiple choice output classes to evaluate in the same format as the examples presented
        current_multiple_choice_classes = self.find_all_multiple_choice_output_classes(
            resolved_prompt, self.original_multiple_choice_output_format)
        original_to_current_multiple_choice_classes = \
            {k: v for k, v in zip(self.original_multiple_choice_classes, current_multiple_choice_classes)} \
                if self.original_multiple_choice_classes else {}

        structured_prompt_format, global_constraints = self.all_structured_prompt_formats[resolved_prompt]
        acc, history = evaluate_prompt_format(
            **self.args_compute_node_score,
            structured_prompt_format=structured_prompt_format,
            original_to_current_multiple_choice_classes=original_to_current_multiple_choice_classes,
            interval_ids_to_test=interval_ids_to_test
        )
        self.all_structured_prompt_formats_last_id_evaluated[resolved_prompt] = interval_ids_to_test[1]
        self.all_structured_prompt_formats_accuracies[resolved_prompt] = acc

        self.metadata['nodes'][resolved_prompt] = history
        return acc

    def _compute_node_score(self, structured_prompt_format, num_samples_to_test=-1):
        # return (0, 0, 0), [0]
        return self._compute_node_score_from_resolved_prompt(
            resolved_prompt=self._get_node_from_format(structured_prompt_format),
            num_samples_to_test=num_samples_to_test)

    def evaluate_node(self, solution, num_samples_to_test):

        # copy structured_prompt_format to avoid modifying the original
        resolved_prompt = self._get_node_from_format(self.initial_structured_prompt_format)
        structured_prompt_format, global_constraints, all_pointers_enumerated = \
            self._copy_objects_before_expanding_node(resolved_prompt)
        pointer_action_pairs = create_pointer_action_type_pairs(
            all_pointers_enumerated, allow_text_action_type=self.allow_text_action_type)
        assert len(self.pointer_action_pairs) == len(pointer_action_pairs)
        assert all([b == e and c == f for (a, b, c), (d, e, f) in zip(self.pointer_action_pairs, pointer_action_pairs)])

        # transform action value ids into a new structured_prompt_format
        all_action_values = []
        all_action_value_names = []
        for (element, element_id, action_type), action_value_id in zip(pointer_action_pairs, solution):
            action_value, action_value_name = MAPPING_ALL_CATEGORIES[action_type][int(action_value_id)]
            all_action_values.append(action_value)
            all_action_value_names.append(action_value_name)
            element.update_field(action_type, action_value)

        # check if value assignments are invalid, and if so give the worst possible accuracy and do not store logs about it
        # importantly, we do not store self.generation_order
        if not holistic_node_format_sanity_checks(structured_prompt_format):
            return -1e6 * (-1 if self.objective == 'lowest_accuracy' else 1)

        # update logs that do not require accuracy
        new_node = self._get_node_from_format(structured_prompt_format)
        if new_node in self.generation_order:
            self.metadata['bit_representations'][new_node].append(all_action_value_names)
            acc = self.all_structured_prompt_formats_accuracies[new_node]
            return acc[0] * (-1 if self.objective == 'lowest_accuracy' else 1)

        self.metadata['bit_representations'][new_node] = [all_action_value_names]
        self.all_structured_prompt_formats[new_node] = [structured_prompt_format, global_constraints]
        self.generation_order[new_node] = len(self.generation_order)

        # compute accuracy and update accuracy logs
        acc = self._compute_node_score(structured_prompt_format, num_samples_to_test)

        self.all_structured_prompt_formats_accuracies[new_node] = acc

        return acc[0] * (-1 if self.objective == 'lowest_accuracy' else 1)

    def main(self, value_assignments: List[List[str]], num_samples_to_test: int):
        """
        Fully evaluate all nodes (prompt formats) passed.

        :param value_assignments: Value assignments for each format, and each field of the format.
            value_assignments[i] shows all strings representing each field value for the i-th sampled format.
        :param num_samples_to_test: number of samples to consider a node fully evaluated
        """

        # convert from list(list(str)) to list(list(int))
        # this func assumes same order as in action_value_pairs, but in text (not id in array, to be robust to changes)
        value_assignments_ids = value_assignment_str_to_indices(value_assignments, self.pointer_action_pairs)

        # run all nodes
        for value_assignment in value_assignments_ids:
            self.evaluate_node(value_assignment, num_samples_to_test)


class ThompsonSamplingAlgorithmAmongPrompts(GeneticAlgorithmAmongPrompts):

    def _compute_node_score_from_resolved_prompt(self, resolved_prompt, num_samples_to_test=-1):
        last_id_analyzed = self.all_structured_prompt_formats_last_id_evaluated.get(resolved_prompt, 0)
        interval_ids_to_test = (last_id_analyzed, last_id_analyzed + num_samples_to_test) \
            if num_samples_to_test != -1 and last_id_analyzed is not None \
            else (None, None)

        if last_id_analyzed is not None and num_samples_to_test == -1:
            interval_ids_to_test = (last_id_analyzed, None)

        if last_id_analyzed is None and num_samples_to_test == -1:
            print("This means we already evaluated all samples, returning empty results.")
            return (0, 0, 0)

        if len(self.args_compute_node_score['selected_dataset_ids'][interval_ids_to_test[0]:interval_ids_to_test[1]]) == 0:
            print("This means we already evaluated all samples, returning empty results.")
            return (0, 0, 0)

        # transform the multiple choice output classes to evaluate in the same format as the examples presented
        current_multiple_choice_classes = self.find_all_multiple_choice_output_classes(
            resolved_prompt, self.original_multiple_choice_output_format)
        original_to_current_multiple_choice_classes = \
            {k: v for k, v in zip(self.original_multiple_choice_classes, current_multiple_choice_classes)} \
                if self.original_multiple_choice_classes else {}

        structured_prompt_format, global_constraints = self.all_structured_prompt_formats[resolved_prompt]
        acc, history = evaluate_prompt_format(
            **self.args_compute_node_score,
            structured_prompt_format=structured_prompt_format,
            original_to_current_multiple_choice_classes=original_to_current_multiple_choice_classes,
            interval_ids_to_test=interval_ids_to_test
        )
        self.all_structured_prompt_formats_last_id_evaluated[resolved_prompt] = interval_ids_to_test[1]
        if resolved_prompt not in self.metadata['nodes']:
            self.metadata['nodes'][resolved_prompt] = []
        self.metadata['nodes'][resolved_prompt].extend(history)
        return acc

    def _add_node_to_structures(self, solution):
        """
        This initializes nodes in our structures. It's easier to add them all at the beginning
        and then only care about sampling.
        """

        # copy structured_prompt_format to avoid modifying the original
        resolved_prompt = self._get_node_from_format(self.initial_structured_prompt_format)
        structured_prompt_format, global_constraints, all_pointers_enumerated = \
            self._copy_objects_before_expanding_node(resolved_prompt)
        pointer_action_pairs = create_pointer_action_type_pairs(
            all_pointers_enumerated, allow_text_action_type=self.allow_text_action_type)
        assert len(self.pointer_action_pairs) == len(pointer_action_pairs)
        assert all([b == e and c == f for (a, b, c), (d, e, f) in zip(self.pointer_action_pairs, pointer_action_pairs)])

        # transform action value ids into a new structured_prompt_format
        all_action_values = []
        all_action_value_names = []
        for (element, element_id, action_type), action_value_id in zip(pointer_action_pairs, solution):
            action_value, action_value_name = MAPPING_ALL_CATEGORIES[action_type][int(action_value_id)]
            all_action_values.append(action_value)
            all_action_value_names.append(action_value_name)
            element.update_field(action_type, action_value)

        # invalid node, give the worst possible accuracy and do not store logs about it
        # especially do not store self.generation_order
        if not holistic_node_format_sanity_checks(structured_prompt_format):
            assert False, "This should not happen because this is run from a file already filtered."

        # update logs that do not require accuracy
        new_node = self._get_node_from_format(structured_prompt_format)
        if new_node in self.generation_order:
            self.metadata['bit_representations'][new_node].append(all_action_value_names)
            return None

        self.metadata['bit_representations'][new_node] = [all_action_value_names]
        self.all_structured_prompt_formats[new_node] = [structured_prompt_format, global_constraints]
        self.generation_order[new_node] = len(self.generation_order)
        self.all_structured_prompt_formats_accuracies[new_node] = (0, 0, 0)  # list of CUMULATIVE accuracies

        return new_node

    def _evaluate_node_on_batch(self, new_node, num_samples):
        """
        Evaluates new_node for num_samples (i.e. one batch).
        """
        structured_prompt_format, global_constraints = self.all_structured_prompt_formats[new_node]
        acc = self._compute_node_score(structured_prompt_format, num_samples)  # (right [0, 1], wrong [0, 1], total)
        new_batch_right, new_batch_wrong, new_batch_total = acc
        right, wrong, total = self.all_structured_prompt_formats_accuracies[new_node]

        cumulative_wrong_counter = wrong * total + new_batch_wrong * new_batch_total
        cumulative_right_counter = right * total + new_batch_right * new_batch_total
        cumulative_total = new_batch_total + total
        cumulative_right = cumulative_right_counter / cumulative_total
        cumulative_wrong = cumulative_wrong_counter / cumulative_total
        self.all_structured_prompt_formats_accuracies[new_node] = (cumulative_right, cumulative_wrong, cumulative_total)

        return cumulative_total, cumulative_right_counter

    def _choose_final_node(self, num_successes, total_elements_evaluated, objective, nodes_sampled):
        accuracy_nodes = [(num_successes[node] / total_elements_evaluated[node], node) for node in nodes_sampled
                          if total_elements_evaluated[node] > 0]
        accuracy_nodes = sorted(accuracy_nodes, reverse=(objective == 'highest'))
        return accuracy_nodes[0][-1]

    def _evaluate_nodes_thompson_sampling(
            self,
            original_node,
            nodes_sampled,
            batch_size,
            max_allowed_number_of_steps=100,
            objective='lowest',
            use_ucb_rule=False,
            num_successes=None,
            total_elements_evaluated=None):

        if num_successes is None or total_elements_evaluated is None:
            total_elements_evaluated = {k: 0 for k in nodes_sampled}
            num_successes = {k: 0 for k in nodes_sampled}

            right, wrong, total = self.all_structured_prompt_formats_accuracies[original_node]
            total_elements_evaluated[original_node], num_successes[original_node] = total, right * total
        upper_bound_worst_node_accuracy = num_successes[original_node] / total_elements_evaluated[original_node]
        num_samples_in_dataset = total_elements_evaluated[original_node]

        # using EV=initial_node, we know that: a * (1 - initial_node) = initial_node * b. We initialize with b=5
        # we also avoid non-bell shape curves
        b = 5
        a = upper_bound_worst_node_accuracy / (1 - upper_bound_worst_node_accuracy) * b
        a = max(a, 1.1)
        initial_a_b_params = (a, b)

        final_nodes = []
        num_successes_list = []
        total_elements_evaluated_list = []

        for allowed_steps in range(max_allowed_number_of_steps):
            samples_list = []
            for node in nodes_sampled:
                if total_elements_evaluated[node] == num_samples_in_dataset:
                    print('node', repr(node), 'has been fully evaluated.', num_samples_in_dataset)
                    samples_list.append(1e9 if objective == 'lowest' else -1e9)
                elif use_ucb_rule:
                    success_ratio = num_successes[node] / total_elements_evaluated[node] if total_elements_evaluated[node] else 0

                    # adding one because time is one-indexed
                    time_var = allowed_steps  # time step, used to be np.sum(total_elements_evaluated[node])
                    sqrt_term = 2 * np.sqrt(np.log(1 + time_var) / total_elements_evaluated[node]) if \
                    total_elements_evaluated[node] else 0
                    samples_list.append(success_ratio + sqrt_term)
                else:
                    a = initial_a_b_params[0] + num_successes[node]
                    b = initial_a_b_params[1] + total_elements_evaluated[node] - num_successes[node]
                    samples_list.append(np.random.beta(a, b))
            if objective == 'lowest' and min(samples_list) == 1e9:
                print('Evaluated all available samples, ending. thompson_sampling')
                break
            if objective == 'highest' and max(samples_list) == -1e9:
                print('Evaluated all available samples, ending. thompson_sampling')
                break

            chosen_node_id = np.argmin(samples_list) if objective == 'lowest' else np.argmax(samples_list)
            chosen_node = nodes_sampled[chosen_node_id]
            print(f'***************** Calling model ***************** (step={allowed_steps}, objective={objective})')
            total_elements_evaluated[chosen_node], num_successes[chosen_node] = self._evaluate_node_on_batch(
                chosen_node, batch_size)
            print('total_elements_evaluated[chosen_node]', repr(chosen_node), total_elements_evaluated[chosen_node])
            final_nodes.append(
                self._choose_final_node(num_successes, total_elements_evaluated, objective, nodes_sampled))
            num_successes_list.append(copy.deepcopy(num_successes))
            total_elements_evaluated_list.append(copy.deepcopy(total_elements_evaluated))

        return final_nodes, num_successes_list, total_elements_evaluated_list

    def main(self, value_assignments, batch_size, num_formats=-1, max_allowed_number_of_model_calls=100):
        max_allowed_number_of_steps = max_allowed_number_of_model_calls // batch_size
        assert max_allowed_number_of_model_calls % batch_size == 0
        assert max_allowed_number_of_steps % 2 == 0

        # Initialize node structures
        print('Initializing node structures...')
        value_assignments_ids = value_assignment_str_to_indices(value_assignments, self.pointer_action_pairs)
        for value_assignment in value_assignments_ids:
            self._add_node_to_structures(value_assignment)
            if num_formats > 0 and len(self.generation_order) == num_formats + 1:
                break

        nodes_sampled = list(self.all_structured_prompt_formats_accuracies.keys())
        # this is already evaluated during initialization
        original_node = [new_node for new_node, order in self.generation_order.items() if order == 0][0]

        # Thompson Sampling
        budget_per_call = max_allowed_number_of_steps // 2
        print('***************** BEGINNING PHASE 1, budget:', budget_per_call)
        final_nodes, num_successes_list, total_elements_evaluated_list = self._evaluate_nodes_thompson_sampling(
            original_node,
            nodes_sampled,
            batch_size=batch_size,
            max_allowed_number_of_steps=budget_per_call,
            objective='highest',
            use_ucb_rule=False,
            num_successes=None,
            total_elements_evaluated=None)

        self.metadata['thompson_sampling'] = {}
        self.metadata['thompson_sampling']['highest-num_successes_list'] = num_successes_list
        self.metadata['thompson_sampling']['highest-total_elements_evaluated_list'] = total_elements_evaluated_list
        self.metadata['thompson_sampling']['highest-final_nodes'] = final_nodes

        best_node = final_nodes[-1]

        print('***************** BEGINNING PHASE 2, budget:', budget_per_call)
        final_node_previous_to_phase_two = self._choose_final_node(
            num_successes_list[-1], total_elements_evaluated_list[-1], 'lowest', nodes_sampled)
        final_nodes, num_successes_list, total_elements_evaluated_list = self._evaluate_nodes_thompson_sampling(
            original_node,
            nodes_sampled,
            batch_size=batch_size,
            max_allowed_number_of_steps=budget_per_call,
            objective='lowest',
            use_ucb_rule=False,
            num_successes=copy.copy(num_successes_list[-1]),
            total_elements_evaluated=copy.copy(total_elements_evaluated_list[-1]))

        worst_node = final_nodes[-1] if final_nodes else final_node_previous_to_phase_two

        self.metadata['thompson_sampling']['lowest-num_successes_list'] = num_successes_list
        self.metadata['thompson_sampling']['lowest-total_elements_evaluated_list'] = total_elements_evaluated_list
        self.metadata['thompson_sampling']['lowest-final_nodes'] = final_nodes if final_nodes else worst_node

        # these evals don't count towards the exploration budget, it's just to report final spreads found accurately
        self._evaluate_node_on_batch(best_node, num_samples=-1)
        self._evaluate_node_on_batch(worst_node, num_samples=-1)

        print('Best Node:', repr(best_node), self.all_structured_prompt_formats_accuracies[best_node])
        print('Worst Node:', repr(worst_node), self.all_structured_prompt_formats_accuracies[worst_node])