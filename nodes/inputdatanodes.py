#!/usr/bin/python
'''Object detection node.'''
# pylint: disable=no-member
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

class InputData:
    '''A node that takes any value and displays it as a string.
    '''

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_data": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    INPUT_IS_LIST = (True,)
    OUTPUT_NODE = True

    CATEGORY = "🧬 Circle Detection Nodes"
    FUNCTION = "input_data"

    def input_data(self, input_data):
        '''Input data.'''
        return input_data
