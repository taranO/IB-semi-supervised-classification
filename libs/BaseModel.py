import sys

# ======================================================================================================================
class BaseModel():
    def __init__(self):
        super(BaseModel, self).__init__(name='base')

    def debug_print(self, x):

        is_debug = False
        gettrace = getattr(sys, 'gettrace', None)

        if gettrace():
            is_debug = True

        if is_debug:
            print(x)

