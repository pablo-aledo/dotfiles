# -*- coding: utf-8 -*-
from .__module__ import Module, dependency, source
from .python import Python


@dependency(Python)
@source('pip')
class Chainer(Module):

    def build(self):
        return r'''
            $PIP_INSTALL \
            '''.rstrip() + (
                '' if self.composer.cpu_only else \
                r'''
                cupy \
                '''.rstrip()
            ) + r'''
                chainer \
                && \
        '''
