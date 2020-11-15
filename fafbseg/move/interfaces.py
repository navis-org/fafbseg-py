#    A collection of tools to interface with manually traced and autosegmented
#    data in FAFB.
#
#    Copyright (C) 2019 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
import numpy as np
import navis

from .. import utils

import inquirer
from inquirer.themes import GreenPassion

# This is to prevent FutureWarning from numpy (via vispy)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

use_pbars = utils.use_pbars


def confirm_overlap(x, fragments, viewer=None):
    """Show dialogs to confirm overlapping fragments."""
    print('{}: {} overlapping fragments found'.format(x.name, len(fragments)))
    if fragments:
        fragments.sort_values('n_nodes')
        # Have user inspect fragments
        # Show larger fragments in 3d viewer
        if any(fragments.n_nodes > 10):
            # Generate a summary
            large_frags = fragments[fragments.n_nodes > 10]
            s = large_frags.summary(add_props=['overlap_score', 'id'])[['name',
                                                                        'id',
                                                                        'n_nodes',
                                                                        'n_connectors',
                                                                        'overlap_score']]
            # Show and let user decide which ones to merge
            if not viewer:
                viewer = navis.Viewer(title='Check overlap')
            # Make sure viewer is actually visible and cleared
            viewer.show()
            viewer.clear()
            # Add original skeleton
            viewer.add(x, color='w')
            viewer.add(large_frags)
            viewer.picking = True
            viewer._picking_text.visible = True
            viewer.show_legend = True

            # Print summary
            print('Large (>10 nodes) overlapping fragments:')
            print(s.to_string(index=False, show_dimensions=False))

            msg = """
            Please check these large fragments for overlap and deselect
            neurons that you DO NOT want to have merged by clicking on
            their names in the legend.
            Hit ENTER when you are ready to proceed or CTRL-C to cancel.
            """

            try:
                _ = input(msg)
            except KeyboardInterrupt:
                raise KeyboardInterrupt('Merge process aborted by user.')
            except BaseException:
                raise

            # Remove deselected fragments
            # Mind you not all fragments are on viewer - this is why we remove
            # neurons that has been hidden
            fragments = fragments[~np.isin(fragments.id, viewer.invisible)]

    # Now ask for smaller fragments via CLI
    if fragments:
        s = fragments.summary(add_props=['overlap_score',
                                         'sampler_count', 'id'])[['name',
                                                                  'id',
                                                                  'n_nodes',
                                                                  'n_connectors',
                                                                  'sampler_count',
                                                                  'overlap_score']]

        # Ask user which neuron should be merged
        msg = """
        Please check the fragments that potentially overlap with the input neuron (white).
        Deselect those that should NOT be merged using the arrows keys.
        Hit ENTER when you are ready to proceed or CTRL-C to abort
        """
        print(msg)

        msg = s.to_string(index=False).split('\n')[0]

        s_str = s.to_string(index=False, show_dimensions=False, header=False)
        choices = [(v, i) for i, v in enumerate(s_str.split('\n'))]
        q = [inquirer.Checkbox(name='selection',
                               message=msg,
                               choices=choices,
                               default=list(range(len(choices))))]

        # Ask the question
        selection = inquirer.prompt(q, theme=GreenPassion()).get('selection')

        if isinstance(selection, type(None)):
            raise SystemExit('Merge process aborted by user.')

        # Remove fragments that are not selected
        if selection:
            fragments = fragments[selection]
        else:
            # If no selection, remove all neurons from the list
            fragments = fragments[:0]

    # If no overlapping fragments (either none from the start or all removed
    # during filtering) ask if just proceed with upload
    if not fragments:
        print('No overlapping fragments to be merged into in target instance.')
        msg = 'Proceed with just uploading this neuron?'
        q = [inquirer.Confirm(name='confirm', message=msg)]
        confirm = inquirer.prompt(q, theme=GreenPassion()).get('confirm')

        if not confirm:
            raise SystemExit('Merge process aborted by user.')

        base_neuron = None
    # If any fragments left, ask for base neuron
    else:
        # Ask user which neuron to use as merge target
        s = fragments.summary(add_props=['overlap_score',
                                         'sampler_count',
                                         'id'])[['name',
                                                 'id',
                                                 'n_nodes',
                                                 'n_connectors',
                                                 'sampler_count',
                                                 'overlap_score']]

        msg = """
        Above fragments and your input neuron will be merged into a single neuron.
        All annotations will be preserved but only the neuron used as merge target
        will keep its name and skeleton ID.
        Please select the neuron you would like to use as merge target!
        """ + s.to_string(index=False).split('\n')[0]
        print(msg)

        s_str = s.to_string(index=False, show_dimensions=False, header=False)
        choices = [(v, i) for i, v in enumerate(s_str.split('\n'))]
        q = [inquirer.List(name='base_neuron',
                           message='Choose merge target',
                           choices=choices)]
        # Ask the question
        bn = inquirer.prompt(q, theme=GreenPassion()).get('base_neuron')

        if isinstance(bn, type(None)):
            raise ValueError("Merge aborted by user")

        base_neuron = fragments[bn]

        # Some safeguards:
        # Check if we would delete any samplers
        cond1 = s.id != base_neuron.id
        cond2 = s.sampler_count > 0
        has_sampler = s[cond1 & cond2]
        if not has_sampler.empty:
            print("Merging selected fragments would delete reconstruction "
                  "samplers on the following neurons:")
            print(has_sampler)
            q = [inquirer.Confirm(name='confirm', message='Proceed anyway?')]
            confirm = inquirer.prompt(q, theme=GreenPassion())['confirm']

            if not confirm:
                raise SystemExit('Merge process aborted by user.')

        # Check if we would generate any 2-soma neurons
        has_soma = [not isinstance(s, type(None)) for s in fragments.soma]
        if sum(has_soma) > 1:
            print('Merging the selected fragments would generate a neuron  '
                  'with two somas!')
            q = [inquirer.Confirm(name='confirm', message='Proceed anyway?')]
            confirm = inquirer.prompt(q, theme=GreenPassion())['confirm']

            if not confirm:
                raise SystemExit('Merge process aborted by user.')

    return fragments, base_neuron
