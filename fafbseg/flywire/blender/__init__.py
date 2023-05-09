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

import os
import navis
import platform
import subprocess

from string import Template
from tempfile import NamedTemporaryFile
from subprocess import check_call

import numpy as np
import trimesh as tm
import matplotlib.colors as mcl

from trimesh import exchange
from trimesh.util import log

from ..meshes import get_mesh_neuron


# find the current absolute path to this directory
_pwd = os.path.expanduser(os.path.abspath(os.path.dirname(__file__)))

# Use to cache templates
_CACHE = {}

NEUROPIL_MESH = None


def render_blender(x, style='workbench', fimage=None, fblend=None, neuropil=True,
                   use_flat=True, colors=None, labels=None, views=('frontal', ),
                   debug=False, res_perc=100, samples=64):
    """Render neurons using Blender.

    Parameters
    ----------
    x :         int | list of int | NeuronList
                Neurons to render.
    style :     "workbench" | "eevee"
                Preset style to use.
    fimage :    str
                Filepath to save image to.
    fblend :    str, optional
                Filepath to save blender file to.
    neuropil :  bool
                Whether to render neuropil in top.
    use_flat :  bool
                Whether to use flat segmentation for version 526. Only relevant
                if `x` is root ID(s).
    colors :    dict | list | str | tuple
                List of colors, one for each neuron.
    labels :    list, optional
                One label per mesh. If provided will use these for names in the
                Blender file.
    views :     tuple
                Combination of 'frontal' and/or 'lateral'.
    res_perc :  int
                Resolution in percent of 1920x1080 to render in.
    samples :   int
                Number of samples to use for rendering.
    debug :     bool
                If True, will print Blender's console to stdout.

    """
    if not tm.interfaces.blender.exists:
        raise ImportError('Blender 3D unavailable (no executable not found).')
    _blender_executable = tm.interfaces.blender._blender_executable

    if isinstance(views, str):
        views = (views, )

    assert style in ('workbench', 'eevee')
    for v in views:
        assert v in ('frontal', 'lateral', 'dorsal')

    if isinstance(x, navis.MeshNeuron):
        meshes = [x.trimesh]
    elif isinstance(x, navis.Volume):
        meshes = [tm.Trimesh(x.vertices, x.faces)]
    elif isinstance(x, tm.Trimesh):
        meshes = [x]
    elif isinstance(x, navis.NeuronList):
        meshes = []
        for n in x:
            if isinstance(n, navis.MeshNeuron):
                meshes.append(n.trimesh)
            elif isinstance(n, navis.TreeNeuron):
                meshes.append(navis.conversion.tree2meshneuron(n))
    elif isinstance(x, list):
        meshes = []
        for o in x:
            if isinstance(o, navis.NeuronList):
                for n in o:
                    if isinstance(n, navis.MeshNeuron):
                        meshes.append(n.trimesh)
                    elif isinstance(n, navis.TreeNeuron):
                        meshes.append(navis.conversion.tree2meshneuron(n))
            elif isinstance(o, tm.Trimesh):
                meshes.append(o)
            elif isinstance(o, navis.MeshNeuron):
                meshes.append(o.trimesh)
            else:
                raise TypeError(f'Unable to render {type(o)}')
    else:
        meshes = get_mesh_neuron(x, dataset='production' if not use_flat else 'flat_630')
        if isinstance(x, navis.NeuronList):
            meshes = [n.trimesh for n in meshes]
        else:
            meshes = [meshes.trimesh]

    zero_faces = np.array([len(m.faces) == 0 for m in meshes])
    if any(zero_faces):
        raise ValueError(f'{zero_faces.sum()} meshes have no faces')

    meshes = {f'mesh_{i}': m for i, m in enumerate(meshes)}

    if isinstance(colors, str):
        colors = [mcl.to_rgba(colors)] * len(meshes)
    elif isinstance(colors, tuple):
        if len(colors) == 3:
            colors = (colors[0], colors[1], colors[2], 1)
        colors = [(colors)] * len(meshes)
    elif isinstance(colors, (list, np.ndarray)):
        if len(colors) != len(meshes):
            raise ValueError(f'Got {len(colors)} for {len(meshes)} meshes.')
        colors = [mcl.to_rgba(c) for c in colors]
    else:
        colors = [(.9, .9, .9, 1)] * len(meshes)

    if neuropil:
        import flybrains
        global NEUROPIL_MESH
        if not NEUROPIL_MESH:
            NEUROPIL_MESH = flybrains.FLYWIRE.mesh
        meshes['neuropil'] = NEUROPIL_MESH

    # Load the template
    temp_name = 'blender_render.py.template'
    if temp_name in _CACHE:
        template = _CACHE[temp_name]
    else:
        with open(os.path.join(_pwd, 'templates', temp_name), 'r') as f:
            template = f.read()
        _CACHE[temp_name] = template

    # Let trimesh's MeshScript take care of exectution and clean-up
    with RenderScript(meshes=meshes,
                      script=template,
                      save=fblend,
                      image=fimage,
                      colors=colors,
                      res_perc=res_perc,
                      samples=samples,
                      style=style,
                      views=views,
                      labels=labels,
                      debug=debug) as blend:
        result = blend.run(_blender_executable
                           + ' --background --python $SCRIPT')

    return


class RenderScript:
    def __init__(self,
                 meshes,
                 script,
                 style='workbench',
                 image=None,
                 save=None,
                 debug=False,
                 colors=None,
                 labels=None,
                 res_perc=100,
                 samples=64,
                 views=('frontal', ),
                 keep_temp_files=False,
                 **kwargs):

        self.debug = debug
        self.kwargs = kwargs
        self.meshes = meshes
        self.script = script
        self.save = save
        self.image = image
        self.colors = colors
        self.res_perc = res_perc
        self.keep_temp_files = keep_temp_files
        self.style = style
        self.views = views
        self.samples = samples
        self.labels = labels

        assert isinstance(self.meshes, dict)
        assert isinstance(self.labels, (type(None), list))

    def __enter__(self):
        # Windows has problems with multiple programs using open files so we close
        # them at the end of the enter call, and delete them ourselves at exit

        # Due to the GLB header encoding file size as uint32 we must not create
        # files greater than 4,294,967,295 bytes. File size mostly depends on
        # the number of vertices and faces but there is some extra overhead.
        # We will just try to stay well short of the max file size
        nbytes = 0
        batches = [[]]
        for k, m in self.meshes.items():
            m._label = k  # track order of meshes across files
            nbytes += m.vertices.nbytes + m.faces.nbytes
            # If we go over 85% of the max size start a new batch
            if nbytes > (4_294_967_295 * .85):
                nbytes = 0
                batches.append([])
            batches[-1].append(m)

        # Make as many files as we have batches
        self.mesh_files = [NamedTemporaryFile(suffix='.gbl',
                                              mode='wb',
                                              delete=False) for b in batches]
        self.script_out = NamedTemporaryFile(
            mode='wb', delete=False)

        # export the meshes to a temporary GBL container
        for batch, file_obj in zip(batches, self.mesh_files):
            # note: passing a dict will set the object names in the Blender file
            scene = tm.Scene({f'{m._label}': m for m in batch})
            _ = tm.exchange.export.export_mesh(scene,
                                               file_obj.name,
                                               file_type='glb')
        self.replacement = {}
        self.replacement['MESH_FILES'] = str([f.name for f in self.mesh_files])
        self.replacement['SCRIPT'] = self.script_out.name

        self.replacement['SAVE'] = self.save is not None
        self.replacement['FILEPATH'] = self.save
        self.replacement['RENDER'] = self.image is not None
        self.replacement['IMAGEPATH'] = self.image
        self.replacement['COLORS'] = self.colors
        self.replacement['RESOLUTION_PERC'] = self.res_perc
        self.replacement['STYLE'] = self.style
        self.replacement['VIEWS'] = self.views
        self.replacement['SAMPLES'] = self.samples
        self.replacement['LABELS'] = self.labels

        script_text = Template(self.script).substitute(self.replacement)
        if platform.system() == 'Windows':
            script_text = script_text.replace('\\', '\\\\')
        self.script_out.write(script_text.encode('utf-8'))

        # close all temporary files
        self.script_out.close()
        for file_obj in self.mesh_files:
            file_obj.close()
        return self

    def run(self, command):
        command_run = Template(command).substitute(
            self.replacement).split()
        # run the binary
        # avoid resourcewarnings with null
        with open(os.devnull, 'w') as devnull:
            startupinfo = None
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            if self.debug:
                # in debug mode print the output
                stdout = None
            else:
                stdout = devnull

            if self.debug:
                log.info('executing: {}'.format(' '.join(command_run)))
            check_call(command_run,
                       stdout=stdout,
                       stderr=subprocess.STDOUT,
                       startupinfo=startupinfo)

        return

    def __exit__(self, *args, **kwargs):
        if self.keep_temp_files:
            return
        # delete all the temporary files by name
        # they are closed but their names are still available
        os.remove(self.script_out.name)
        for file_obj in self.mesh_files:
            os.remove(file_obj.name)
