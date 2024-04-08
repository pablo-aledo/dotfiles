---
jupyter:
  jupytext:
    notebook_metadata_filter: all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.11
  plotly:
    description: Using WebGL for increased speed, improved interactivity, and the
      ability to plot even more data!
    display_as: basic
    language: python
    layout: base
    name: WebGL vs SVG
    order: 14
    permalink: python/webgl-vs-svg/
    redirect_from: python/compare-webgl-svg/
    thumbnail: thumbnail/webgl.jpg
---

### SVG and canvas/WebGL: two browser capabilities for rendering

`plotly` figures are rendered by web browsers, which broadly speaking have two families of capabilities for rendering graphics: the SVG API which supports vector rendering, and the Canvas API which supports raster rendering, and can exploit GPU hardware acceleration via a browser technology known as WebGL. Each `plotly` trace type is primarily rendered with either SVG or WebGL, although WebGL-powered traces also use some SVG. The following trace types use WebGL for part or all of the rendering:

* Accelerated versions of SVG trace types: `scattergl`, `scatterpolargl`, `heatmapgl`
* High-performance multidimensional trace types: `splom`, or `parcoords`
* 3-d trace types `scatter3d`, `surface`, `mesh3d`, `cone`, `streamtube`
* Mapbox Gl JS-powered trace types: `scattermapbox`, `choroplethmapbox`, `densitymapbox`

### WebGL Limitations and Tradeoffs

WebGL is a powerful technology for accelerating computation but comes with some strict limitations:

1. GPU requirement: WebGL is a GPU (graphics card) technology and therefore requires specific hardware which is available in most but not all cases and is supported by most but not all browsers
2. Rasterization: WebGL-rendered data is drawn as a grid of pixels rather than as individual shapes, so can appear pixelated or fuzz in certain cases, and when exported to static file formats will appear pixelated on zoom. In addition: text rendering will differ between SVG and WebGL-powered traces.
3. Context limits: browsers impose a strict limit on the number of WebGL "contexts" that any given web document can access. WebGL-powered traces in `plotly` can use multiple contexts in some cases but as a general rule, **it may not be possible to render more than 8 WebGL-involving figures on the same page at the same time.**
4. Size limits: browsers impose hardware-dependent limits on the height and width of figures using WebGL which users may encounter with extremely large plots (e.g. tens of thousands of pixels of height)

In addition to the above limitations, the WebGL-powered version of certain SVG-powered trace types (`scattergl`, `scatterpolargl`, `heatmapgl`) are not complete drop-in replacements for their SVG counterparts yet
* Available symbols will differ
* Area fills are not yet supported in WebGL
* Range breaks on time-series axes are not yet supported
* Axis range heuristics may differ

### Multiple WebGL Contexts

*New in 5.19*

Most browsers have a limit of between 8 and 16 WebGL contexts per page. A Plotly WebGL-based figure may use multiple WebGL contexts, but generally you'll be able to render between 4 and 8 figures on one page.

If you exceed the browser limit on WebGL contexts, some figures won't render and you'll see an error. In the console in Chrome, for example, you'll see the error: "Too many active WebGL contexts. Oldest context will be lost".

If you encounter WebGL context limits when using WebGL-based figures, you can use [Virtual WebGL](https://github.com/greggman/virtual-webgl), which virtualizes a single WebGL context into multiple contexts.

To use it, in the environment where your Plotly figures are being rendered, load the Virtual WebGL script, "https://unpkg.com/virtual-webgl@1.0.6/src/virtual-webgl.js", for example, using a `<script>` tag.

In a Jupyter notebook environment that supports magic commands, you can load it with the [HTML magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-html):

```
%%html
<script src=“https://unpkg.com/virtual-webgl@1.0.6/src/virtual-webgl.js”></script>
```

### WebGL for Scatter Performance

In the examples below we show that it is possible to represent up to around a million points with WebGL-enabled traces.
For larger datasets, or for a clearer visualization of the density of points,
it is also possible to use [datashader](/python/datashader/).

### WebGL with Plotly Express

The `rendermode` argument to supported Plotly Express functions (e.g. `scatter` and `scatter_polar`) can be used to enable WebGL rendering.

> **Note** The default `rendermode` is `"auto"`, in which case Plotly Express will automatically set `rendermode="webgl"` if the input data is more than 1,000 rows long. If WebGL acceleration is *not* desired in this case, `rendermode` can be forced to `"svg"` for vectorized, if slower, rendering.

Here is an example that creates a 100,000 point scatter plot using Plotly Express with WebGL rendering explicitly enabled.

```python
import plotly.express as px

import pandas as pd
import numpy as np
np.random.seed(1)

N = 100000

df = pd.DataFrame(dict(x=np.random.randn(N),
                       y=np.random.randn(N)))

fig = px.scatter(df, x="x", y="y", render_mode='webgl')

fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'))

fig.show()
```


#### WebGL with 100,000 points with Graph Objects

If Plotly Express does not provide a good starting point, it is also possible to use [the more generic `go.Scattergl` class from `plotly.graph_objects`](/python/graph-objects/).

```python
import plotly.graph_objects as go

import numpy as np

N = 100000

# Create figure
fig = go.Figure()

fig.add_trace(
    go.Scattergl(
        x = np.random.randn(N),
        y = np.random.randn(N),
        mode = 'markers',
        marker = dict(
            line = dict(
                width = 1,
                color = 'DarkSlateGrey')
        )
    )
)

fig.show()
```

#### WebGL Rendering with 1 Million Points

```python
import plotly.graph_objects as go

import numpy as np

N = 1000000

# Create figure
fig = go.Figure()

fig.add_trace(
    go.Scattergl(
        x = np.random.randn(N),
        y = np.random.randn(N),
        mode = 'markers',
        marker = dict(
            line = dict(
                width = 1,
                color = 'DarkSlateGrey')
        )
    )
)

fig.show()
```

#### WebGL with many traces

```python
import plotly.graph_objects as go

import numpy as np

fig = go.Figure()

trace_num = 10
point_num = 5000
for i in range(trace_num):
    fig.add_trace(
        go.Scattergl(
                x = np.linspace(0, 1, point_num),
                y = np.random.randn(point_num)+(i*5)
        )
    )

fig.update_layout(showlegend=False)

fig.show()
```

### Reference

See https://plotly.com/python/reference/scattergl/ for more information and chart attribute options!
