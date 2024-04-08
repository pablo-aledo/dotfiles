---
jupyter:
  jupytext:
    notebook_metadata_filter: all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
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
    version: 3.6.5
  plotly:
    description: How to use group by in Python with Plotly.
    display_as: transforms
    language: python
    layout: base
    name: Group By
    order: 2
    page_type: example_index
    permalink: python/group-by/
    thumbnail: thumbnail/groupby.jpg
---

> **Note** `transforms` are deprecated in `plotly` v5 and will be removed in a future version.

#### Basic Example

```python
import plotly.io as pio

subject = ['Moe','Larry','Curly','Moe','Larry','Curly','Moe','Larry','Curly','Moe','Larry','Curly']
score = [1,6,2,8,2,9,4,5,1,5,2,8]

data = [dict(
  type = 'scatter',
  x = subject,
  y = score,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = subject,
    styles = [
        dict(target = 'Moe', value = dict(marker = dict(color = 'blue'))),
        dict(target = 'Larry', value = dict(marker = dict(color = 'red'))),
        dict(target = 'Curly', value = dict(marker = dict(color = 'black')))
    ]
  )]
)]

fig_dict = dict(data=data)
pio.show(fig_dict, validate=False)
```

#### Reference
See https://plotly.com/python/reference/ for more information and chart attribute options!
