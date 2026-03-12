# iongraph-go

A Go implementation of the [iongraph layout algorithm](https://spidermonkey.dev/blog/2025/10/28/iongraph-web.html)
that renders Mermaid-style control flow graphs to SVG.

## Build

```bash
go build -o iongraph iongraph.go
```

## Usage

```bash
# From file
iongraph-go input.mmd output.svg

# From stdin
iongraph-go < input.mmd > output.svg
```

## Input Format (subset of Mermaid)

```
graph TD
  A["Entry block"]
  B["Loop Header"]:::loopheader
  C["Loop Body"]
  D["Backedge"]:::backedge
  E["Exit"]

  A --> B
  B --> C
  C --> D
  D --> B
  B --> E
```

### Node types

| Syntax                | Meaning                       | Visual style      |
|-----------------------|-------------------------------|-------------------|
| `A["label"]`          | Normal block                  | White rectangle   |
| `A["label"]:::loopheader` | Loop header block         | Green rectangle   |
| `A["label"]:::backedge`   | Back edge (loop end)      | Yellow rectangle  |

If no `:::loopheader` or `:::backedge` annotations are present, loops are
detected automatically via DFS.

## Layout Algorithm

Implements the 5-step iongraph algorithm:

1. **Loop detection** — back edges identified by DFS or explicit annotation
2. **Loop depth assignment** — each block gets a loop depth and loop-header ID  
3. **Layering** — blocks assigned to horizontal layers; loop bodies are pushed
   down so exits from a loop appear *below* the entire loop body
4. **Node placement** — iterative straightening pulls children under parents,
   dummy nodes bridge multi-layer edges
5. **SVG rendering** — nodes drawn as rounded rectangles, edges as
   rounded-corner paths with arrowheads; back edges drawn as dashed curves
