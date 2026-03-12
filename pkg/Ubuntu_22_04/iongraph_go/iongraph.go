// iongraph-go: Mermaid-style CFG → SVG
// Layout algorithm based on https://spidermonkey.dev/blog/2025/10/28/iongraph-web.html
//
// Usage:
//   iongraph-go [input.mmd] [output.svg]
//   iongraph-go < input.mmd > output.svg
//
// Input subset of Mermaid flowchart:
//   graph TD
//     A["Entry"]
//     B["Loop"]:::loopheader
//     C["Body"]
//     D["Backedge"]:::backedge
//     E["Exit"]
//     A --> B
//     B --> C
//     C --> D
//     D --> B
//     B --> E

package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
)

// ─── Layout constants ─────────────────────────────────────────────────────────

const (
	blockW      = 130.0
	blockH      = 44.0
	blockGapX   = 50.0 // horizontal gap between blocks in same layer
	layerGapY   = 80.0 // vertical gap between layers
	canvasPadX  = 40.0
	canvasPadY  = 40.0
	arrowR      = 10.0
	arrowTip    = 7.0
	portStart   = 16.0
	portStep    = 18.0
)

// ─── Graph data ───────────────────────────────────────────────────────────────

type Block struct {
	id    string
	label string

	succs []string
	preds []string

	isLoopHeader bool
	isBackedge   bool

	// Set by assignLoopInfo
	loopDepth    int
	loopHeaderID string // ID of the innermost loop header containing this block ("" = top-level)

	// Set by assignLayers
	layer int // -1 = unassigned

	// Loop-header bookkeeping (only valid when isLoopHeader=true)
	loopHeight    int      // filled in during layering
	parentLHID    string   // ID of parent loop header ("" = none)
	deferredSuccs []string // successors deferred until loop is fully layered
}

type Graph struct {
	blocks    map[string]*Block
	order     []string // insertion order
	backEdges map[[2]string]bool // set of (from,to) pairs that are back-edges (auto-detected only)
}

func newGraph() *Graph { return &Graph{blocks: map[string]*Block{}, backEdges: map[[2]string]bool{}} }

func (g *Graph) getOrAdd(id string) *Block {
	if b, ok := g.blocks[id]; ok {
		return b
	}
	b := &Block{id: id, label: id, layer: -1}
	g.blocks[id] = b
	g.order = append(g.order, id)
	return b
}

func (g *Graph) addEdge(from, to string) {
	f := g.getOrAdd(from)
	t := g.getOrAdd(to)
	if !hasStr(f.succs, to) {
		f.succs = append(f.succs, to)
	}
	if !hasStr(t.preds, from) {
		t.preds = append(t.preds, from)
	}
}

func hasStr(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func addUniq(s []string, v string) []string {
	if !hasStr(s, v) {
		s = append(s, v)
	}
	return s
}

// ─── Mermaid parser ───────────────────────────────────────────────────────────

var (
	reEdge    = regexp.MustCompile(`(\w+)(?:\[[^\]]*\])?\s*--[->]+\s*(\w+)`)
	reLabelDQ = regexp.MustCompile(`(\w+)\["([^"]+)"\]`)
	reLabelSQ = regexp.MustCompile(`(\w+)\[([^\]]+)\]`)
	reClass   = regexp.MustCompile(`(\w+):::(\w+)`)
)

func parseMermaid(sc *bufio.Scanner) *Graph {
	g := newGraph()
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "graph") ||
			strings.HasPrefix(line, "%%") || strings.HasPrefix(line, "//") {
			continue
		}
		// class annotations  A:::loopheader
		for _, m := range reClass.FindAllStringSubmatch(line, -1) {
			b := g.getOrAdd(m[1])
			switch m[2] {
			case "loopheader":
				b.isLoopHeader = true
			case "backedge":
				b.isBackedge = true
			}
		}
		// labels
		if m := reLabelDQ.FindStringSubmatch(line); m != nil {
			g.getOrAdd(m[1]).label = m[2]
		} else if m := reLabelSQ.FindStringSubmatch(line); m != nil {
			g.getOrAdd(m[1]).label = m[2]
		}
		// edges
		for _, m := range reEdge.FindAllStringSubmatch(line, -1) {
			g.addEdge(m[1], m[2])
		}
	}
	return g
}

// ─── Step 0: Detect back-edges ───────────────────────────────────────────────
//
// Two modes:
//   • Explicit: user annotated nodes with :::backedge / :::loopheader.
//     We respect those as-is (iongraph IR convention where a back-edge is
//     represented as a dedicated block node).
//   • Auto: no annotations. We run DFS to find back-edges and record them
//     as (from→to) pairs in g.backEdges. The nodes themselves are NOT marked
//     isBackedge; they keep their normal layer and rendering.

func (g *Graph) autoDetectLoops() {
	// If any explicit annotations exist, skip auto-detection entirely.
	for _, b := range g.blocks {
		if b.isLoopHeader || b.isBackedge {
			return
		}
	}
	color := map[string]int{} // 0=white 1=gray 2=black
	var dfs func(id string)
	dfs = func(id string) {
		color[id] = 1
		for _, s := range g.blocks[id].succs {
			switch color[s] {
			case 0:
				dfs(s)
			case 1: // back-edge: id → s
				g.backEdges[[2]string{id, s}] = true
				g.blocks[s].isLoopHeader = true
			}
		}
		color[id] = 2
	}
	for _, id := range g.order {
		if color[id] == 0 {
			dfs(id)
		}
	}
}

// isBackEdge returns true if the edge from→to is a back-edge.
// For explicit :::backedge blocks: the edge FROM the backedge block to its successor.
// For auto-detected: the (from,to) pair stored in g.backEdges.
func (g *Graph) isBackEdge(from, to string) bool {
	src := g.blocks[from]
	if src != nil && src.isBackedge {
		return true // edge leaving an explicit backedge block
	}
	return g.backEdges[[2]string{from, to}]
}

// skipForLayout returns true if an edge from→to should be skipped during
// layer assignment (i.e. it is a back-edge in the layout sense).
// For explicit :::backedge nodes, the NODE itself is the signal.
// For auto-detected back-edges, the EDGE pair is the signal.
func (g *Graph) skipForLayout(from, to string) bool {
	dst := g.blocks[to]
	if dst != nil && dst.isBackedge {
		return true // explicit backedge node: handled by assignLayers specially
	}
	return g.backEdges[[2]string{from, to}]
}

// ─── Step 1: Assign loop depth and loop-header membership ────────────────────
//
// For each loop header L, compute its member set by forward BFS from L that
// stays within the loop (only nodes that can reach L's backedge).
// Then assign each block to its deepest containing loop header.

func (g *Graph) assignLoopInfo() {
	// Build reverse-adjacency over non-backedge edges (for backwards reachability)
	revEdges := map[string][]string{}
	for _, b := range g.blocks {
		for _, s := range b.succs {
			if !g.skipForLayout(b.id, s) {
				revEdges[s] = append(revEdges[s], b.id)
			}
		}
	}

	// For each loop header, compute which blocks are "inside" it.
	// Forward BFS from LH (skipping back-edges), backward BFS from the
	// node(s) that carry the back-edge to LH.
	type loopDef struct {
		id      string
		members map[string]bool
	}
	var loopDefs []loopDef

	for _, lhID := range g.order {
		lh := g.blocks[lhID]
		if !lh.isLoopHeader {
			continue
		}
		// Forward BFS from lhID (skip back-edges)
		fwd := map[string]bool{}
		q := []string{lhID}
		fwd[lhID] = true
		for len(q) > 0 {
			cur := q[0]; q = q[1:]
			for _, s := range g.blocks[cur].succs {
				if !g.skipForLayout(cur, s) && !fwd[s] {
					fwd[s] = true
					q = append(q, s)
				}
			}
		}
		// Find all nodes that carry a back-edge pointing to lhID:
		//   - explicit: nodes whose isBackedge=true and have lhID as successor
		//   - auto-detected: any (from, lhID) in g.backEdges
		var beNodes []string
		for _, pid := range lh.preds {
			if g.skipForLayout(pid, lhID) {
				beNodes = append(beNodes, pid)
			}
		}
		// Backward BFS from those back-edge source nodes
		back := map[string]bool{}
		var bq []string
		for _, beID := range beNodes {
			if !back[beID] {
				back[beID] = true
				bq = append(bq, beID)
			}
		}
		for len(bq) > 0 {
			cur := bq[0]; bq = bq[1:]
			for _, prev := range revEdges[cur] {
				if fwd[prev] && !back[prev] {
					back[prev] = true
					bq = append(bq, prev)
				}
			}
		}
		// Members = forward-reachable ∩ backward-reachable (can reach the back-edge)
		members := map[string]bool{lhID: true}
		for id := range fwd {
			if back[id] {
				members[id] = true
			}
		}
		loopDefs = append(loopDefs, loopDef{lhID, members})
	}

	// Sort by member count descending (outer loops first, inner loops later
	// so inner assignments override outer ones)
	sort.Slice(loopDefs, func(a, b int) bool {
		return len(loopDefs[a].members) > len(loopDefs[b].members)
	})

	// Assign loopHeaderID = deepest loop header containing each block
	for _, b := range g.blocks {
		b.loopHeaderID = ""
		b.loopDepth = 0
		b.parentLHID = ""
		if b.isLoopHeader {
			b.loopHeight = 0
			b.deferredSuccs = nil
		}
	}
	for _, loop := range loopDefs {
		for mID := range loop.members {
			b := g.blocks[mID]
			// Override with deeper loop (smaller member count = more nested)
			curSz := len(g.blocks) + 1
			if b.loopHeaderID != "" {
				for _, ld := range loopDefs {
					if ld.id == b.loopHeaderID {
						curSz = len(ld.members)
						break
					}
				}
			}
			if len(loop.members) < curSz {
				b.loopHeaderID = loop.id
			}
		}
	}
	// A loop header with no outer loop → set loopHeaderID to itself
	for _, b := range g.blocks {
		if b.isLoopHeader && b.loopHeaderID == "" {
			b.loopHeaderID = b.id
		}
	}

	// Assign parentLHID for loop headers: the smallest loop that contains lhID (other than itself)
	for _, loop := range loopDefs {
		lh := g.blocks[loop.id]
		bestSz := len(g.blocks) + 1
		for _, outer := range loopDefs {
			if outer.id == loop.id {
				continue
			}
			if outer.members[loop.id] && len(outer.members) < bestSz {
				bestSz = len(outer.members)
				lh.parentLHID = outer.id
			}
		}
	}

	// Assign loopDepth = length of loop-header ancestor chain
	for _, b := range g.blocks {
		d := 0
		lhID := b.loopHeaderID
		seen := map[string]bool{}
		for lhID != "" && lhID != b.id && !seen[lhID] {
			seen[lhID] = true
			lh := g.blocks[lhID]
			if lh == nil {
				break
			}
			// Only count if we're NOT the loop header itself
			if lhID != b.id {
				d++
			}
			lhID = lh.parentLHID
		}
		b.loopDepth = d
	}
}

// ─── Step 2: Assign layers ────────────────────────────────────────────────────
//
// Recursive post-order walk. Each block gets layer = max over all paths.
// Backedge blocks are placed at the same layer as their target (the loop
// header). Outgoing loop exits are deferred until the entire loop body is
// layered, then placed below it.

// isInsideLoop reports whether block `id` is nested inside the loop headed by `loopID`.
// A block is inside if its loopHeaderID == loopID, or any ancestor loop header
// has parentLHID == loopID, etc.
func (g *Graph) isInsideLoop(id, loopID string) bool {
	b := g.blocks[id]
	if b == nil {
		return false
	}
	lhID := b.loopHeaderID
	for lhID != "" {
		if lhID == loopID {
			return true
		}
		lh := g.blocks[lhID]
		if lh == nil {
			break
		}
		lhID = lh.parentLHID
	}
	return false
}

func (g *Graph) assignLayers(id string, layer int) {
	b := g.blocks[id]

	// Explicit backedge node: same row as the loop header it jumps to
	if b.isBackedge {
		if len(b.succs) > 0 {
			b.layer = g.blocks[b.succs[0]].layer
		}
		return
	}

	// Early-out: already placed at this depth or deeper
	if layer <= b.layer {
		return
	}

	b.layer = layer

	// Update loopHeight on every enclosing loop header
	lhID := b.loopHeaderID
	for lhID != "" {
		lh := g.blocks[lhID]
		if lh == nil {
			break
		}
		if h := b.layer - lh.layer + 1; h > lh.loopHeight {
			lh.loopHeight = h
		}
		lhID = lh.parentLHID
	}

	// Recurse into successors
	for _, sID := range b.succs {
		// Skip back-edges entirely during layer assignment
		if g.skipForLayout(id, sID) {
			continue
		}
		s := g.blocks[sID]
		// An edge exits the current loop when the successor lives outside
		// the loop body we are currently layering.
		exitsLoop := b.loopHeaderID != "" &&
			s.loopHeaderID != b.loopHeaderID &&
			!g.isInsideLoop(sID, b.loopHeaderID)
		if exitsLoop {
			lh := g.blocks[b.loopHeaderID]
			if lh != nil {
				lh.deferredSuccs = addUniq(lh.deferredSuccs, sID)
			}
		} else {
			g.assignLayers(sID, layer+1)
		}
	}

	// After the loop body is done, lay out the deferred exits
	if b.isLoopHeader {
		for _, sID := range b.deferredSuccs {
			g.assignLayers(sID, layer+b.loopHeight)
		}
	}
}

// ─── LayoutNode ───────────────────────────────────────────────────────────────

type Node struct {
	nid     int
	blockID string  // "" = dummy
	dstBlk  string  // for dummies: final destination block

	x, y float64
	w, h float64

	srcs     []*Node
	dsts     []*Node // indexed by outgoing port
	backPort map[int]bool // port indices that are back-edges
	backSrc  map[*Node]bool // src nodes connected via back-edge
	jointOffset []float64 // per-port vertical offset of horizontal segment
}

func (n *Node) dummy() bool { return n.blockID == "" }

func link(src *Node, port int, dst *Node) {
	for len(src.dsts) <= port {
		src.dsts = append(src.dsts, nil)
	}
	src.dsts[port] = dst
	for _, s := range dst.srcs {
		if s == src {
			return
		}
	}
	dst.srcs = append(dst.srcs, src)
}

// linkBack is like link but marks the connection as a back-edge so
// barycenter calculations can skip it (avoids layout divergence).
// It does NOT add src to dst.srcs, so the destination remains a layout root.
func linkBack(src *Node, port int, dst *Node) {
	for len(src.dsts) <= port {
		src.dsts = append(src.dsts, nil)
	}
	src.dsts[port] = dst
	// Do NOT append src to dst.srcs — back-edge sources must not make the
	// destination appear as a non-root in the layout algorithm.
	if src.backPort == nil {
		src.backPort = map[int]bool{}
	}
	src.backPort[port] = true
	if dst.backSrc == nil {
		dst.backSrc = map[*Node]bool{}
	}
	dst.backSrc[src] = true
}

// ─── Step 3: Build node layers (with dummy nodes for multi-layer edges) ───────

type pendingEdge struct {
	src  *Node
	port int
	dst  string // destination blockID
}

func buildNodeLayers(g *Graph) [][]*Node {
	// Determine max layer
	maxLayer := 0
	for _, b := range g.blocks {
		if b.layer > maxLayer {
			maxLayer = b.layer
		}
	}

	// Group blocks by layer, preserving insertion order within each layer
	byLayer := make([][]*Block, maxLayer+1)
	for _, id := range g.order {
		b := g.blocks[id]
		if b.layer >= 0 {
			byLayer[b.layer] = append(byLayer[b.layer], b)
		}
	}

	layers := make([][]*Node, maxLayer+1)
	nodeMap := map[string]*Node{} // blockID → its Node
	nid := 0

	active := []pendingEdge{} // edges whose destination is not yet on a layer

	for li := 0; li <= maxLayer; li++ {
		blocks := byLayer[li]

		// Separate active edges: terminating here vs. passing through
		var term, cont []pendingEdge
		for _, e := range active {
			found := false
			for _, b := range blocks {
				if b.id == e.dst {
					found = true
					break
				}
			}
			if found {
				term = append(term, e)
			} else {
				cont = append(cont, e)
			}
		}
		active = cont

		// Create one dummy per distinct destination for continuing edges
		dummyMap := map[string]*Node{}
		for i := range active {
			e := &active[i]
			dm, ok := dummyMap[e.dst]
			if !ok {
				dm = &Node{nid: nid, dstBlk: e.dst}
				nid++
				layers[li] = append(layers[li], dm)
				dummyMap[e.dst] = dm
			}
			link(e.src, e.port, dm)
			// Update the pending edge so it continues from this dummy
			e.src = dm
			e.port = 0
		}

		// Create real nodes for each block on this layer
		for _, b := range blocks {
			node := &Node{
				nid:     nid,
				blockID: b.id,
				w:       blockW,
				h:       blockH,
			}
			nid++
			// Connect terminating edges
			for _, e := range term {
				if e.dst == b.id {
					link(e.src, e.port, node)
				}
			}
			layers[li] = append(layers[li], node)
			nodeMap[b.id] = node

			// Queue outgoing edges, skipping all back-edges (handled separately)
			if !b.isBackedge {
				for port, sID := range b.succs {
					if g.skipForLayout(b.id, sID) {
						continue // back-edge: wired up after all nodes exist
					}
					s := g.blocks[sID]
					if s.layer <= li {
						continue // already placed (cycle guard)
					}
					active = append(active, pendingEdge{src: node, port: port, dst: sID})
				}
			}
		}
	}

	// Wire up all back-edges directly (both explicit :::backedge nodes and auto-detected)
	for _, b := range g.blocks {
		// Explicit backedge node → its loop header
		if b.isBackedge && len(b.succs) > 0 {
			src := nodeMap[b.id]
			dst := nodeMap[b.succs[0]]
			if src != nil && dst != nil {
				linkBack(src, 0, dst)
			}
		}
		// Auto-detected back-edges: from→to pairs in g.backEdges
		for _, sID := range b.succs {
			if g.backEdges[[2]string{b.id, sID}] {
				src := nodeMap[b.id]
				dst := nodeMap[sID]
				if src != nil && dst != nil {
					port := len(src.dsts) // use next available port slot
					linkBack(src, port, dst)
				}
			}
		}
	}

	return layers
}

// ─── Step 4: Compute x/y positions ───────────────────────────────────────────

func nodeCenter(n *Node) float64 { return n.x + n.w/2 }

func barycenterOfSrcs(n *Node) float64 {
	count, sum := 0, 0.0
	for _, s := range n.srcs {
		if n.backSrc != nil && n.backSrc[s] {
			continue // skip back-edge sources
		}
		sum += nodeCenter(s)
		count++
	}
	if count == 0 {
		return -1
	}
	return sum / float64(count)
}

func barycenterOfDsts(n *Node) float64 {
	count, sum := 0, 0.0
	for port, d := range n.dsts {
		if d == nil {
			continue
		}
		if n.backPort != nil && n.backPort[port] {
			continue // skip back-edge destinations
		}
		sum += nodeCenter(d)
		count++
	}
	if count == 0 {
		return -1
	}
	return sum / float64(count)
}

// resolveLayer enforces minimum spacing and ensures no node goes left of canvasPadX.
func resolveLayer(nodes []*Node, minGap float64) {
	if len(nodes) == 0 {
		return
	}
	// Clamp leftmost node to canvasPadX (never go left of margin)
	if nodes[0].x < canvasPadX {
		nodes[0].x = canvasPadX
	}
	// Left-to-right: push right to fix overlaps
	for i := 1; i < len(nodes); i++ {
		p, c := nodes[i-1], nodes[i]
		minX := p.x + p.w + minGap
		if c.x < minX {
			c.x = minX
		}
	}
	// Right-to-left: pull left where there's room (only decrease x, never below canvasPadX)
	for i := len(nodes) - 2; i >= 0; i-- {
		p, next := nodes[i], nodes[i+1]
		maxX := next.x - p.w - minGap
		if p.x > maxX {
			p.x = maxX
		}
		if p.x < canvasPadX {
			p.x = canvasPadX
		}
	}
}

// ─── Joint (horizontal segment) track assignment ─────────────────────────────
//
// For each layer, edges that span more than one column need a horizontal
// segment (a "joint") at some y between the source and destination layers.
// If two joints overlap horizontally they would visually cross; we assign
// them to separate vertical tracks to eliminate that.
//
// This mirrors the finagleJoints() algorithm in the original TypeScript source.

const jointSpacing = 14.0 // vertical distance between tracks (px)

type joint struct {
	x1, x2  float64
	src     *Node
	srcPort int
}

// assignJointOffsets computes per-port joint offsets for every node and
// returns a slice of extra vertical space required for each layer gap.
func assignJointOffsets(layers [][]*Node) []float64 {
	trackHeights := make([]float64, len(layers))

	for li, nodes := range layers {
		// Collect all joints for this layer (edges with a horizontal segment).
		var joints []joint
		for _, n := range nodes {
			n.jointOffset = make([]float64, len(n.dsts))
			for port, dst := range n.dsts {
				if dst == nil {
					continue
				}
				if n.backPort != nil && n.backPort[port] {
					continue // back-edges routed separately
				}
				x1 := portX(n, port)
				x2 := dstPortX(dst)
				if math.Abs(x2-x1) < 2*arrowR {
					continue // straight vertical, no horizontal segment needed
				}
				joints = append(joints, joint{x1: x1, x2: x2, src: n, srcPort: port})
			}
		}

		// Sort joints left-to-right by their leftmost x coordinate.
		sort.Slice(joints, func(a, b int) bool {
			return math.Min(joints[a].x1, joints[a].x2) < math.Min(joints[b].x1, joints[b].x2)
		})

		// Assign joints to tracks. Rightward and leftward joints get separate
		// track sets. Within each set: walk innermost-first; place in the
		// innermost track that doesn't overlap, or start a new outer track.
		var rightTracks, leftTracks [][]joint

		for _, jt := range joints {
			trackSet := &rightTracks
			if jt.x2 < jt.x1 {
				trackSet = &leftTracks
			}
			al := math.Min(jt.x1, jt.x2)
			ar := math.Max(jt.x1, jt.x2)

			placed := false
			var lastValid int = -1
			for ti := len(*trackSet) - 1; ti >= 0; ti-- {
				track := (*trackSet)[ti]
				overlaps := false
				for _, other := range track {
					if jt.src == other.src && jt.srcPort == other.srcPort {
						// same edge (shouldn't happen but guard)
						continue
					}
					bl := math.Min(other.x1, other.x2)
					br := math.Max(other.x1, other.x2)
					if ar >= bl && al <= br {
						overlaps = true
						break
					}
				}
				if overlaps {
					break
				}
				lastValid = ti
			}
			if lastValid >= 0 {
				(*trackSet)[lastValid] = append((*trackSet)[lastValid], jt)
				placed = true
			}
			if !placed {
				*trackSet = append(*trackSet, []joint{jt})
			}
		}

		// Compute total height needed for all tracks.
		nTracks := len(rightTracks) + len(leftTracks)
		totalH := 0.0
		if nTracks > 1 {
			totalH = float64(nTracks-1) * jointSpacing
		}
		trackHeights[li] = totalH

		// Assign offsets: right tracks get negative offsets (closer to source),
		// left tracks get positive offsets (farther from source), centered.
		// We spread them symmetrically around the midpoint of the gap.
		allTracks := append(rightTracks, leftTracks...)
		offset := -totalH / 2
		for _, track := range allTracks {
			for _, jt := range track {
				for len(jt.src.jointOffset) <= jt.srcPort {
					jt.src.jointOffset = append(jt.src.jointOffset, 0)
				}
				jt.src.jointOffset[jt.srcPort] = offset
			}
			offset += jointSpacing
		}
	}

	return trackHeights
}

// portX returns the x coordinate of a source port on node n.
func portX(n *Node, port int) float64 {
	if n.dummy() {
		return n.x + n.w/2
	}
	total := 0
	for _, d := range n.dsts {
		if d != nil {
			total++
		}
	}
	if total <= 1 {
		return n.x + n.w/2
	}
	active := 0
	for pp, d := range n.dsts {
		if d != nil && pp < port {
			active++
		}
	}
	spread := n.w * 0.6
	step := spread / float64(total-1)
	return n.x + n.w*0.2 + float64(active)*step
}

// dstPortX returns the x coordinate where an edge arrives at node dst.
func dstPortX(dst *Node) float64 {
	return dst.x + dst.w/2
}

func computePositions(g *Graph, layers [][]*Node) {
	// First pass: assign x positions (barycenter iterations) with fixed gap.
	// We need x before we can compute joint offsets.
	y := canvasPadY
	for _, nodes := range layers {
		for _, n := range nodes {
			n.y = y
			if n.dummy() {
				n.w = 0
				n.h = 0
			}
		}
		y += blockH + layerGapY
	}

	// Initial placement: pack evenly left-to-right from canvasPadX
	for _, nodes := range layers {
		x := canvasPadX
		for _, n := range nodes {
			n.x = x
			x += n.w + blockGapX
		}
	}

	// Iterative refinement: alternate top-down and bottom-up barycenter
	for iter := 0; iter < 12; iter++ {
		for li := 1; li < len(layers); li++ {
			for _, n := range layers[li] {
				bc := barycenterOfSrcs(n)
				if bc >= 0 {
					n.x = bc - n.w/2
				}
			}
			resolveLayer(layers[li], blockGapX)
		}
		for li := len(layers) - 2; li >= 0; li-- {
			for _, n := range layers[li] {
				if len(n.srcs) == 0 {
					continue
				}
				bc := barycenterOfDsts(n)
				if bc >= 0 {
					n.x = bc - n.w/2
				}
			}
			resolveLayer(layers[li], blockGapX)
		}
	}

	// Final pass: center root nodes over their children.
	for _, nodes := range layers {
		for _, n := range nodes {
			if len(n.srcs) == 0 {
				bc := barycenterOfDsts(n)
				if bc >= 0 {
					n.x = bc - n.w/2
					if n.x < canvasPadX {
						n.x = canvasPadX
					}
				}
			}
		}
	}

	// Now that x positions are final, compute joint offsets and track heights.
	trackHeights := assignJointOffsets(layers)

	// Re-assign y positions using dynamic per-layer gaps.
	y = canvasPadY
	for li, nodes := range layers {
		for _, n := range nodes {
			n.y = y
		}
		gap := layerGapY + trackHeights[li]
		y += blockH + gap
	}
}

// ─── Step 5: Render SVG ───────────────────────────────────────────────────────

func renderSVG(g *Graph, layers [][]*Node, pseudoRoots map[string]bool) string {
	// Canvas bounds
	maxX, maxY := 0.0, 0.0
	for _, nodes := range layers {
		for _, n := range nodes {
			if r := n.x + n.w; r > maxX {
				maxX = r
			}
			if b := n.y + n.h; b > maxY {
				maxY = b
			}
		}
	}
	W := maxX + canvasPadX
	H := maxY + canvasPadY

	var sb strings.Builder
	p := func(f string, a ...interface{}) { fmt.Fprintf(&sb, f, a...) }

	p(`<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">
<defs>
  <style>
    .blk  { fill:#fff;    stroke:#444; stroke-width:1.5 }
    .lhdr { fill:#d4f4dd; stroke:#2a7a2a; stroke-width:2 }
    .bkdg { fill:#fff3cd; stroke:#7a5a00; stroke-width:1.5 }
    .lbl  { font-family:ui-monospace,SFMono-Regular,monospace;
            font-size:12px; dominant-baseline:middle; text-anchor:middle; fill:#111 }
    .sid  { font-family:ui-monospace,SFMono-Regular,monospace;
            font-size:9px;  dominant-baseline:middle; text-anchor:middle; fill:#888 }
    .fwd  { fill:none; stroke:#444; stroke-width:1.5 }
    .bck  { fill:none; stroke:#b45309; stroke-width:1.5; stroke-dasharray:6 3 }
  </style>
  <marker id="ah"  markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0,8 3,0 6" fill="#444"/>
  </marker>
  <marker id="ahb" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
    <polygon points="0 0,8 3,0 6" fill="#b45309"/>
  </marker>
</defs>
<rect width="%.0f" height="%.0f" fill="#f6f6f6"/>
`, W, H, W, H, W, H)

	// Helper: is a block→block edge a back-edge?
	edgeIsBack := func(srcBlkID, dstBlkID string) bool {
		if srcBlkID == "" || dstBlkID == "" {
			return false
		}
		return g.isBackEdge(srcBlkID, dstBlkID)
	}

	// ── Edges ──
	// Multi-layer edges pass through dummy nodes. We stitch each chain of
	// dummies into a single SVG path so only one arrowhead appears at the end.
	// We only start rendering from real (non-dummy) source nodes; edges that
	// originate from a dummy are part of a chain already handled upstream.
	p("\n<!-- Edges -->\n")
	for _, nodes := range layers {
		for _, n := range nodes {
			if n.dummy() {
				continue // handled as part of a chain starting from a real node
			}
			for port, dst := range n.dsts {
				if dst == nil {
					continue
				}

				srcBlkID := n.blockID
				// Walk the dummy chain to find the final real destination.
				finalDst := dst
				for finalDst.dummy() {
					if len(finalDst.dsts) > 0 && finalDst.dsts[0] != nil {
						finalDst = finalDst.dsts[0]
					} else {
						break
					}
				}
				finalDstID := finalDst.blockID
				if finalDstID == "" {
					finalDstID = finalDst.dstBlk
				}
				be := (n.backPort != nil && n.backPort[port]) || edgeIsBack(srcBlkID, finalDstID)

				cls, mkr := "fwd", "url(#ah)"
				if be {
					cls, mkr = "bck", "url(#ahb)"
				}

				if be {
					x1 := n.x
					y1 := n.y + n.h/2
					x2 := finalDst.x
					y2 := finalDst.y + finalDst.h/2
					drawBack(p, cls, mkr, x1, y1, x2, y2)
					continue
				}

				// Build the sequence of (x,y) waypoints by following the dummy chain.
				// Each waypoint is the exit point of a node (bottom-center for real,
				// center for dummy) and the entry of the next.
				type waypoint struct{ x, y, ym float64 }
				var segments []waypoint

				cur := n
				curPort := port
				for {
					x1 := portX(cur, curPort)
					y1 := cur.y + cur.h
					next := cur.dsts[curPort]
					if next == nil {
						break
					}
					x2 := next.x + next.w/2
					y2 := next.y

					ymBase := y1 + (y2-y1)/2
					if cur.jointOffset != nil && curPort < len(cur.jointOffset) {
						ymBase = y1 + (y2-y1)/2 + cur.jointOffset[curPort]
					}
					segments = append(segments, waypoint{x1, y1, ymBase})
					// Also record the destination x for the horizontal segment landing
					_ = x2

					if next.dummy() {
						cur = next
						curPort = 0
					} else {
						break
					}
				}

				if len(segments) == 0 {
					continue
				}

				// Re-walk to collect full seg info (x2, y2 per hop).
				var segs []segStep
				cur = n
				curPort = port
				for {
					x1 := portX(cur, curPort)
					y1 := cur.y + cur.h
					next := cur.dsts[curPort]
					if next == nil {
						break
					}
					x2 := next.x + next.w/2
					y2 := next.y

					ymBase := y1 + (y2-y1)/2
					if cur.jointOffset != nil && curPort < len(cur.jointOffset) {
						ymBase = y1 + (y2-y1)/2 + cur.jointOffset[curPort]
					}
					segs = append(segs, segStep{x1, y1, x2, y2, ymBase})

					if next.dummy() {
						cur = next
						curPort = 0
					} else {
						break
					}
				}

				// Emit a single compound path through all segments.
				drawChain(p, cls, mkr, segs)
			}
		}
	}

	// ── Nodes ──
	p("\n<!-- Nodes -->\n")
	for _, nodes := range layers {
		for _, n := range nodes {
			if n.dummy() {
				continue
			}
			b := g.blocks[n.blockID]
			cls := "blk"
			if b.isLoopHeader && !pseudoRoots[b.id] {
				cls = "lhdr"
			} else if b.isBackedge {
				cls = "bkdg"
			}
			x, y, w, h := n.x, n.y, n.w, n.h
			p(`<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="5" class="%s"/>`,
				x, y, w, h, cls)
			showID := b.id != b.label
			ly := y + h/2
			if showID {
				ly = y + h/2 - 6
			}
			p(`<text x="%.2f" y="%.2f" class="lbl">%s</text>`, x+w/2, ly, xmlEsc(b.label))
			if showID {
				p(`<text x="%.2f" y="%.2f" class="sid">%s</text>`, x+w/2, y+h-8, xmlEsc(b.id))
			}
			p("\n")
		}
	}
	p("</svg>\n")
	return sb.String()
}

// segStep describes one layer-to-layer hop of a forward edge.
type segStep struct{ x1, y1, x2, y2, ym float64 }
// Each seg describes one layer-to-layer hop with its own joint y (ym).
// The arrowhead marker is placed only at the very end.
func drawChain(p func(string, ...interface{}), cls, mkr string, segs []segStep) {
	if len(segs) == 0 {
		return
	}
	r := arrowR

	var sb strings.Builder
	wp := func(f string, a ...interface{}) { fmt.Fprintf(&sb, f, a...) }

	first := segs[0]
	wp("M %.2f %.2f ", first.x1, first.y1)

	for i, s := range segs {
		dx := s.x2 - s.x1
		isLast := i == len(segs)-1
		y2e := s.y2
		if isLast {
			y2e = s.y2 - arrowTip
		}

		if math.Abs(dx) < 2*r {
			// Straight vertical segment — no horizontal jog.
			wp("L %.2f %.2f ", s.x2, y2e)
			continue
		}

		dir := 1.0
		if dx < 0 {
			dir = -1
		}
		s1, s2 := 0, 1
		if dir < 0 {
			s1, s2 = 1, 0
		}

		wp("L %.2f %.2f A %.2f %.2f 0 0 %d %.2f %.2f L %.2f %.2f A %.2f %.2f 0 0 %d %.2f %.2f L %.2f %.2f ",
			s.x1, s.ym-r,
			r, r, s1, s.x1+r*dir, s.ym,
			s.x2-r*dir, s.ym,
			r, r, s2, s.x2, s.ym+r,
			s.x2, y2e)
	}

	p(`<path d="%s" class="%s" marker-end="%s"/>`, strings.TrimSpace(sb.String()), cls, mkr)
}

// drawDown draws a single-hop forward edge with orthogonal H/V segments and
// rounded corners. ym is the y coordinate of the horizontal segment (joint).
func drawDown(p func(string, ...interface{}), cls, mkr string, x1, y1, x2, y2, ym float64) {
	r := arrowR
	y2e := y2 - arrowTip
	dx := x2 - x1

	if math.Abs(dx) < 2*r {
		// Straight or near-straight vertical
		p(`<path d="M %.2f %.2f L %.2f %.2f" class="%s" marker-end="%s"/>`,
			x1, y1, x2, y2e, cls, mkr)
		return
	}

	dir := 1.0
	if dx < 0 {
		dir = -1
	}

	// sweep=0 → CCW, sweep=1 → CW
	// Going right: down→right corner is CW (sweep=0 in SVG convention for this orientation)
	//              right→down corner is CCW (sweep=1)
	// Going left:  reversed
	s1, s2 := 0, 1
	if dir < 0 {
		s1, s2 = 1, 0
	}

	p(`<path d="M %.2f %.2f L %.2f %.2f A %.2f %.2f 0 0 %d %.2f %.2f L %.2f %.2f A %.2f %.2f 0 0 %d %.2f %.2f L %.2f %.2f" class="%s" marker-end="%s"/>`,
		x1, y1,
		x1, ym-r,
		r, r, s1, x1+r*dir, ym,
		x2-r*dir, ym,
		r, r, s2, x2, ym+r,
		x2, y2e,
		cls, mkr)
}

// drawBack draws a backward (upward) edge with orthogonal H/V segments and
// rounded corners, matching the visual style of drawDown.
// Exits the LEFT side of the source (mid-height), routes left to a margin
// column, goes up, and enters the LEFT side of the destination (mid-height).
//
// path: (x1,y1) →left→ (col,y1) →up→ (col,y2) →right→ (x2,y2)
// corner 1: left→up   sweep=1 (CW)
// corner 2: up→right  sweep=1 (CW)
func drawBack(p func(string, ...interface{}), cls, mkr string, x1, y1, x2, y2 float64) {
	r := arrowR
	// Margin column: to the left of both source and destination left edges.
	// Clamped to at least 20px so it stays within the SVG canvas.
	col := math.Min(x1, x2) - 50
	if col < 20 {
		col = 20
	}

	// Arrowhead enters from the left, pointing right: stop arrowTip short of x2
	x2tip := x2 - arrowTip

	p(`<path d="M %.2f %.2f `+
		`L %.2f %.2f `+                       // left toward column
		`A %.2f %.2f 0 0 1 %.2f %.2f `+       // corner 1: left→up (CW, sweep=1)
		`L %.2f %.2f `+                       // up toward dest
		`A %.2f %.2f 0 0 1 %.2f %.2f `+       // corner 2: up→right (CW, sweep=1)
		`L %.2f %.2f" `+                      // right to dest left edge
		`class="%s" marker-end="%s"/>`,
		x1, y1,
		col+r, y1,
		r, r, col, y1-r,
		col, y2+r,
		r, r, col+r, y2,
		x2tip, y2,
		cls, mkr)
}

func xmlEsc(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

// ─── Entry point ──────────────────────────────────────────────────────────────

func main() {
	var inFile, outFile string
	switch len(os.Args) - 1 {
	case 0:
	case 1:
		inFile = os.Args[1]
	case 2:
		inFile, outFile = os.Args[1], os.Args[2]
	default:
		fmt.Fprintln(os.Stderr, "Usage: iongraph-go [input.mmd] [output.svg]")
		os.Exit(1)
	}

	in := os.Stdin
	if inFile != "" {
		f, err := os.Open(inFile)
		must(err)
		defer f.Close()
		in = f
	}
	out := os.Stdout
	if outFile != "" {
		f, err := os.Create(outFile)
		must(err)
		defer f.Close()
		out = f
	}

	g := parseMermaid(bufio.NewScanner(in))
	g.autoDetectLoops()
	g.assignLoopInfo()

	// Find layout roots (no predecessors)
	roots := []string{}
	for _, id := range g.order {
		if len(g.blocks[id].preds) == 0 {
			roots = append(roots, id)
		}
	}
	if len(roots) == 0 && len(g.order) > 0 {
		roots = []string{g.order[0]}
	}

	// Assign layers. Roots are simply started at layer 0.
	// We do NOT make them pseudo loop-headers; the deferral mechanism only
	// activates for real loop headers.
	pseudoRoots := map[string]bool{} // for styling only; stays empty
	for _, id := range roots {
		g.assignLayers(id, 0)
	}
	// Fallback: any block still unassigned gets layer 0
	for _, b := range g.blocks {
		if b.layer < 0 {
			b.layer = 0
		}
	}

	// Sort within each layer by insertion order for stable output
	orderIdx := map[string]int{}
	for i, id := range g.order {
		orderIdx[id] = i
	}
	layers := buildNodeLayers(g)
	for li := range layers {
		sort.SliceStable(layers[li], func(a, b int) bool {
			na, nb := layers[li][a], layers[li][b]
			ia := orderIdx[na.blockID]
			if na.dummy() {
				ia = orderIdx[na.dstBlk]
			}
			ib := orderIdx[nb.blockID]
			if nb.dummy() {
				ib = orderIdx[nb.dstBlk]
			}
			return ia < ib
		})
	}

	computePositions(g, layers)
	svg := renderSVG(g, layers, pseudoRoots)
	fmt.Fprint(out, svg)
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
