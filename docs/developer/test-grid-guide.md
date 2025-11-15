---
title: Grid-Guide Interactive Test
description: Test page for manual graph layout refinement with grid-guide
---

# Grid-Guide Interactive Test

This page demonstrates the grid-guide extension for manual refinement of knowledge graph layouts.

## Instructions

1. **Run fCoSE Layout** - Click to apply initial force-directed layout with constraints
2. **Toggle Grid** - Show/hide 50px grid overlay
3. **Enable Snap** - Nodes snap to grid when dragged
4. **Drag Nodes** - Manually refine positions (alignment guides appear)
5. **Align Selected** - Select multiple nodes (Shift+Click), then click to align
6. **Save Positions** - Export refined positions as JSON

<div style="margin: 20px 0; padding: 15px; background: #e7f3ff; border-left: 4px solid #4A90E2; border-radius: 3px;">
<strong>Try this workflow:</strong> Run fCoSE → Toggle Grid → Enable Snap → Drag nodes to refine → Save Positions
</div>

## Controls

<div style="margin: 20px 0;">
<button id="btn-fcose" style="background: #4A90E2; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-right: 10px;">▶ Run fCoSE Layout</button>
<button id="btn-toggle-grid" style="background: #4A90E2; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-right: 10px;">🔲 Toggle Grid</button>
<button id="btn-snap" style="background: #4A90E2; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-right: 10px;">📌 Enable Snap-to-Grid</button>
<button id="btn-align" style="background: #4A90E2; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-right: 10px;">⬉ Align Selected (Top-Left)</button>
<button id="btn-save" style="background: #4A90E2; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-right: 10px;">💾 Save Positions</button>
<button id="btn-load" style="background: #51C27F; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer;">📥 Reload Saved Positions</button>
</div>

## Graph

<div id="test-graph" style="width: 100%; height: 600px; border: 2px solid #ddd; border-radius: 5px; background: white;"></div>

<div id="status" style="margin-top: 15px; padding: 10px; background: #e7f3ff; border-left: 4px solid #4A90E2; border-radius: 3px;">
<strong>Status:</strong> Ready. Click "Run fCoSE Layout" to start.
</div>

<div id="saved-output" style="display: none; margin-top: 15px; padding: 10px; background: #f8f8f8; border-left: 4px solid #51C27F; border-radius: 3px;">
<strong>Saved Positions:</strong>
<pre id="positions-json" style="background: white; padding: 10px; border-radius: 3px; overflow-x: auto; font-size: 12px;"></pre>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
  // Initialize Cytoscape instance
  const cy = cytoscape({
    container: document.getElementById('test-graph'),

    elements: [
      // Nodes
      { data: { id: 'core', label: 'Core Concepts' } },
      { data: { id: 'impl', label: 'Implementations' } },
      { data: { id: 'utils', label: 'Utilities' } },
      { data: { id: 'parser', label: 'Parser' } },
      { data: { id: 'validator', label: 'Validator' } },
      { data: { id: 'transformer', label: 'Transformer' } },
      { data: { id: 'file-util', label: 'File Utils' } },
      { data: { id: 'string-util', label: 'String Utils' } },

      // Edges
      { data: { source: 'impl', target: 'core' } },
      { data: { source: 'parser', target: 'impl' } },
      { data: { source: 'validator', target: 'impl' } },
      { data: { source: 'transformer', target: 'impl' } },
      { data: { source: 'file-util', target: 'utils' } },
      { data: { source: 'string-util', target: 'utils' } },
      { data: { source: 'impl', target: 'utils' } },
      { data: { source: 'parser', target: 'string-util' } }
    ],

    style: [
      {
        selector: 'node',
        style: {
          'label': 'data(label)',
          'background-color': '#4A90E2',
          'color': '#fff',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '14px',
          'width': 100,
          'height': 100,
          'border-width': 3,
          'border-color': '#2E6DA4'
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 3,
          'line-color': '#999',
          'target-arrow-color': '#999',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier'
        }
      },
      {
        selector: ':selected',
        style: {
          'background-color': '#E94B3C',
          'border-color': '#C62F21',
          'line-color': '#E94B3C',
          'target-arrow-color': '#E94B3C',
          'border-width': 5
        }
      }
    ],

    layout: {
      name: 'random'
    }
  });

  // State
  let gridEnabled = false;
  let snapEnabled = false;
  const statusEl = document.getElementById('status');
  const savedOutputEl = document.getElementById('saved-output');
  const positionsJsonEl = document.getElementById('positions-json');

  // Initialize grid-guide (disabled initially)
  cy.gridGuide({
    snapToGridOnRelease: false,
    snapToGridDuringDrag: false,
    gridSpacing: 50,
    drawGrid: false,
    gridColor: '#ddd',
    lineWidth: 1,
    zoomDash: true,

    geometricGuideline: true,
    distributionGuidelines: true,
    guidelinesTolerance: 3.0,
    centerToEdgeAlignment: true,

    strokeStyle: '#4A90E2',
    horizontalDistColor: '#E94B3C',
    verticalDistColor: '#51C27F'
  });

  // Button: Run fCoSE Layout
  document.getElementById('btn-fcose').addEventListener('click', () => {
    statusEl.innerHTML = '<strong>Status:</strong> Running fCoSE layout...';

    cy.layout({
      name: 'fcose',
      quality: 'default',
      animate: true,
      animationDuration: 1000,

      // Fix core node at top
      fixedNodeConstraint: [
        { nodeId: 'core', position: { x: 0, y: -200 } }
      ],

      // Align implementation modules vertically
      alignmentConstraint: {
        vertical: [
          ['parser', 'validator', 'transformer'],
          ['file-util', 'string-util']
        ]
      },

      // LLM-friendly spacing
      nodeSeparation: 120,
      idealEdgeLength: 80,
      gravity: 0.8,
      gravityRange: 5.0,

      fit: true,
      padding: 50
    }).run();

    setTimeout(() => {
      statusEl.innerHTML = '<strong>Status:</strong> ✓ fCoSE layout complete. Toggle grid and try dragging nodes!';
    }, 1100);
  });

  // Button: Toggle Grid
  document.getElementById('btn-toggle-grid').addEventListener('click', () => {
    gridEnabled = !gridEnabled;
    cy.gridGuide({ drawGrid: gridEnabled });
    statusEl.innerHTML = `<strong>Status:</strong> Grid ${gridEnabled ? '<span style="color: #51C27F;">ENABLED</span>' : 'DISABLED'}. ${gridEnabled ? 'Grid provides 50px alignment reference.' : ''}`;
  });

  // Button: Enable/Disable Snap
  document.getElementById('btn-snap').addEventListener('click', () => {
    snapEnabled = !snapEnabled;
    cy.gridGuide({
      snapToGridOnRelease: snapEnabled,
      snapToGridDuringDrag: false
    });
    statusEl.innerHTML = `<strong>Status:</strong> Snap-to-grid ${snapEnabled ? '<span style="color: #51C27F;">ENABLED</span>' : 'DISABLED'}. ${snapEnabled ? 'Drag nodes - they will snap to grid!' : ''}`;
    document.getElementById('btn-snap').textContent = snapEnabled ? '✓ Snap Enabled' : '📌 Enable Snap-to-Grid';
  });

  // Button: Align Selected
  document.getElementById('btn-align').addEventListener('click', () => {
    const selected = cy.nodes(':selected');
    if (selected.length === 0) {
      statusEl.innerHTML = '<strong>Status:</strong> ⚠ No nodes selected. Hold Shift and click nodes to select multiple.';
      return;
    }
    selected.align('top', 'left');
    statusEl.innerHTML = `<strong>Status:</strong> ✓ Aligned ${selected.length} node(s) to top-left.`;
  });

  // Check for saved positions on page load
  const storageKey = 'test-grid-guide-positions';
  let savedPositions = null;

  // Try to load from localStorage on startup
  try {
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      savedPositions = JSON.parse(stored);
      statusEl.innerHTML = '<strong>Status:</strong> Found saved positions from previous session! Click "Reload Saved Positions" to apply them.';
    }
  } catch (e) {
    console.warn('Could not load saved positions:', e);
  }

  // Button: Save Positions
  document.getElementById('btn-save').addEventListener('click', () => {
    const positions = {};
    cy.nodes().forEach(node => {
      positions[node.id()] = {
        x: Math.round(node.position().x),
        y: Math.round(node.position().y)
      };
    });

    savedPositions = positions;

    // Save to localStorage (persists across page refreshes)
    localStorage.setItem(storageKey, JSON.stringify(positions));

    positionsJsonEl.textContent = JSON.stringify(positions, null, 2);
    savedOutputEl.style.display = 'block';
    statusEl.innerHTML = '<strong>Status:</strong> ✓ Positions saved to browser storage! Refresh the page and click "Reload" to test persistence.';

    // Scroll to output
    savedOutputEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });

  // Button: Reload Saved Positions
  document.getElementById('btn-load').addEventListener('click', () => {
    if (!savedPositions) {
      statusEl.innerHTML = '<strong>Status:</strong> ⚠ No saved positions yet. Click "Save Positions" first!';
      return;
    }

    // Apply saved positions
    cy.nodes().forEach(node => {
      if (savedPositions[node.id()]) {
        node.position(savedPositions[node.id()]);
      }
    });

    // Fit viewport
    cy.fit(null, 40);

    statusEl.innerHTML = '<strong>Status:</strong> ✓ Reloaded saved positions! Graph now shows your refined layout.';
  });

  console.log('✓ Grid-Guide test loaded. Extensions available:', {
    fcose: cy.layout({ name: 'fcose' }).hasCompoundNodes !== undefined,
    gridGuide: typeof cy.gridGuide === 'function'
  });
});
</script>

## What to Observe

### When you run fCoSE:
- "Core Concepts" node fixed at top (constraint working)
- Parser/Validator/Transformer aligned vertically (alignment constraint)
- Nodes spread with 120px separation (LLM-friendly spacing)

### When you toggle grid:
- 50px grid overlay appears
- Grid scales with zoom (try mouse wheel)

### When you enable snap:
- Drag any node - it snaps to nearest grid intersection
- Watch alignment guidelines appear (blue lines show alignment with other nodes)

### When you drag nodes (snap enabled):
- **Blue lines** = geometric alignment guides (node centers/edges align)
- **Red/green lines** = distribution guides (even spacing detected)

### When you save:
- JSON output shows rounded x,y coordinates
- Use this JSON in production docs with `layout: "preset"`

## Next Steps

After testing, see the [Advanced Graph Layouts](../../core_knowledge_graph/.claude/skills/md-to-mkdocs/resources/advanced_graph_layouts.md) documentation for complete guidance on creating LLM-optimal knowledge graphs.
