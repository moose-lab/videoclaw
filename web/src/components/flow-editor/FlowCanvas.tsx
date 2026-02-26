/**
 * Flow Canvas Component
 *
 * Main React Flow canvas for the visual DAG editor.
 */

"use client";

import { useCallback, useRef } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type NodeChange,
  type EdgeChange,
  type NodeTypes,
  BackgroundVariant,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { CustomNode } from "./CustomNode";
import {
  type FlowNode,
  type FlowEdge,
  type TaskType,
  TASK_TYPE_COLORS,
} from "./types";
import { generateStepId, reactFlowToFlowDef } from "./utils";
import type { FlowDef } from "./types";

// Register custom node types
const nodeTypes: NodeTypes = {
  flow: CustomNode,
};

interface FlowCanvasProps {
  initialNodes: FlowNode[];
  initialEdges: FlowEdge[];
  onChange?: (flow: FlowDef) => void;
  existingFlow?: FlowDef;
  onNodeSelect?: (nodeId: string | null) => void;
}

export function FlowCanvas({
  initialNodes,
  initialEdges,
  onChange,
  existingFlow,
  onNodeSelect,
}: FlowCanvasProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // Handle node changes
  const handleNodesChange = useCallback(
    (changes: NodeChange<FlowNode>[]) => {
      onNodesChange(changes);
      
      // Notify parent of changes
      setTimeout(() => {
        const flow = reactFlowToFlowDef(nodes, edges, existingFlow);
        onChange?.(flow);
      }, 0);
    },
    [onNodesChange, nodes, edges, existingFlow, onChange]
  );

  // Handle edge changes
  const handleEdgesChange = useCallback(
    (changes: EdgeChange<FlowEdge>[]) => {
      onEdgesChange(changes);
      
      // Notify parent of changes
      setTimeout(() => {
        const flow = reactFlowToFlowDef(nodes, edges, existingFlow);
        onChange?.(flow);
      }, 0);
    },
    [onEdgesChange, nodes, edges, existingFlow, onChange]
  );

  // Handle new connections
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge(params, eds));
      
      // Notify parent after edge is added
      setTimeout(() => {
        const flow = reactFlowToFlowDef(nodes, edges, existingFlow);
        onChange?.(flow);
      }, 0);
    },
    [setEdges, nodes, edges, existingFlow, onChange]
  );

  // Handle node click
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: FlowNode) => {
      onNodeSelect?.(node.id);
    },
    [onNodeSelect]
  );

  // Handle pane click (deselect)
  const onPaneClick = useCallback(() => {
    onNodeSelect?.(null);
  }, [onNodeSelect]);

  // Handle node deletion
  const onNodesDelete = useCallback(
    (deleted: FlowNode[]) => {
      setNodes((nds) => nds.filter((n) => !deleted.find((d) => d.id === n.id)));
      
      setTimeout(() => {
        const flow = reactFlowToFlowDef(
          nodes.filter((n) => !deleted.find((d) => d.id === n.id)),
          edges.filter((e) => 
            !deleted.find((d) => d.id === e.source || d.id === e.target)
          ),
          existingFlow
        );
        onChange?.(flow);
      }, 0);
    },
    [setNodes, nodes, edges, existingFlow, onChange]
  );

  // Handle edge deletion
  const onEdgesDelete = useCallback(
    (deleted: FlowEdge[]) => {
      setEdges((eds) => eds.filter((e) => !deleted.find((d) => d.id === e.id)));
      
      setTimeout(() => {
        const flow = reactFlowToFlowDef(nodes, edges.filter((e) => 
          !deleted.find((d) => d.id === e.id)
        ), existingFlow);
        onChange?.(flow);
      }, 0);
    },
    [setEdges, nodes, edges, existingFlow, onChange]
  );

  // MiniMap node color
  const nodeColor = (node: FlowNode) => TASK_TYPE_COLORS[node.data.type];

  return (
    <div ref={reactFlowWrapper} className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onNodesDelete={onNodesDelete}
        onEdgesDelete={onEdgesDelete}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        deleteKeyCode={["Backspace", "Delete"]}
        className="bg-muted/30"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
        <Controls className="!bg-background !border !border-border !shadow-lg" />
        <MiniMap
          nodeColor={nodeColor}
          className="!bg-background !border !border-border"
          maskColor="rgba(0, 0, 0, 0.1)"
        />
        <Panel position="top-left" className="!m-0">
          <div className="bg-background/80 backdrop-blur-sm border border-border rounded-lg px-3 py-2 text-xs text-muted-foreground">
            Drag nodes from the left panel • Click to select • Delete to remove
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
}

// Export for external use
export { type FlowNode, type FlowEdge };
