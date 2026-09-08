import { Plus, RotateCcw, Trash2 } from "lucide-react";
import type { Point } from "@/algorithms/linearRegression";
import { Button } from "@/components/ui/button";

interface Props {
  points: Point[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onChange: (id: string, key: "x" | "y", value: number) => void;
  onAdd: () => void;
  onDelete: (id: string) => void;
  onReset: () => void;
}

export function DatasetEditor({
  points,
  selectedId,
  onSelect,
  onChange,
  onAdd,
  onDelete,
  onReset,
}: Props) {
  return (
    <div>
      <div className="max-h-64 overflow-y-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-muted text-left">
            <tr>
              <th className="px-3 py-2 font-medium">x</th>
              <th className="px-3 py-2 font-medium">y</th>
              <th className="w-9" />
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr
                key={p.id}
                onMouseEnter={() => onSelect(p.id)}
                className={selectedId === p.id ? "bg-accent/15" : undefined}
              >
                <td className="px-2 py-1">
                  <input
                    type="number"
                    step="any"
                    value={p.x}
                    aria-label="x value"
                    onChange={(e) => onChange(p.id, "x", Number(e.target.value))}
                    className="w-full rounded bg-transparent px-1 py-1 font-mono tabular-nums outline-none focus:bg-background focus:ring-1 focus:ring-ring"
                  />
                </td>
                <td className="px-2 py-1">
                  <input
                    type="number"
                    step="any"
                    value={p.y}
                    aria-label="y value"
                    onChange={(e) => onChange(p.id, "y", Number(e.target.value))}
                    className="w-full rounded bg-transparent px-1 py-1 font-mono tabular-nums outline-none focus:bg-background focus:ring-1 focus:ring-ring"
                  />
                </td>
                <td className="px-1">
                  <button
                    onClick={() => onDelete(p.id)}
                    aria-label="Delete point"
                    className="rounded p-1 text-muted-foreground transition-colors hover:text-destructive"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex gap-2">
        <Button variant="outline" size="sm" onClick={onAdd} className="flex-1">
          <Plus className="size-3.5" /> Add point
        </Button>
        <Button variant="ghost" size="sm" onClick={onReset} className="flex-1">
          <RotateCcw className="size-3.5" /> Reset dataset
        </Button>
      </div>
    </div>
  );
}
