export function ExplanationPanel({ message }: { message: string }) {
  return (
    <aside className="rounded-md border border-dashed border-axis/50 bg-muted/40 p-4">
      <h3 className="text-xs uppercase tracking-widest text-muted-foreground">
        What is happening?
      </h3>
      <p key={message} className="mt-2 text-sm leading-relaxed animate-in fade-in duration-300">
        {message}
      </p>
    </aside>
  );
}
