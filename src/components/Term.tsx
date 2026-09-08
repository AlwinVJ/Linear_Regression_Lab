import type { ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export const GLOSSARY = {
  model: {
    label: "Model",
    text: "The rule the line follows: ŷ = β₀ + β₁x. It turns an input into a prediction.",
  },
  parameters: {
    label: "Parameters",
    text: "The two numbers β₀ and β₁ that define the line. Learning means choosing them.",
  },
  prediction: {
    label: "Prediction",
    text: "ŷ — what the model says the output should be for a given input.",
  },
  error: {
    label: "Error",
    text: "y − ŷ — the gap between what actually happened and what the model predicted. Also called the residual.",
  },
  loss: {
    label: "Loss",
    text: "MSE — one number summarising all the squared errors. Lower is better.",
  },
} as const;

export function Term({
  name,
  children,
}: {
  name: keyof typeof GLOSSARY;
  children?: ReactNode;
}) {
  const entry = GLOSSARY[name];
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="cursor-help underline decoration-dotted underline-offset-4">
          {children ?? entry.label}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64 text-pretty">{entry.text}</TooltipContent>
    </Tooltip>
  );
}
