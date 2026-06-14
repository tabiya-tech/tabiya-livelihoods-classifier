import type { HTMLAttributes, ReactNode } from "react";
import { useState } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "69e59609-0d9a-4928-80d6-9c727815def9";

export const DATA_TEST_ID = {
  CONTAINER: `code-block-container-${uniqueId}`,
  COPY_BUTTON: `code-block-copy-button-${uniqueId}`,
};

export interface CodeBlockProps extends HTMLAttributes<HTMLPreElement> {
  /** Dim the background — used for response examples in Docs. */
  muted?: boolean;
  /** Show a copy button in the top-right that copies the rendered text. */
  copyable?: boolean;
  /** The text content. Children are rendered if provided; otherwise `code` is shown. */
  code?: string;
  children?: ReactNode;
  /** Max height before the block scrolls vertically. */
  maxHeight?: number | string;
}

export function CodeBlock({
  muted,
  copyable,
  code,
  children,
  className,
  maxHeight,
  ...rest
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const text = code ?? (typeof children === "string" ? children : "");

  async function onCopy() {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // navigator.clipboard may be unavailable in some test envs; swallow.
    }
  }

  return (
    <pre
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("code-block", muted && "muted", className)}
      style={maxHeight ? { maxHeight, overflowY: "auto" } : undefined}
      {...rest}
    >
      {copyable && (
        <div className="absolute right-2.5 top-2.5 flex gap-1.5">
          <button
            type="button"
            data-testid={DATA_TEST_ID.COPY_BUTTON}
            onClick={onCopy}
            aria-label={copied ? "Copied" : "Copy code"}
            className={mergeClassNames(
              "rounded-sm border border-white/10 bg-white/10 px-2 py-0.5",
              "font-mono text-[10px] text-white/70 hover:bg-white/20 hover:text-white",
            )}
          >
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      )}
      {children ?? code}
    </pre>
  );
}
