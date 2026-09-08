import {
  Children,
  cloneElement,
  isValidElement,
  useId,
  type HTMLAttributes,
  type ReactElement,
  type ReactNode,
} from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import { Label } from "@/components";

const uniqueId = "3ff67c65-2da4-4ead-bbd5-61904d9b827f";

export const DATA_TEST_ID = {
  CONTAINER: `form-field-container-${uniqueId}`,
  HELP: `form-field-help-${uniqueId}`,
  ERROR: `form-field-error-${uniqueId}`,
};

export interface FormFieldProps extends HTMLAttributes<HTMLDivElement> {
  /** Visible label. Omit only for unlabeled inline fields. */
  label?: ReactNode;
  /** Mark the field required. */
  required?: boolean;
  /** Helper text rendered below the control. */
  help?: ReactNode;
  /** Error message — when set, the field renders in error state. */
  error?: ReactNode;
  /** The form control. Must accept id, aria-describedby, and aria-invalid props. */
  children: ReactNode;
}

export function FormField({
  label,
  required,
  help,
  error,
  className,
  children,
  ...rest
}: FormFieldProps) {
  const fieldId = useId();
  const helpId = `${fieldId}-help`;
  const errorId = `${fieldId}-error`;
  const describedBy =
    [error ? errorId : null, help ? helpId : null].filter(Boolean).join(" ") ||
    undefined;

  // Inject id + describedBy + invalid into the single child control if it's an element.
  const enhancedChild =
    isValidElement(children) && Children.count(children) === 1
      ? cloneElement(children as ReactElement, {
          id: (children as ReactElement).props.id ?? fieldId,
          "aria-describedby": describedBy,
          invalid: error
            ? true
            : (children as ReactElement).props.invalid,
        })
      : children;

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("w-full", className)}
      {...rest}
    >
      {label && (
        <Label htmlFor={fieldId} required={required}>
          {label}
        </Label>
      )}
      {enhancedChild}
      {error ? (
        <p
          id={errorId}
          data-testid={DATA_TEST_ID.ERROR}
          role="alert"
          className="mt-1.5 text-xs leading-snug text-error"
        >
          {error}
        </p>
      ) : help ? (
        <p
          id={helpId}
          data-testid={DATA_TEST_ID.HELP}
          className="mt-1.5 text-xs leading-snug text-muted"
        >
          {help}
        </p>
      ) : null}
    </div>
  );
}
