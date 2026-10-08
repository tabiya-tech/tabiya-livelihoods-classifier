import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { DATA_TEST_ID, UploadDropzone } from "./UploadDropzone";

/**
 * jsdom's File doesn't implement `.text()`; stub it per file.
 */
function makeReadableTextFile(content: string, filename: string): File {
  const file = new File([content], filename, { type: "text/plain" });
  Object.defineProperty(file, "text", {
    configurable: true,
    value: () => Promise.resolve(content),
  });
  return file;
}

describe("UploadDropzone", () => {
  it("delivers the file's text to onText when the user picks a file", async () => {
    // GIVEN an onText spy and a sample .txt file
    const onText = vi.fn();
    const givenFileContent = "We are hiring a data scientist.";
    const givenFile = makeReadableTextFile(givenFileContent, "ad.txt");

    // WHEN we render and the user picks the file
    render(<UploadDropzone onText={onText} />);
    const fileInput = screen.getByTestId(
      DATA_TEST_ID.FILE_INPUT,
    ) as HTMLInputElement;
    await userEvent.upload(fileInput, givenFile);

    // THEN onText is called with the file content + name
    await waitFor(() => expect(onText).toHaveBeenCalledTimes(1));
    expect(onText).toHaveBeenCalledWith(givenFileContent, "ad.txt");
  });

  it("rejects files larger than maxBytes with a localized error", async () => {
    // GIVEN a 1KB limit and a 2KB file
    const givenMaxBytes = 1024;
    const givenLargeFile = new File(["x".repeat(2048)], "big.txt", {
      type: "text/plain",
    });
    const expectedErrorText = i18n.t("classifier.upload.errorTooLarge", {
      maxKb: 1,
    });

    // WHEN we render with the small cap and the user picks the file
    const onText = vi.fn();
    render(<UploadDropzone onText={onText} maxBytes={givenMaxBytes} />);
    const fileInput = screen.getByTestId(
      DATA_TEST_ID.FILE_INPUT,
    ) as HTMLInputElement;
    await userEvent.upload(fileInput, givenLargeFile);

    // THEN onText is not called and the error message is shown
    expect(onText).not.toHaveBeenCalled();
    expect(screen.getByTestId(DATA_TEST_ID.ERROR)).toHaveTextContent(
      expectedErrorText,
    );
  });

  it("does not open the file dialog when disabled", async () => {
    // GIVEN an onText spy and a disabled dropzone
    const onText = vi.fn();

    // WHEN we render and the user clicks the container
    render(<UploadDropzone onText={onText} disabled />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN no error appears, no callback fires, and the input is disabled
    expect(onText).not.toHaveBeenCalled();
    expect(screen.queryByTestId(DATA_TEST_ID.ERROR)).not.toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.FILE_INPUT)).toBeDisabled();
  });
});
