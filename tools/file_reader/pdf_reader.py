import json
from pathlib import Path
from typing import Optional, Dict, Any
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError


class PdfCreditReportParser:
    """
    Reads a potentially password-protected PDF credit report and parses it
    into a structured JSON format based on a predefined schema.

    Note: The parsing logic (_parse_content) is highly dependent on the
    specific format of the credit report and needs to be implemented
    accordingly. This class provides the basic structure for reading
    and extracting text.
    """

    def __init__(self, pdf_path: str | Path, password: Optional[str] = None):
        """
        Initializes the parser with the path to the PDF file and an optional password.

        Args:
            pdf_path: Path to the PDF file.
            password: Password for decrypting the PDF (if required).
                      This can be the user password (for opening) or the owner
                      password (for permissions like text extraction).
        """
        if PdfReader is None:
            raise ImportError("pypdf library is required but not installed.")

        self.pdf_path = Path(pdf_path)
        self.password = password
        self._text_content: Optional[str] = None
        self._parsed_data: Optional[Dict[str, Any]] = None

        if not self.pdf_path.exists() or not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

    def _load_and_extract_text(self) -> str:
        """Loads the PDF, handles decryption, and extracts text content."""
        try:
            reader = PdfReader(self.pdf_path)

            # Handle encryption
            if reader.is_encrypted:
                if self.password:
                    try:
                        decrypt_result = reader.decrypt(self.password)
                        # pypdf > 3.8 returns DecryptionResult enum
                        # Older versions returned an int status code (1 or 2 for success)
                        # Check type to be future-proof, fallback to checking value
                        if hasattr(decrypt_result, 'value'): # Check if it's an enum
                             if decrypt_result.value <= 0: # 0 = PasswordType.OWNER, -1 = PasswordType.USER -> These indicate failure in older versions semantics reversed now
                                 # Let's rely on exception being raised by pypdf itself if decryption fails
                                 pass # Assume success if no exception
                        elif decrypt_result <= 0: # Older pypdf check (where 1 or 2 meant success)
                             raise FileNotDecryptedError(f"Invalid password provided for {self.pdf_path}. (Legacy pypdf check)")
                        print(f"Successfully decrypted {self.pdf_path}")
                    except FileNotDecryptedError as e:
                        # pypdf should raise this if the password is wrong
                        print(f"Error decrypting PDF: Invalid password? {e}")
                        raise
                    except Exception as e: # Catch other potential pypdf errors during decryption
                        print(f"An unexpected error occurred during decryption: {e}")
                        raise
                else:
                    # If it's encrypted and no password given, text extraction will likely fail or return empty.
                    # We can raise here, or let it proceed and potentially fail during extraction.
                    # Let's raise early for clarity.
                     raise FileNotDecryptedError(
                        f"PDF file {self.pdf_path} is encrypted, but no password was provided."
                     )


            # Extract text from all pages
            text_parts = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                    else:
                        # Handle cases where text extraction might fail (e.g., image-based PDFs)
                        # Or if permissions restrict extraction even after opening.
                        print(f"Warning: No text extracted from page {i+1} of {self.pdf_path}. Page might be image-based, empty, or permissions restricted.")
                except Exception as e:
                    # Catch potential errors during text extraction
                    print(f"Error extracting text from page {i+1}: {e}")
                    # Decide if we should raise or continue. Continuing might yield partial results.
                    if i == 0:
                         raise RuntimeError(f"Failed to extract text from the first page. Check permissions or PDF integrity.") from e
                    else:
                         print(f"Warning: Skipping text extraction for page {i+1} due to error.")


            self._text_content = "\\n".join(text_parts)
            if not self._text_content:
                 print(f"Warning: No text could be extracted from the entire document: {self.pdf_path}")
            return self._text_content

        except FileNotDecryptedError as e:
            # Re-raise specifically handled decryption errors
            raise e
        except Exception as e:
            # Catch-all for other pypdf or file handling errors
            raise IOError(f"Failed to read or process PDF file {self.pdf_path}: {e}") from e

    def _parse_content(self, text: str) -> Dict[str, Any]:
        """
        Parses the extracted text into a structured dictionary based on a fixed schema.

        **This method needs to be implemented based on the specific format
        of the credit report PDF.**

        Args:
            text: The full text content extracted from the PDF.

        Returns:
            A dictionary representing the parsed data according to the defined schema.
        """
        # Placeholder implementation: Returns the raw text and a note
        # TODO: Replace this with actual parsing logic (e.g., using regex,
        #       keyword extraction, or more advanced NLP techniques).
        print("Parsing extracted text (using placeholder logic)...")

        # Example Schema (adjust as needed for your credit reports)
        parsed_schema = {
            "metadata": {
                "report_date": None, # Example: Extract from text using regex or keywords
                "source_file": str(self.pdf_path),
                "parser_status": "placeholder"
            },
            "personal_info": {
                "name": None,
                "addresses": [],
                "ssn_last4": None,
                # Add other fields as needed
            },
            "credit_score": {
                 "score": None,
                 "provider": None, # e.g., FICO, VantageScore
                 "bureau": None # e.g., Experian, Equifax, TransUnion
            },
            "accounts": [
                # Example structure for one account:
                # {
                #     "creditor": None,
                #     "account_number_masked": None,
                #     "type": None, # e.g., Revolving, Installment
                #     "open_date": None,
                #     "balance": None,
                #     "credit_limit": None,
                #     "payment_status": None, # e.g., Current, 30 days late
                #     "payment_history": None, # Could be a string or structured data
                # }
            ],
            "inquiries": [
                 # Example structure:
                 # { "date": None, "creditor": None, "type": None }
            ],
            "public_records": [
                 # Example structure:
                 # { "type": None, "date_filed": None, "details": None }
            ],
            # Keep a snippet of raw text for debugging or verification
            "raw_text_summary": text[:500] + "..." if text else ""
        }

        # --- Actual Parsing Logic Would Go Here ---
        # This section requires significant development based on report structure.
        # You might use:
        # - Regular expressions (import re)
        # - String searching (if keywords are consistent)
        # - More advanced table extraction if data is tabular
        # - Potentially external NLP libraries if structure is very complex

        # Example (very basic, non-functional placeholder):
        # import re
        # score_match = re.search(r"Your Credit Score:\s*(\d+)", text, re.IGNORECASE)
        # if score_match:
        #    parsed_schema["credit_score"]["score"] = int(score_match.group(1))

        print("Warning: PDF parsing logic is currently a placeholder. You need to implement '_parse_content'.")
        self._parsed_data = parsed_schema # Store the placeholder result
        return self._parsed_data


    def parse_to_json(self, output_json_path: Optional[str | Path] = None) -> str:
        """
        Orchestrates the PDF reading, text extraction, and parsing process.
        Optionally saves the resulting JSON to a file.

        Args:
            output_json_path: Optional path to save the output JSON file.
                              If None, the JSON string is returned but not saved.

        Returns:
            A JSON string representing the parsed credit report data.
        """
        try:
            if self._text_content is None:
                self._load_and_extract_text()

            # Check again if text content is available after attempting load/extract
            if self._text_content is None or not self._text_content.strip():
                 # Handle case where text extraction yielded nothing or only whitespace
                 print(f"Error: No usable text content could be extracted from {self.pdf_path}. Cannot parse.")
                 # Call parse with empty string to get base schema with status
                 empty_structure = self._parse_content("")
                 empty_structure["metadata"]["parser_status"] = "failed_no_text_extracted"
                 json_output = json.dumps(empty_structure, indent=4)
            else:
                # Proceed with parsing the extracted text
                parsed_data = self._parse_content(self._text_content)
                # Update status if parsing was just a placeholder
                if parsed_data.get("metadata", {}).get("parser_status") == "placeholder":
                     parsed_data["metadata"]["parser_status"] = "completed_placeholder_parsing"
                json_output = json.dumps(parsed_data, indent=4)

            if output_json_path:
                output_path = Path(output_json_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                print(f"Parsed data saved to: {output_path}")

            return json_output

        except (FileNotFoundError, FileNotDecryptedError, IOError, ImportError, RuntimeError) as e:
             print(f"Error during parsing process for {self.pdf_path}: {e}")
             # Optionally, create a minimal JSON indicating failure
             error_structure = {
                 "metadata": {
                     "source_file": str(self.pdf_path),
                     "parser_status": "failed",
                     "error_message": str(e),
                     "error_type": type(e).__name__
                 }
             }
             json_output = json.dumps(error_structure, indent=4)
             if output_json_path:
                output_path = Path(output_json_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                print(f"Error state saved to: {output_path}")
             # Re-raise the exception after logging/saving state if needed,
             # or return the error JSON string
             # raise e
             return json_output


# Example Usage Block (keep commented out or configure for actual use)
if __name__ == "__main__":
    # --- Configuration ---
    # Replace with the actual path to your test PDF file
    # pdf_file_path = "path/to/your/credit_report_20211107.pdf"
    # Replace with the actual password if the PDF is encrypted, otherwise set to None
    # pdf_password = "your_pdf_password" # or None
    # Specify where to save the output JSON file
    # output_file_path = "parsed_credit_report.json"

    # --- Execution ---
    # print(f"Attempting to parse: {pdf_file_path}")
    # try:
    #     # Ensure pypdf is installed: uv pip install pypdf
    #     parser = PdfCreditReportParser(pdf_file_path, password=pdf_password)
    #     json_result = parser.parse_to_json(output_json_path=output_file_path)
    #
    #     print("\\n--- Parsing Complete ---")
    #     # print("Full JSON output saved to file (if path provided).")
    #     # print("Preview of JSON result:")
    #     # print(json_result[:1000] + ("..." if len(json_result) > 1000 else ""))
    #
    # except ImportError as e:
    #      print(f"Import Error: {e}. Did you install pypdf (`uv pip install pypdf`)?")
    # except FileNotFoundError as e:
    #      print(f"File Not Found Error: {e}")
    # except FileNotDecryptedError as e:
    #      print(f"Decryption Error: {e}. Check the password or PDF encryption settings.")
    # except IOError as e:
    #      print(f"IO Error reading PDF: {e}")
    # except RuntimeError as e:
    #      print(f"Runtime Error during processing: {e}")
    # except Exception as e:
    #      print(f"An unexpected error occurred: {type(e).__name__} - {e}")

    print("Example usage section is commented out. Uncomment and configure paths/password to run.")
    pass # Keep the file valid Python even when example is commented out
