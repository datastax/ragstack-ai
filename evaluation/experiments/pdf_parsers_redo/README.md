# Which PDF Parser works the best?

## Standard Parsers

The following PDF Parsers were used to embed the test datasets into unique AstraDB collections via the `embedding_standard.py` script.

### PyPDFium2Loader

pypdfium2 is an ABI-level Python 3 binding to PDFium, a powerful and liberal-licensed library for PDF rendering, inspection, manipulation and creation.

* https://github.com/pypdfium2-team/pypdfium2
* ☆ 231
* https://pdfium.googlesource.com/pdfium/

### PDFMinerLoader

Pdfminer.six is a community maintained fork of the original PDFMiner. It is a tool for extracting information from PDF documents. It focuses on getting and analyzing text data. Pdfminer.six extracts the text from a page directly from the sourcecode of the PDF. It can also be used to get the exact location, font or color of the text.

* https://github.com/pdfminer/pdfminer.six
* ☆ 5.1k

### PyPDFLoader

A pure-python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files

* https://github.com/py-pdf/pypdf
* ☆ 7.0k

### PyMuPDFLoader

A high performance Python library for data extraction, analysis, conversion & manipulation of PDF (and other) documents.

* https://github.com/pymupdf/PyMuPDF
* ☆ 3.6k

## Modern Parsers

### LayoutPDFReader

Most PDF to text parsers do not provide layout information. Often times, even the sentences are split with arbitrary CR/LFs making it very difficult to find paragraph boundaries. This poses various challenges in chunking and adding long running contextual information such as section header to the passages while indexing/vectorizing PDFs for LLM applications such as retrieval augmented generation (RAG).

LayoutPDFReader solves this problem by parsing PDFs along with hierarchical layout information such as:

Sections and subsections along with their levels.
Paragraphs - combines lines.
Links between sections and paragraphs.
Tables along with the section the tables are found in.
Lists and nested lists.
Join content spread across pages.
Removal of repeating headers and footers.
Watermark removal.
With LayoutPDFReader, developers can find optimal chunks of text to vectorize, and a solution for limited context window sizes of LLMs.

* https://github.com/nlmatics/llmsherpa
* ☆ 586

The `embedding_llmsherpa.py` script was used to chunk, embed, and store the test datasets for the LayoutPDFReader parser.

### Unstructured

The `unstructured` library provides open-source components for ingesting and pre-processing images and text documents, such as PDFs, HTML, Word docs, and many more. The use cases of unstructured revolve around streamlining and optimizing the data processing workflow for LLMs. unstructured modular functions and connectors form a cohesive system that simplifies data ingestion and pre-processing, making it adaptable to different platforms and efficient in transforming unstructured data into structured outputs.

* https://github.com/Unstructured-IO/unstructured
* ☆ 4.6k

The `embedding_unstructured.py` script was used to chunk, embed, and store the test datasets for the LayoutPDFReader parser.
