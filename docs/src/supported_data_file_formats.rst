.. _SupportedDataFileFormats:

Supported Data File Formats
===========================

Files containing NaN and/or non-numeric values can be uploaded for viewing but are not supported during training, evaluation, or prediction processes, and may result in errors. If a file does not contain a header row or has fewer than 2 columns or rows, it will not be uploaded. Currently, two file formats are supported:

.. list-table:: Supported Data File Formats
   :widths: 50 50 50
   :header-rows: 1

   * - Delimiter 
     - Decimal separator
     - Extension
   * - comma (,)
     - dot (.)
     - .csv
   * - semicolon (;)
     - dot (.)
     - .csv
