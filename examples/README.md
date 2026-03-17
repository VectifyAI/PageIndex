# PageIndex Examples

This directory contains example scripts demonstrating how to use PageIndex with different file types.

## Process Text File

The `process_text_file.py` script demonstrates how to process plain text files (.txt) with automatic semantic section detection.

### Features Demonstrated

- Windowing approach with overlap for handling long documents
- Automatic semantic heading/subheading detection using LLM
- Hierarchical tree structure generation
- Summary generation for each section
- Document description generation

### Usage

1. Make sure you have set up your environment:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```bash
   CHATGPT_API_KEY=your_key_here
   ```

3. Run the example:
   ```bash
   cd examples
   python3 process_text_file.py
   ```

### Output

The script will:
1. Process the sample text file using windowing with overlap
2. Detect semantic sections automatically
3. Build a hierarchical tree structure
4. Generate summaries for each section
5. Display the table of contents and structure
6. Save results to `results/sample_structure.json`

### Customization

You can customize the processing by modifying these parameters in the script:

- `WINDOW_SIZE`: Size of each text window in characters (default: 5000)
- `OVERLAP`: Number of characters to overlap between windows (default: 500)
- `MODEL`: OpenAI model to use (default: 'gpt-4o-2024-11-20')
- `summary_token_threshold`: Threshold for summary generation (default: 200)

## Command Line Usage

You can also use the main runner script to process text files:

```bash
python3 run_pageindex.py --txt_path /path/to/your/file.txt
```

Optional arguments:
```bash
--window-size WINDOW_SIZE    # Window size in characters (default: 5000)
--overlap OVERLAP            # Overlap size in characters (default: 500)
--if-add-node-summary yes    # Generate summaries (default: yes)
--if-add-doc-description yes # Generate document description (default: no)
```

See the main README for more details.
