import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import subprocess

def execute_and_convert(input_file, output_format='html'):
    with open(input_file) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})

    with open(input_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    subprocess.run(['jupyter', 'nbconvert', '--to', output_format, input_file])

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'html'
    execute_and_convert(input_file, output_format)

