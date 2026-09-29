content = open('feature_selection_corrected.py', encoding='utf-8').read()
content = content.replace('\\\"\\\"\\\";', '\"\"\"').replace('\\\"\\\"\\\"', '\"\"\"')
open('feature_selection_corrected.py', 'w', encoding='utf-8').write(content)
import ast
try:
    ast.parse(content)
    print('Syntax OK')
except SyntaxError as e:
    print(f'SyntaxError: {e}')
